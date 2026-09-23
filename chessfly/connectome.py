"""Load the Janelia MaleCNS v1.0 connectome (https://male-cns.janelia.org/download/).

Adapted from nfly (https://github.com/zhengxuyu/nfly, MIT).  Data is CC-BY 4.0, cite Berg et al., Cell 2026.

Files (public, Google Storage, Apache Arrow feather):
    body-annotations       one row per body (segment); neurons are the bodies with a superclass
    body-neurotransmitters per-body transmitter prediction (consensus_nt)
    connectome-weights     body -> body synapse counts (minconf >= 0.5), 152M rows incl. fragments

Mapping onto the tensors used by ConnectomeRNN:
    bodyId                           -> node index 0..N-1
    (body_pre, body_post, weight)    -> W[post, pre], magnitude = weight / total input of post
    consensus_nt of presynaptic body -> sign (Dale's law): ACh/DA/5-HT/OA +, GABA/Glu/His -
    superclass                       -> flow: afferent / intrinsic / efferent
"""

from __future__ import annotations

import dataclasses
import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch

log = logging.getLogger(__name__)

BASE_URL = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "body-annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

# Drosophila sign convention.  Histamine (photoreceptors) acts through HisCl chloride channels
# and glutamate mostly through GluCl, so both count as inhibitory.
SIGN = {"acetylcholine": 1.0, "dopamine": 1.0, "serotonin": 1.0, "octopamine": 1.0,
        "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}

# superclass -> flow.  Sensory neurons are the network's inputs; motor / efferent / endocrine
# are its outputs; everything else is hidden.
AFFERENT = {"ol_sensory", "cb_sensory", "vnc_sensory", "sensory_ascending", "sensory_descending",
            "cb_sensory_tbc", "vnc_sensory_tbc", "sensory_ascending_tbc"}
EFFERENT = {"vnc_motor", "cb_motor", "vnc_efferent", "cb_efferent", "vnc_endocrine", "cb_endocrine",
            "efferent_ascending", "efferent_descending"}
CACHE_VERSION = 2


@dataclasses.dataclass
class Connectome:
    neurons: pd.DataFrame     # index 0..N-1; columns root_id, super_class, cell_type, nt, flow
    pre: torch.Tensor         # (E,) int64  presynaptic node index
    post: torch.Tensor        # (E,) int64  postsynaptic node index
    syn_count: torch.Tensor   # (E,) float32
    sign: torch.Tensor        # (E,) float32 +1 / -1 from the presynaptic transmitter

    @property
    def n_neurons(self) -> int:
        return len(self.neurons)

    @property
    def n_edges(self) -> int:
        return int(self.pre.numel())

    @property
    def weight(self) -> torch.Tensor:
        """w_ij = syn_ij / sum_j syn_ij  (each postsynaptic neuron's inputs sum to 1)."""
        total = torch.zeros(self.n_neurons).index_add_(0, self.post, self.syn_count)
        return self.syn_count / total[self.post].clamp_min(1.0)

    def nodes(self, flow: str) -> torch.Tensor:
        return torch.as_tensor(np.flatnonzero((self.neurons["flow"] == flow).to_numpy()), dtype=torch.long)

    def nodes_of_superclass(self, *names: str) -> torch.Tensor:
        return torch.as_tensor(np.flatnonzero(self.neurons["super_class"].isin(names).to_numpy()), dtype=torch.long)

    def shuffled(self, seed: int = 0) -> "Connectome":
        """Control: same neurons, same in-degree and sign/count multiset, random presynaptic
        partners.  If the real wiring matters, this network should train worse."""
        g = torch.Generator().manual_seed(seed)
        pre = self.pre[torch.randperm(self.n_edges, generator=g)]
        order = torch.as_tensor(np.lexsort((pre.numpy(), self.post.numpy())))     # keep CSR order
        nt_sign = torch.ones(self.n_neurons).index_put_((self.pre,), self.sign)   # sign belongs to the pre neuron
        return Connectome(self.neurons, pre[order], self.post[order], self.syn_count[order], nt_sign[pre[order]])

    def summary(self) -> str:
        flows = self.neurons["flow"].value_counts().to_dict()
        return (f"{self.n_neurons:,} neurons, {self.n_edges:,} edges, "
                f"{float((self.sign > 0).float().mean()):.1%} excitatory, flow={flows}")


def download(data_dir: str | Path = "data") -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for local, remote in FILES.items():
        if not (data_dir / local).exists():
            log.info("downloading %s", remote)
            urllib.request.urlretrieve(f"{BASE_URL}/{remote}", data_dir / local)


def _read_neurons(data_dir: Path) -> pd.DataFrame:
    ann = feather.read_feather(data_dir / "body-annotations.feather")
    ann = ann[ann["superclass"].notna()]
    df = pd.DataFrame({"root_id": ann["bodyId"].astype(np.int64),
                       "super_class": ann["superclass"].astype(str),
                       "cell_type": ann["type"].astype(object).where(ann["type"].notna(), "").astype(str)})
    nt = feather.read_feather(data_dir / "body-neurotransmitters.feather", columns=["body", "consensus_nt"])
    df = df.merge(nt.rename(columns={"body": "root_id", "consensus_nt": "nt"}), on="root_id", how="left")
    df["nt"] = df["nt"].astype(object).where(df["nt"].notna(), "unclear").astype(str)
    df["flow"] = "intrinsic"
    df.loc[df["super_class"].isin(AFFERENT), "flow"] = "afferent"
    df.loc[df["super_class"].isin(EFFERENT), "flow"] = "efferent"
    return df.sort_values("root_id").reset_index(drop=True)


def _read_weights(data_dir: Path, keep_ids: np.ndarray, min_syn: int) -> pd.DataFrame:
    """The table has 152M rows including unannotated fragments, so it is filtered batch by
    batch in pyarrow before anything is materialised in pandas."""
    reader = pa.ipc.open_file(data_dir / "connectome-weights.feather")
    names = reader.schema.names
    pre, post, wt = [next(c for c in cands if c in names)
                     for cands in (("body_pre", "pre"), ("body_post", "post"), ("weight", "syn_count"))]
    ids = pa.array(keep_ids)
    parts = []
    for i in range(reader.num_record_batches):
        b = reader.get_batch(i).select([pre, post, wt])
        mask = pc.and_(pc.is_in(b.column(pre), value_set=ids), pc.is_in(b.column(post), value_set=ids))
        mask = pc.and_(mask, pc.greater_equal(b.column(wt), min_syn))
        b = b.filter(mask)
        if b.num_rows:
            parts.append(b)
    w = pa.Table.from_batches(parts).rename_columns(["pre", "post", "syn"]).to_pandas()
    return w.groupby(["pre", "post"], sort=False, as_index=False)["syn"].sum()


def load_malecns(data_dir: str | Path = "data", min_syn: int = 3, cache: bool = True) -> Connectome:
    """Parse the MaleCNS release (or a cached .pt) into a Connectome; `min_syn` drops weak pairs."""
    data_dir = Path(data_dir)
    cache_path = data_dir / "cache" / f"malecns_min{min_syn}_v{CACHE_VERSION}.pt"
    if cache and cache_path.exists():
        d = torch.load(cache_path, weights_only=False)
        return Connectome(**d)

    download(data_dir)
    neurons = _read_neurons(data_dir)
    w = _read_weights(data_dir, neurons["root_id"].to_numpy(), min_syn)
    lut = pd.Series(np.arange(len(neurons)), index=neurons["root_id"].to_numpy())
    pre = lut.loc[w["pre"].to_numpy()].to_numpy()
    post = lut.loc[w["post"].to_numpy()].to_numpy()
    sign = neurons["nt"].map(SIGN).fillna(1.0).to_numpy(dtype=np.float32)[pre]   # unknown -> excitatory

    # (post, pre) order is CSR order of W[post, pre], so edge tensors double as CSR values
    order = np.lexsort((pre, post))
    c = Connectome(neurons, torch.as_tensor(pre[order]), torch.as_tensor(post[order]),
                   torch.as_tensor(w["syn"].to_numpy(dtype=np.float32)[order]), torch.as_tensor(sign[order]))
    log.info(c.summary())
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({f.name: getattr(c, f.name) for f in dataclasses.fields(c)}, cache_path)
    return c
