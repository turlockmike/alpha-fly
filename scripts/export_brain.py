"""A drawable sample of the fly's neurons for the site: soma positions from the MaleCNS annotations, projected to 2D
(dorsal view: anterior to the left, so the brain sits left and the nerve cord runs right), for K neurons that have
a soma, plus a coarse class per neuron.  docs/model/brain.bin, little-endian:

    u32 K | u32 idx[K] (neuron index in the network) | i16 x[K] | i16 y[K] | i16 z[K] | u8 cls[K]
    x: anterior -> posterior, y: left -> right, z: dorsal -> ventral; centred, one scale for all three (+-32767 = the
    longest extent), so proportions are true
    cls: 0 central brain, 1 optic lobe, 2 nerve cord, 3 sensory (input), 4 descending / motor (output)

    uv run python scripts/export_brain.py docs/model/brain.bin 30000
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.feather as feather

from chessfly.connectome import load_malecns

dst, K = Path(sys.argv[1]), int(sys.argv[2]) if len(sys.argv) > 2 else 30000
conn = load_malecns("data", 3)
ann = feather.read_feather(Path("data") / "body-annotations.feather", columns=["bodyId", "somaLocation", "superclass"])
ann = ann.set_index("bodyId")
neurons = conn.neurons
loc = ann["somaLocation"].reindex(neurons["root_id"].to_numpy())
has = loc.notna().to_numpy()
pos = np.zeros((len(neurons), 3), dtype=np.float64)
pos[has] = np.stack([np.asarray(v, dtype=np.float64) for v in loc[has]])
sc = neurons["super_class"].to_numpy()
cls = np.zeros(len(neurons), dtype=np.uint8)
cls[np.char.startswith(sc.astype(str), "ol_") | np.isin(sc, ["visual_projection", "visual_centrifugal"])] = 1
cls[np.char.startswith(sc.astype(str), "vnc_") | np.isin(sc, ["ascending_neuron", "sensory_ascending"])] = 2
cls[np.isin(neurons["flow"].to_numpy(), ["afferent"])] = 3
cls[np.isin(sc, ["descending_neuron"]) | np.isin(neurons["flow"].to_numpy(), ["efferent"])] = 4

rng = np.random.default_rng(0)
cand = np.flatnonzero(has)
keep_all = cand[np.isin(cls[cand], [3, 4])]                     # every input and output neuron with a soma
rest = cand[~np.isin(cls[cand], [3, 4])]
idx = np.concatenate([keep_all, rng.choice(rest, size=max(0, K - len(keep_all)), replace=False)])
idx.sort()
p = pos[idx][:, [2, 0, 1]]                                        # anterior->posterior, left->right, dorsal->ventral
mid, span = (p.max(0) + p.min(0)) / 2, np.ptp(p, axis=0).max()
q = np.round((p - mid) / span * 2 * 32767).astype("<i2")
with open(dst, "wb") as f:
    f.write(np.array([len(idx)], dtype="<u4").tobytes())
    f.write(idx.astype("<u4").tobytes())
    for a in range(3): f.write(q[:, a].tobytes())
    f.write(cls[idx].tobytes())
print(f"{dst}: {len(idx):,} neurons ({len(keep_all):,} inputs/outputs), classes {np.bincount(cls[idx], minlength=5).tolist()}, "
      f"{dst.stat().st_size / 1024:.0f} KB; extents {np.ptp(p, axis=0).round(0).tolist()}")
