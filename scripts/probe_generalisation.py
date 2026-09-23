"""Is a train/validation gap memorisation, or a train-mode / eval-mode mismatch of the readout normalisation?
    uv run python scripts/probe_generalisation.py runs/<name>"""
import pickle, random, sys
from pathlib import Path
import torch
from chessfly.connectome import load_malecns
from chessfly.model import FlyChessNet, ModelConfig
from chessfly.train import collate

run = Path(sys.argv[1])
ck = torch.load(run / "last.pt", map_location="cuda", weights_only=False)
net = FlyChessNet(load_malecns("data"), ModelConfig(**ck["config"])).cuda()
net.load_state_dict(ck["model"])
saved = pickle.loads((run / "buffer.pkl").read_bytes())
random.seed(0)
sets = {"train": random.sample(saved["buffer"], 1024), "val": saved["val"][:1024]}


@torch.no_grad()
def policy_loss(data, train_mode):
    net.train(train_mode)
    net.norm.momentum = 0.0                  # batch statistics without touching the running ones
    tot = 0.0
    for i in range(0, len(data), 128):
        x, pi, legal, *_ = collate(data[i:i + 128], "cuda")
        tot += -(pi * net(x)[0].float().masked_fill(~legal, -1e9).log_softmax(-1)).sum().item()
    return tot / len(data)


for name, data in sets.items():
    print(f"{name:5s} n={len(data)}  eval-mode {policy_loss(data, False):.3f}   batch-stat mode {policy_loss(data, True):.3f}")
rv = net.norm.var
print(f"running sd of output neurons: median {rv.sqrt().median():.2e}, min {rv.sqrt().min():.2e}")
