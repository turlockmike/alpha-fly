"""Hop distance from sensory to output neurons, and forward/backward throughput per kernel."""

import gc
import time

import torch

from chessfly.connectome import load_malecns
from chessfly.encoding import N_FEATURES
from chessfly.model import FlyChessNet, ModelConfig

conn = load_malecns("data")
print(conn.summary())

# ---- hops: fraction of output neurons reached after k synapses from any sensory neuron
net = FlyChessNet(conn, ModelConfig())
reached = torch.zeros(conn.n_neurons, dtype=torch.bool)
reached[net.in_idx] = True

dev = "cuda"


def bench(dtype, batch, steps, train=False, reps=5):
    gc.collect(); torch.cuda.empty_cache()
    net = FlyChessNet(conn, ModelConfig(steps=steps)).to(dev)
    net.infer_dtype = dtype
    x = torch.rand(batch, N_FEATURES, device=dev)
    torch.cuda.reset_peak_memory_stats()
    for i in range(reps + 1):
        if i == 1:
            torch.cuda.synchronize(); t = time.time()
        if train:
            p, v = net(x)
            (p.square().mean() + v.square().mean()).backward()
        else:
            with torch.no_grad():
                net(x)
    torch.cuda.synchronize()
    dt = (time.time() - t) / reps
    print(f"{'train' if train else 'infer'} {str(dtype)[6:]:8s} B={batch:4d} T={steps}: "
          f"{dt * 1000:7.1f} ms  {batch / dt:8.0f} pos/s  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")


for dtype in (torch.float32, torch.float16):
    for batch in (64, 256, 1024):
        bench(dtype, batch, steps=4)
for batch in (64, 128, 256):
    bench(torch.float32, batch, steps=4, train=True)
