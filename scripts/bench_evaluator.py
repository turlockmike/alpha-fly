"""The evaluator exactly as self-play uses it (float16 features in, legal-logit gather, results to the CPU),
by batch size, with a breakdown of one forward pass.   uv run python scripts/bench_evaluator.py"""
import time, numpy as np, torch
from chessfly.connectome import load_malecns
from chessfly.encoding import N_FEATURES, N_MOVES
from chessfly.model import Evaluator, FlyChessNet, ModelConfig

net = FlyChessNet(load_malecns("data"), ModelConfig()).cuda().eval()
ev = Evaluator(net, torch.device("cuda"))
rng = np.random.default_rng(0)

def batch(B):
    x = (rng.random((B, N_FEATURES)) < 0.04).astype(np.float16)
    rows = np.repeat(np.arange(B), 25); cols = rng.integers(0, N_MOVES, size=B * 25)
    return x, rows, cols

def sync(): torch.cuda.synchronize()
for B in (256, 512, 530, 584, 768, 1024, 1536, 2048):
    req = batch(B); ev(*req); sync(); t = time.time()
    for _ in range(10): ev(*req)
    sync(); dt = (time.time() - t) / 10
    print(f"B={B:5d}: {dt * 1000:6.1f} ms  {B / dt:8.0f} evals/s", flush=True)

B = 530; x, rows, cols = batch(B)
xt = torch.from_numpy(x).cuda().float(); r, c = torch.from_numpy(rows).cuda(), torch.from_numpy(cols).cuda()
def timed(name, f, n=20):
    f(); sync(); t = time.time()
    for _ in range(n): out = f()
    sync(); print(f"  {name:34s} {(time.time() - t) / n * 1000:6.2f} ms"); return out
with torch.no_grad():
    print(f"breakdown at B={B}:")
    timed("host -> device + float", lambda: torch.from_numpy(x).cuda().float())
    enc = timed("encoder (838 -> 17,937)", lambda: net.encoder(xt) * net.cfg.input_gain)
    def drive():
        u = xt.new_zeros(net.brain.n, B, dtype=torch.float16); u[net.in_idx] = enc.to(torch.float16).T; return u
    u = timed("scatter into (N, B) drive", drive)
    h = timed("brain: 4 recurrent steps", lambda: net.brain(u.clone(), 4))
    w = net.brain.edge_weights().to(torch.float16)
    timed("  of which: edge_weights() + csr build", lambda: net.brain.csr(net.brain.edge_weights().to(torch.float16)))
    W = net.brain.csr(w)
    timed("  of which: one SpMM", lambda: W @ h)
    rd = timed("readout gather + norm", lambda: net.norm(h[net.out_idx].T.float()))
    lg = timed("policy + value heads", lambda: (net.policy(rd), net.value(rd)))
    timed("legal gather + to CPU", lambda: (lg[0][r, c].float().cpu(), lg[1].float().cpu()))
