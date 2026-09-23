"""Where GPU time goes at the run's settings: inference per light-cone step, and one training step phase by phase.
    uv run python scripts/profile_gpu.py [checkpoint]"""
import sys, time, random, pickle, numpy as np, torch
from chessfly.model import load_net
from chessfly.suite import load_suite
from chessfly.train import TrainConfig, make_optimizer, collate
from chessfly.selfplay import Sample

net = load_net(sys.argv[1] if len(sys.argv) > 1 else "runs/fly6_noadj/last.pt")
brain, steps = net.brain, net.cfg.steps
suite = load_suite(); xs = torch.from_numpy(suite.x).cuda()
def sync(): torch.cuda.synchronize()
def timed(f, n=10):
    f(); sync(); t = time.time()
    for _ in range(n): f()
    sync(); return (time.time() - t) / n * 1000

cone = brain.light_cone(net.in_idx, net.out_idx, steps)
print(f"steps {steps}; light-cone rows per step {[len(r) for r in cone.rows[1:]]}; sub-matrix non-zeros {[len(op.edges) if op is not None else 0 for op in cone.ops[1:]]}")
B = 624
x = xs[:B] if len(xs) >= B else xs.repeat(2, 1)[:B]
with torch.no_grad():
    net.eval()
    print(f"\ninference, B={B}, float16:  whole network {timed(lambda: net(x)):.1f} ms")
    u_in = (net.encoder(x) * net.cfg.input_gain).T.contiguous()
    brain.forward_io(u_in, net.in_idx, net.out_idx, steps)
    cs, ks, mats, alpha, _ = brain._cached[1]
    h = None
    for t in range(1, steps + 1):
        rows = cone.rows[t]
        if mats[t] is not None:
            hp = h
            ms_spmm = timed(lambda: mats[t] @ hp)
            y = mats[t] @ hp
        else:
            ms_spmm, y = 0.0, ks[t].expand(-1, B).clone()
        def rest():
            z = y.clone().add_(ks[t]).index_add_(0, cone.u_pos[t], u_in[cone.u_src[t]]).clamp_(0, brain.h_max)
            prev = cs[t - 1][rows].expand(-1, B)
            if len(cone.keep_next[t]) and h is not None: prev = prev.index_copy(0, cone.keep_next[t], h[cone.keep_prev[t]])
            return torch.lerp(prev, z, alpha[rows])
        ms_rest = timed(rest)
        h = rest()
        print(f"  step {t}: rows {len(rows):7,d}  nnz {0 if cone.ops[t] is None else len(cone.ops[t].edges):10,d}   SpMM {ms_spmm:6.2f} ms   everything else {ms_rest:6.2f} ms")

# ---- training step, phase by phase
net.train(); tcfg = TrainConfig(); opt = make_optimizer(net, tcfg)
rng = np.random.default_rng(0)
batch = [Sample(suite.x[i].astype(np.float16), suite.cols[suite.offsets[i]:suite.offsets[i + 1]].astype(np.int16),
                np.full(suite.offsets[i + 1] - suite.offsets[i], 1 / (suite.offsets[i + 1] - suite.offsets[i]), np.float16), 0.0, int(rng.integers(3)))
         for i in rng.integers(len(suite.x), size=256)]
import torch.nn.functional as F
def phase_times(n=6):
    acc = {}
    for k in range(n + 1):
        sync(); t = time.time(); xb, pi, legal, wdl, q = collate(batch, "cuda"); sync(); t_collate = time.time() - t
        t = time.time(); logits, wl = net(xb); loss = -(pi * logits.masked_fill(~legal, -1e9).log_softmax(-1)).sum(-1).mean() + F.cross_entropy(wl, wdl); sync(); t_fwd = time.time() - t
        t = time.time(); opt.zero_grad(set_to_none=True); loss.backward(); sync(); t_bwd = time.time() - t
        t = time.time(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); sync(); t_clip = time.time() - t
        t = time.time(); opt.step(); sync(); t_opt = time.time() - t
        if k: 
            for name, v in (("collate (CPU + transfer)", t_collate), ("forward", t_fwd), ("backward", t_bwd), ("gradient clipping", t_clip), ("optimiser step", t_opt)):
                acc[name] = acc.get(name, 0) + v / n
    return acc
acc = phase_times()
tot = sum(acc.values())
print(f"\none training step, batch 256, float32: {tot * 1000:.0f} ms = {256 / tot:,.0f} positions/s")
for k, v in acc.items(): print(f"  {k:26s} {v * 1000:7.1f} ms  {v / tot:5.0%}")
