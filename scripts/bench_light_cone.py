"""Light-cone evaluation against the literal recurrence on a real checkpoint: agreement, then speed.
    uv run python scripts/bench_light_cone.py [checkpoint]"""
import sys, time, numpy as np, torch
from chessfly.model import load_net
from chessfly.suite import load_suite

net = load_net(sys.argv[1] if len(sys.argv) > 1 else "runs/fly/last.pt")
brain, steps = net.brain, net.cfg.steps
x = torch.from_numpy(load_suite().x[:512]).cuda()

def full(u_in):                                   # the old path: whole state, every step
    u = u_in.new_zeros(brain.n, u_in.shape[1]); u[net.in_idx] = u_in
    return brain(u, steps)[net.out_idx]

cone = brain.light_cone(net.in_idx, net.out_idx, steps)
print(f"steps {steps}: rows computed per position {[len(r) for r in cone.rows[1:]]} of {brain.n:,}; "
      f"synapse visits {cone.edge_visits:,} instead of {steps * brain.pre.numel():,} ({steps * brain.pre.numel() / cone.edge_visits:.1f}x fewer)")
with torch.no_grad():
    for dtype in (torch.float32, torch.float16):
        u_in = (net.encoder(x) * net.cfg.input_gain).to(dtype).T.contiguous()
        a, b = full(u_in.clone()).float(), brain.forward_io(u_in, net.in_idx, net.out_idx, steps).float()
        print(f"{str(dtype)[6:]}: max |difference| {float((a - b).abs().max()):.2e} on outputs of scale {float(a.abs().mean()):.3f}; "
              f"policy argmax agrees on {float((net.policy(net.norm(a.T)).argmax(-1) == net.policy(net.norm(b.T)).argmax(-1)).float().mean()):.1%}")

def timeit(f, n=10):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(n): f()
    torch.cuda.synchronize(); return (time.time() - t) / n

with torch.no_grad():
    u16 = (net.encoder(x) * net.cfg.input_gain).half().T.contiguous()
    t_old, t_new = timeit(lambda: full(u16.clone())), timeit(lambda: brain.forward_io(u16, net.in_idx, net.out_idx, steps))
    print(f"inference B=512 fp16 (GPU shared with the live run): full {t_old * 1000:.1f} ms, light cone {t_new * 1000:.1f} ms  ->  {t_old / t_new:.1f}x")
    for B in (512, 1024, 2048):
        xb = x.repeat(B // 512, 1); t = timeit(lambda: net(xb))
        print(f"  whole network, B={B}: {t * 1000:.1f} ms = {B / t:,.0f} evals/s")
net.train()
xb = x[:256]
def train_new():
    net.zero_grad(); p, v = net(xb); (p.square().mean() + v.square().mean()).backward()
def train_old():
    net.zero_grad(); u_in = (net.encoder(xb) * net.cfg.input_gain).T.contiguous()
    r = net.norm(full(u_in).T); (net.policy(r).square().mean() + net.value(r).square().mean()).backward()
t_old, t_new = timeit(train_old, 4), timeit(train_new, 4)
print(f"training B=256 fp32 forward+backward: full {t_old * 1000:.0f} ms ({256 / t_old:,.0f} pos/s), light cone {t_new * 1000:.0f} ms ({256 / t_new:,.0f} pos/s)  ->  {t_old / t_new:.1f}x")
