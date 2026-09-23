"""The backward pass of the recurrent core, piece by piece, per light-cone step (B = 256)."""
import sys, time, torch
from chessfly.model import load_net

net = load_net(sys.argv[1] if len(sys.argv) > 1 else "runs/fly6_noadj/last.pt")
brain, steps, B = net.brain, net.cfg.steps, 256
cone = brain.light_cone(net.in_idx, net.out_idx, steps)
w_full = brain.edge_weights().detach()
def sync(): torch.cuda.synchronize()
def timed(f, n=5):
    f(); sync(); t = time.time()
    for _ in range(n): f()
    sync(); return (time.time() - t) / n * 1000

tot = {"spmm_T": 0, "sddmm": 0, "gather32": 0, "gather16": 0, "sddmm16": 0}
for t in range(2, steps + 1):
    op = cone.ops[t]; nnz = len(op.edges); rows, cols = op.shape
    w = w_full[op.edges].contiguous()
    h = torch.rand(cols, B, device="cuda"); g = torch.rand(rows, B, device="cuda")
    WT, pattern = op.csr_t(w), op.csr(torch.zeros_like(w))
    a = timed(lambda: WT @ g)
    def sddmm(dt=torch.float32, blk=64):
        out = torch.zeros(nnz, device="cuda", dtype=dt); pat = op.csr(torch.zeros(nnz, device="cuda", dtype=dt))
        for c in range(0, B, blk):
            out += torch.sparse.sampled_addmm(pat, g[:, c:c + blk].to(dt).contiguous(), h[:, c:c + blk].to(dt).T.contiguous(), beta=0.0).values()
        return out
    b = timed(sddmm)
    def gather(dt):
        hh, gg = h.to(dt), g.to(dt); out = torch.empty(nnz, device="cuda", dtype=torch.float32); chunk = max((1 << 26) // B, 1)
        for s in range(0, nnz, chunk):
            out[s:s + chunk] = (hh[op.pre[s:s + chunk]] * gg[op.post[s:s + chunk]]).sum(1, dtype=torch.float32)
        return out
    c32, c16 = timed(lambda: gather(torch.float32)), timed(lambda: gather(torch.float16))
    try: s16 = timed(lambda: sddmm(torch.float16))
    except Exception as e: s16 = float("nan")
    err = float((gather(torch.float32) - sddmm()).abs().max() / sddmm().abs().max())
    print(f"step {t}: {rows:7,d} x {cols:7,d}, nnz {nnz:10,d} | grad wrt state (SpMM^T) {a:6.1f} ms | per-synapse grad: SDDMM {b:6.1f} ms, SDDMM fp16 {s16:6.1f} ms, gather fp32 {c32:6.1f} ms, gather fp16 {c16:6.1f} ms (gather vs SDDMM rel. diff {err:.1e})")
    for k, v in (("spmm_T", a), ("sddmm", b), ("gather32", c32), ("gather16", c16), ("sddmm16", s16)): tot[k] += v
print("totals over the steps, ms:", {k: round(v, 1) for k, v in tot.items()})
# the constant trajectory: full-matrix ops at batch 1
h1 = torch.rand(brain.n, 1, device="cuda"); g1 = torch.rand(brain.n, 1, device="cuda")
W, WT = brain.csr(w_full), brain.csr_t(w_full); pat = brain.csr(torch.zeros_like(w_full))
print(f"constant trajectory, full matrix at batch 1: SpMV {timed(lambda: W @ h1):.1f} ms, SpMV^T {timed(lambda: WT @ g1):.1f} ms, "
      f"SDDMM {timed(lambda: torch.sparse.sampled_addmm(pat, g1, h1.T.contiguous(), beta=0.0)):.1f} ms, csr_t build (reorders 10.5M values) {timed(lambda: brain.csr_t(w_full)):.1f} ms, "
      f"edge_weights() {timed(lambda: brain.edge_weights()):.1f} ms  (each of these runs {steps - 1}x per training step)")
