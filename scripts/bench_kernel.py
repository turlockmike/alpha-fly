"""Micro-benchmark of one recurrent step W @ h: state layout and index width."""
import time, torch
from chessfly.connectome import load_malecns
from chessfly.brain import ConnectomeRNN

rnn = ConnectomeRNN(load_malecns("data")).cuda()
n = rnn.n

def timeit(f, reps=20):
    f(); torch.cuda.synchronize(); t = time.time()
    for _ in range(reps): f()
    torch.cuda.synchronize(); return (time.time() - t) / reps * 1000

for dtype in (torch.float16, torch.float32):
    w = rnn.edge_weights().to(dtype).detach()
    W64 = torch.sparse_csr_tensor(rnn.crow, rnn.pre, w, (n, n), check_invariants=False)
    W32 = torch.sparse_csr_tensor(rnn.crow.int(), rnn.pre.int(), w, (n, n), check_invariants=False)
    for B in (64, 256, 1024):
        h_bn = torch.rand(B, n, device="cuda", dtype=dtype)          # current layout: (B, N), used through .T
        h_nb = h_bn.T.contiguous()                                    # (N, B) contiguous
        print(f"{str(dtype)[6:]:8s} B={B:4d}  current (B,N).T int64: {timeit(lambda: W64 @ h_bn.T):7.2f} ms   "
              f"(N,B) contiguous int64: {timeit(lambda: W64 @ h_nb):7.2f} ms   "
              f"(N,B) contiguous int32: {timeit(lambda: W32 @ h_nb):7.2f} ms", flush=True)
