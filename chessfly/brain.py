"""Connectome-constrained rate RNN (model after nfly, MIT; kernels rewritten on cuSPARSE).

    h[t+1] = (1 - alpha) * h[t] + alpha * phi( W h[t] + b + u )

W is sparse with the fly's connectivity and fixed signs (Dale's law); every edge has a
learnable positive gain on top of the connectome-derived magnitude.  alpha (per-neuron leak)
and b (per-neuron bias) are also learnable.  Nothing can create or re-sign a synapse.

`forward` is that recurrence, literally.  `forward_io` computes the same output rows from the
same inputs while doing a fraction of the work (the "light cone", see LightCone); the network
is evaluated through it, and tests hold the two equal.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from .connectome import Connectome

warnings.filterwarnings("ignore", message="Sparse (CSR tensor support is in beta|invariant checks are implicitly)")

CHUNK_ELEMS = 1 << 26    # batch x edges per chunk in the per-synapse gradient (bounds transients)


def _crow(sorted_rows: torch.Tensor, n: int) -> torch.Tensor:
    crow = torch.zeros(n + 1, dtype=torch.long, device=sorted_rows.device)
    crow[1:] = torch.bincount(sorted_rows, minlength=n).cumsum(0)
    return crow


# One state update  h' = p + a * (clamp(y + k, 0, hmax) - p),  p = carried ? h_prev : c,  as a single CUDA kernel compiled by
# PyTorch's NVRTC jiterator (no compiler needed).  As separate torch ops this was four to five passes over a
# (neurons x batch) tensor and 38% of an inference; the passes are memory-bound, so one pass is 2.4x faster.
FUSED_UPDATE = FUSED_DEVIATION = None
try:
    from torch.cuda.jiterator import _create_jit_fn
    FUSED_UPDATE = _create_jit_fn("""
template <typename T> T fused_update(T y, T k, T h_prev, T carried, T c, T a, T hmax) {
    T x = y + k;
    x = x < T(0) ? T(0) : (x > hmax ? hmax : x);
    T p = carried > T(0) ? h_prev : c;
    return p + a * (x - p);
}""", hmax=10.0)
    # Half precision done right.  Plain float16 inference failed (see FlyChessNet.infer_dtype): a neuron's position-dependent
    # signal is ~1% of its resting activity, and float16 rounds it away.  So carry only the DEVIATION d = h - c from the
    # resting trajectory c (what the network does with no position) between steps: small numbers, which float16 holds to full
    # relative precision, and the SpMM is linear so  A h = A c + A d  with A c precomputed exactly.  The clamp is rewritten
    # so that the large constants cancel exactly:  clamp(k + s, 0, H) - clamp(k, 0, H) = clamp(s, -k, H - k) + (k - clamp(k, 0, H)),
    # where s = A d + u is small and the last term is exactly -lo or -hi on a clamped row and 0 elsewhere.
    FUSED_DEVIATION = _create_jit_fn("""
template <typename T> T fused_deviation(T s, T lo, T hi, T e, T d_prev, T carried, T a) {
    T x = (s < lo ? lo : (s > hi ? hi : s)) + e;
    T p = carried > T(0) ? d_prev : T(0);
    return p + a * (x - p);
}""")
except Exception:                                   # CPU-only builds
    FUSED_DEVIATION = None

SAMPLED_ADDMM = True     # per-synapse gradient through cuSPARSE SDDMM instead of chunked gathers
SDDMM_COLS = 64


class _SparseRecurrent(torch.autograd.Function):
    """y = W h with W in CSR and the state laid out (N, B) contiguous, so that the B values an edge
    touches are adjacent in memory (15x faster than a (B, N) state used through .T).  Forward and
    grad_h are SpMM; only the per-synapse gradient dL/dw_e = sum_b h[pre_e, b] * g[post_e, b]
    needs a gather, done in chunks."""

    @staticmethod
    def forward(ctx, h, w, rnn):
        ctx.save_for_backward(h, w)
        ctx.rnn = rnn
        return rnn.csr(w) @ h

    @staticmethod
    def backward(ctx, grad_y):
        h, w = ctx.saved_tensors
        rnn = ctx.rnn
        grad_y = grad_y.contiguous()
        grad_h = rnn.csr_t(w) @ grad_y if ctx.needs_input_grad[0] else None
        grad_w = None
        if ctx.needs_input_grad[1] and SAMPLED_ADDMM and grad_y.is_cuda:
            # (grad_y @ h^T) evaluated only where W has an entry: cuSPARSE SDDMM, values in CSR = edge order
            # It slows down superlinearly in the batch width (B=256 is 3x slower per position than B=64),
            # hence the column blocks.
            pattern, grad_w = rnn.csr(torch.zeros_like(w)), torch.zeros_like(w)
            for a in range(0, h.shape[1], SDDMM_COLS):
                grad_w += torch.sparse.sampled_addmm(pattern, grad_y[:, a:a + SDDMM_COLS].contiguous(),
                                                     h[:, a:a + SDDMM_COLS].T.contiguous(), beta=0.0).values()
        elif ctx.needs_input_grad[1]:
            grad_w = torch.empty_like(w)
            chunk = max(CHUNK_ELEMS // h.shape[1], 1)
            for a in range(0, w.numel(), chunk):
                b = a + chunk
                grad_w[a:b] = (h[rnn.pre[a:b]] * grad_y[rnn.post[a:b]]).sum(1)
        return grad_h, grad_w, None


class _SubMatrix:
    """Rows x columns of W as its own CSR, with the interface _SparseRecurrent needs.  Edge order is W's,
    restricted, so `edges` gathers its values from the full weight vector."""

    def __init__(self, rows: torch.Tensor, cols: torch.Tensor, pre: torch.Tensor, post: torch.Tensor, n: int):
        dev = pre.device
        row_of = torch.full((n,), -1, dtype=torch.long, device=dev)
        row_of[rows] = torch.arange(len(rows), device=dev)
        col_of = torch.full((n,), -1, dtype=torch.long, device=dev)
        col_of[cols] = torch.arange(len(cols), device=dev)
        self.edges = torch.nonzero((row_of[post] >= 0) & (col_of[pre] >= 0)).squeeze(1)
        self.post, self.pre = row_of[post[self.edges]], col_of[pre[self.edges]]          # local indices, still (post, pre) sorted
        self.shape = (len(rows), len(cols))
        self.crow = _crow(self.post, len(rows))
        self.t_order = torch.argsort(self.pre * len(rows) + self.post)
        self.t_crow, self.t_col = _crow(self.pre[self.t_order], len(cols)), self.post[self.t_order]

    def csr(self, w):
        return torch.sparse_csr_tensor(self.crow, self.pre, w, self.shape, check_invariants=False)

    def csr_t(self, w):
        return torch.sparse_csr_tensor(self.t_crow, self.t_col, w[self.t_order], self.shape[::-1], check_invariants=False)


class LightCone:
    """Which rows of the state have to be computed per position, and which never change.

    The state starts at zero and the position enters only through the input neurons, so after t
    steps only neurons within t-1 synapses of an input can differ between positions (V_t): the
    rest follow one trajectory c_t, the same for every position, computed once.  And only
    neurons within T-t synapses of an output can still matter at step t (D_t).  Per position the
    network therefore needs the rows R_t = V_t & D_t, a sub-matrix of W per step, and the rest
    of W enters as a constant drive.  For MaleCNS at 4 steps that is 3.0M synapse visits instead
    of 42M: step 1 needs none at all (W times a zero state).  Same function, same gradients."""

    def __init__(self, rnn: "ConnectomeRNN", in_idx: torch.Tensor, out_idx: torch.Tensor, steps: int):
        n, pre, post, dev = rnn.n, rnn.pre, rnn.post, rnn.pre.device
        mask = lambda idx: torch.zeros(n, dtype=torch.bool, device=dev).index_fill_(0, idx, True)
        varying = [torch.zeros(n, dtype=torch.bool, device=dev), mask(in_idx)]          # V_0, V_1
        for _ in range(2, steps + 1):
            v = varying[-1].clone()
            v[post[varying[-1][pre]]] = True
            varying.append(v)
        needed = [None] * steps + [mask(out_idx)]                                       # D_T ... D_0
        for t in range(steps - 1, -1, -1):
            d = needed[t + 1].clone()
            d[pre[needed[t + 1][post]]] = True
            needed[t] = d
        self.steps = steps
        self.rows = [torch.nonzero(varying[t] & needed[t]).squeeze(1) for t in range(steps + 1)]      # R_0 is empty
        self.ops, self.keep_next, self.keep_prev, self.u_pos, self.u_src = [None], [None], [None], [None], [None]
        self.carried, self.prev_row = [None], [None]
        in_pos = torch.full((n,), -1, dtype=torch.long, device=dev)
        in_pos[in_idx] = torch.arange(len(in_idx), device=dev)
        for t in range(1, steps + 1):
            r_prev, r_next = self.rows[t - 1], self.rows[t]
            self.ops.append(_SubMatrix(r_next, r_prev, pre, post, n) if len(r_prev) else None)
            where_prev = torch.full((n,), -1, dtype=torch.long, device=dev)
            where_prev[r_prev] = torch.arange(len(r_prev), device=dev)
            carried = where_prev[r_next] >= 0                       # rows that already varied: their previous state is per position
            self.keep_next.append(torch.nonzero(carried).squeeze(1))
            self.keep_prev.append(where_prev[r_next][carried])
            self.carried.append(carried.unsqueeze(1))
            self.prev_row.append(where_prev[r_next].clamp_min(0))   # for the fused kernel: a row to gather for every row (any row if not carried)
            fed = in_pos[r_next] >= 0                               # rows that receive the position directly
            self.u_pos.append(torch.nonzero(fed).squeeze(1))
            self.u_src.append(in_pos[r_next][fed])
        where_last = torch.full((n,), -1, dtype=torch.long, device=dev)
        where_last[self.rows[steps]] = torch.arange(len(self.rows[steps]), device=dev)
        hit = where_last[out_idx] >= 0
        self.out_pos, self.out_src, self.out_idx = torch.nonzero(hit).squeeze(1), where_last[out_idx][hit], out_idx
        self.edge_visits = sum(len(op.edges) for op in self.ops if op is not None)


class ConnectomeRNN(nn.Module):
    def __init__(self, conn: Connectome, alpha_init: float = 0.5, bias_init: float = 0.0,
                 global_scale: float = 1.0, h_max: float | None = 10.0):
        """
        global_scale multiplier on the normalised connectome weights
        h_max        activity is clamped to [0, h_max]
        """
        super().__init__()
        self.n = conn.n_neurons
        self.h_max = h_max
        # the wiring is rebuilt from the connectome, not stored in checkpoints (it would add 340 MB to each)
        self.register_buffer("pre", conn.pre, persistent=False)      # edges are in (post, pre) order = CSR order of W
        self.register_buffer("post", conn.post, persistent=False)
        self.register_buffer("sign", conn.sign, persistent=False)
        self.register_buffer("w0", conn.weight * global_scale, persistent=False)
        self.register_buffer("crow", _crow(conn.post, self.n), persistent=False)
        t_order = torch.argsort(conn.pre * self.n + conn.post)      # CSR order of W^T
        self.register_buffer("t_order", t_order, persistent=False)
        self.register_buffer("t_crow", _crow(conn.pre[t_order], self.n), persistent=False)
        self.register_buffer("t_col", conn.post[t_order], persistent=False)
        self.log_gain = nn.Parameter(torch.zeros(conn.n_edges))
        self.bias = nn.Parameter(torch.full((self.n,), float(bias_init)))
        a = torch.full((self.n,), float(alpha_init))
        self.alpha_logit = nn.Parameter(torch.log(a / (1 - a)))

    def edge_weights(self) -> torch.Tensor:
        return self.sign * self.w0 * torch.exp(self.log_gain)

    def csr(self, w: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(self.crow, self.pre, w, (self.n, self.n), check_invariants=False)

    def csr_t(self, w: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(self.t_crow, self.t_col, w[self.t_order], (self.n, self.n),
                                       check_invariants=False)

    def forward(self, u: torch.Tensor, steps: int) -> torch.Tensor:
        """u: (N, B) contiguous constant external drive.  Returns the state after `steps` updates, (N, B).
        (A (B, N) state used through .T keeps its strides through elementwise ops and is 9x slower.)"""
        assert u.is_contiguous()
        alpha = torch.sigmoid(self.alpha_logit).to(u.dtype).unsqueeze(1)
        w = self.edge_weights().to(u.dtype)
        if not torch.is_grad_enabled():                     # inference: three passes over the state per step
            bias = u.add_(self.bias.to(u.dtype).unsqueeze(1))
            h = torch.zeros_like(bias)
            for _ in range(steps):
                x = (self.csr(w) @ h).add_(bias).clamp_(0, self.h_max)
                h.lerp_(x, alpha)
            return h
        bias = self.bias.to(u.dtype).unsqueeze(1) + u
        h = torch.zeros_like(bias)
        for _ in range(steps):
            x = (_SparseRecurrent.apply(h, w, self) + bias).clamp(0, self.h_max)
            h = torch.lerp(h, x, alpha)
        return h

    # ---- the same function through the light cone ------------------------------------------
    def light_cone(self, in_idx: torch.Tensor, out_idx: torch.Tensor, steps: int) -> LightCone:
        key = (steps, in_idx.data_ptr(), out_idx.data_ptr(), str(self.pre.device))
        if getattr(self, "_cone", None) is None or self._cone[0] != key:
            self._cone = (key, LightCone(self, in_idx, out_idx, steps))
        return self._cone[1]

    def _constants(self, cone: LightCone, w, alpha, dtype):
        """c_1..c_T, the trajectory of a network that sees no position, and per step the constant drive
        k_t = (W c_{t-1})[R_t] - A_t c_{t-1}[R_{t-1}] + b[R_t]   (what the non-varying neurons feed into R_t)."""
        c = torch.zeros(self.n, 1, dtype=w.dtype, device=w.device)
        cs, ks = [c], [None]
        bias = self.bias.to(w.dtype).unsqueeze(1)
        for t in range(1, cone.steps + 1):
            s = _SparseRecurrent.apply(c, w, self) if t > 1 else torch.zeros_like(c)      # W c_0 = 0
            k = (s + bias)[cone.rows[t]]
            if cone.ops[t] is not None:
                k = k - _SparseRecurrent.apply(c[cone.rows[t - 1]], w[cone.ops[t].edges], cone.ops[t])
            ks.append(k.to(dtype))
            c = torch.lerp(c, (s + bias).clamp(0, self.h_max), alpha)
            cs.append(c)
        return [x.to(dtype) for x in cs], ks

    @torch.no_grad()
    def _forward_io_half(self, u_in: torch.Tensor, cone: LightCone, steps: int) -> torch.Tensor:
        """Inference in float16 on deviations from the resting trajectory (see FUSED_DEVIATION).  Returns float32."""
        key = (self.log_gain._version, self.bias._version, self.alpha_logit._version, id(cone))
        if getattr(self, "_cached_half", None) is None or self._cached_half[0] != key:
            w = self.edge_weights()
            alpha = torch.sigmoid(self.alpha_logit).unsqueeze(1)
            bias = self.bias.unsqueeze(1)
            c, per_step = torch.zeros(self.n, 1, device=w.device), [None]
            for t in range(1, steps + 1):
                k_full = (_SparseRecurrent.apply(c, w, self) if t > 1 else torch.zeros_like(c)) + bias      # W c + b, float32, exact
                k = k_full[cone.rows[t]]
                xc = k.clamp(0, self.h_max)
                per_step.append(tuple(v.half().contiguous() for v in (-k, self.h_max - k, k - xc, alpha[cone.rows[t]]))
                                + (cone.carried[t].half(), cone.ops[t].csr(w[cone.ops[t].edges].half()) if cone.ops[t] is not None else None))
                c = torch.lerp(c, k_full.clamp(0, self.h_max), alpha)
            self._cached_half = (key, (per_step, c[cone.out_idx]))
        per_step, c_out = self._cached_half[1]
        u16, d = u_in.half(), None
        for t in range(1, steps + 1):
            lo, hi, e, a, carried, mat = per_step[t]
            s = mat @ d if mat is not None else u16.new_zeros(len(cone.rows[t]), u16.shape[1])
            s.index_add_(0, cone.u_pos[t], u16[cone.u_src[t]])
            d = FUSED_DEVIATION(s, lo, hi, e, d[cone.prev_row[t]] if d is not None else s, carried, a)
        out = c_out.expand(-1, u16.shape[1]).clone()
        out[cone.out_pos] += d[cone.out_src].float()
        return out

    def forward_io(self, u_in: torch.Tensor, in_idx: torch.Tensor, out_idx: torch.Tensor, steps: int) -> torch.Tensor:
        """u_in: (len(in_idx), B) drive of the input neurons.  Returns the output neurons after `steps`
        updates, (len(out_idx), B): exactly forward(u, steps)[out_idx] with u zero outside in_idx."""
        cone, dtype, batch = self.light_cone(in_idx, out_idx, steps), u_in.dtype, u_in.shape[1]
        grad = torch.is_grad_enabled()
        if not grad and dtype == torch.float16 and FUSED_DEVIATION is not None and u_in.is_cuda:
            return self._forward_io_half(u_in, cone, steps)
        if grad:
            w = self.edge_weights()
            alpha = torch.sigmoid(self.alpha_logit).unsqueeze(1)
            cs, ks = self._constants(cone, w, alpha, dtype)
            vals = [None] + [w[op.edges].to(dtype) if op is not None else None for op in cone.ops[1:]]
            alpha = alpha.to(dtype)
        else:                                       # everything that depends only on the weights is cached per weight version
            key = (self.log_gain._version, self.bias._version, self.alpha_logit._version, dtype, id(cone))
            if getattr(self, "_cached", None) is None or self._cached[0] != key:
                w = self.edge_weights()
                alpha = torch.sigmoid(self.alpha_logit).unsqueeze(1)
                cs, ks = self._constants(cone, w, alpha, dtype)
                mats = [None] + [op.csr(w[op.edges].to(dtype)) if op is not None else None for op in cone.ops[1:]]
                alpha = alpha.to(dtype)
                fused = [None] + [(cone.carried[t].to(dtype), cs[t - 1][cone.rows[t]].contiguous(), alpha[cone.rows[t]].contiguous())
                                  for t in range(1, steps + 1)]
                self._cached = (key, (cs, ks, mats, alpha, fused))
            cs, ks, mats, alpha, fused = self._cached[1]
        h = None
        for t in range(1, steps + 1):
            rows = cone.rows[t]
            if not grad and FUSED_UPDATE is not None and u_in.is_cuda and cone.ops[t] is not None and self.h_max == 10.0:
                y = (mats[t] @ h).index_add_(0, cone.u_pos[t], u_in[cone.u_src[t]])
                carried, c, a = fused[t]
                h = FUSED_UPDATE(y, ks[t], h[cone.prev_row[t]], carried, c, a)
                continue
            if cone.ops[t] is None:
                x = ks[t].expand(-1, batch)
            elif grad:
                x = _SparseRecurrent.apply(h, vals[t], cone.ops[t]) + ks[t]
            else:
                x = (mats[t] @ h).add_(ks[t])
            x = x.index_add(0, cone.u_pos[t], u_in[cone.u_src[t]]) if grad or cone.ops[t] is None \
                else x.index_add_(0, cone.u_pos[t], u_in[cone.u_src[t]])
            x = x.clamp(0, self.h_max) if grad else x.clamp_(0, self.h_max)
            prev = cs[t - 1][rows].expand(-1, batch)
            if len(cone.keep_next[t]):
                prev = prev.index_copy(0, cone.keep_next[t], h[cone.keep_prev[t]])
            h = torch.lerp(prev, x, alpha[rows])
        out = cs[steps][out_idx].expand(-1, batch)
        return out.index_copy(0, cone.out_pos, h[cone.out_src])

