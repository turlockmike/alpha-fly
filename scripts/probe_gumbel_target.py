"""How good would a completed-Q policy target be, given the current value head?  For every suite position, evaluate all
children one ply deep, form  pi' = softmax(logits + sigma(q))  with sigma(q) = (c_visit + 1) * c_scale * (q + 1) / 2,
and score pi' by its expected centipawn loss.    uv run python scripts/probe_gumbel_target.py runs/fly/last.pt"""
import sys
import chess, numpy as np
from chessfly.encoding import encode, legal_move_indices
from chessfly.model import Evaluator, load_net
from chessfly.suite import load_suite

suite = load_suite(); ev = Evaluator(load_net(sys.argv[1]), "cuda")
logits, _ = [], None
for a in range(0, len(suite.x), 500):
    b = min(a + 500, len(suite.x)); m0, m1 = suite.offsets[a], suite.offsets[b]
    logits.append(ev(suite.x[a:b], suite.rows[m0:m1] - a, suite.cols[m0:m1])[0])
logits = np.concatenate(logits)

xs, owner, terminal = [], [], {}
for i, fen in enumerate(suite.fens):                       # children in the suite's move order
    board = chess.Board(fen)
    for k, m in enumerate(board.legal_moves):
        board.push(m)
        if board.is_checkmate(): terminal[(i, k)] = 1.0
        elif board.is_stalemate() or board.is_insufficient_material(): terminal[(i, k)] = 0.0
        xs.append(encode(board)); owner.append((i, k)); board.pop()
xs = np.stack(xs); q = np.zeros(len(xs), np.float32)
for a in range(0, len(xs), 1024):
    chunk = xs[a:a + 1024]; z = np.zeros(0, np.int64)
    q[a:a + 1024] = -ev(chunk, z, z)[1]                     # the child's value is the opponent's: negate
for j, key in enumerate(owner):
    if key in terminal: q[j] = terminal[key]

def expected_loss(c_scale, use_logits=True, c_visit=50):
    out, top, agree = [], [], []
    for i in range(len(suite.fens)):
        s, e = suite.offsets[i], suite.offsets[i + 1]
        loss = suite.best_cp[i] - suite.cp[s:e]
        z = (logits[s:e] if use_logits else 0) + (c_visit + 1) * c_scale * (q[s:e] + 1) / 2
        p = np.exp(z - z.max()); p /= p.sum()
        out.append((p * loss).sum()); top.append(p.max()); agree.append(loss[p.argmax()] == 0)
    return np.mean(out), np.mean(top), np.mean(agree)

print("reference: random mover 194, greedy mover 123, current raw policy below (c_scale 0)")
for c in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 100.0):
    l, t, g = expected_loss(c)
    print(f"c_scale {c:6.2f}: expected cp loss of the target {l:6.1f} | top move gets {t:4.0%} | argmax is Stockfish's best {g:5.1%}")
