"""What does the raw policy like?  On the centipawn suite: which kinds of move it picks, and what they cost.
    uv run python scripts/probe_policy.py runs/<name>/last.pt"""
import sys, collections
import chess, numpy as np, torch
from chessfly.model import Evaluator, load_net
from chessfly.suite import load_suite

suite = load_suite()
net = load_net(sys.argv[1])
ev = Evaluator(net, "cuda")
legal, values = [], []
for a in range(0, len(suite.x), 500):
    b = min(a + 500, len(suite.x)); m0, m1 = suite.offsets[a], suite.offsets[b]
    l, v = ev(suite.x[a:b], suite.rows[m0:m1] - a, suite.cols[m0:m1]); legal.append(l); values.append(v)
legal, values = np.concatenate(legal), np.concatenate(values)

def kind(board, move):
    if board.is_capture(move): return "capture"
    if board.gives_check(move): return "check"
    return {chess.PAWN: "pawn move", chess.KING: "king move"}.get(board.piece_type_at(move.from_square), "piece move")

stats = {k: collections.defaultdict(list) for k in ("policy", "random", "best")}
entropy, top_p = [], []
for i, fen in enumerate(suite.fens):
    board = chess.Board(fen); moves = list(board.legal_moves)
    s, e = suite.offsets[i], suite.offsets[i + 1]
    loss = suite.best_cp[i] - suite.cp[s:e]
    z = legal[s:e] - legal[s:e].max(); p = np.exp(z) / np.exp(z).sum()
    entropy.append(-(p * np.log(p + 1e-12)).sum() / np.log(len(moves))); top_p.append(p.max())
    k = int(p.argmax()); stats["policy"][kind(board, moves[k])].append(loss[k])
    k = int(loss.argmin()); stats["best"][kind(board, moves[k])].append(loss[k])
    for k, m in enumerate(moves): stats["random"][kind(board, m)].append(loss[k])
n = len(suite.fens)
print(f"{'kind':12s} | policy picks: share, avg cp loss | Stockfish's best: share | all legal moves: share, avg cp loss")
for k in ("capture", "check", "pawn move", "piece move", "king move"):
    P, B, R = stats["policy"][k], stats["best"][k], stats["random"][k]
    tot = sum(len(v) for v in stats["random"].values())
    print(f"{k:12s} | {len(P) / n:6.1%}  {np.mean(P) if P else 0:6.1f}          | {len(B) / n:6.1%}                  | {len(R) / tot:6.1%}  {np.mean(R):6.1f}")
print(f"policy: mean top probability {np.mean(top_p):.2f}, normalised entropy {np.mean(entropy):.2f} (1 = uniform)")
print(f"value head: mean {values.mean():+.3f}, sd {values.std():.3f}, corr with tanh(cp/300) {np.corrcoef(values, np.tanh(suite.best_cp / 300))[0, 1]:.3f}")
