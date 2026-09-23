"""Centipawn loss without an engine in the loop.

`scripts/build_eval_suite.py` stores, once, Stockfish's evaluation of *every legal move* in a
fixed set of positions.  Scoring a network is then a table lookup: which move does it pick, and
how many centipawns worse than the best move is that?  Cheap enough to run every iteration.

Stockfish is only a yardstick: the suite never reaches the training data."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import chess
import numpy as np

from .encoding import encode, legal_move_indices
from .mcts import MCTSConfig, Search, run_simulations

SUITE_FILE = Path(__file__).parent.parent / "eval" / "suite.json"
CP_CAP = 1000            # evaluations are clipped to +-10 pawns; mates count as the cap


@dataclasses.dataclass
class Suite:
    fens: list[str]
    x: np.ndarray            # (n, F)
    rows: np.ndarray         # (M,) position of each legal move
    cols: np.ndarray         # (M,) its policy index
    cp: np.ndarray           # (M,) Stockfish evaluation after that move, for the side to move
    offsets: np.ndarray      # (n + 1,)
    best_cp: np.ndarray      # (n,)
    baselines: dict          # acpl of the random and greedy movers on this suite

    def loss_of(self, position: int, policy_index: int) -> float:
        a, b = self.offsets[position], self.offsets[position + 1]
        return float(self.best_cp[position] - self.cp[a:b][self.cols[a:b] == policy_index][0])


def load_suite(path: Path = SUITE_FILE) -> Suite | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    xs, rows, cols, cps, offsets = [], [], [], [], [0]
    for i, (fen, evals) in enumerate(zip(d["fens"], d["evals"])):
        board = chess.Board(fen)
        moves, idx = legal_move_indices(board)
        xs.append(encode(board))
        rows.append(np.full(len(moves), i))
        cols.append(idx)
        cps.append([evals[m.uci()] for m in moves])
        offsets.append(offsets[-1] + len(moves))
    cp = np.concatenate([np.asarray(c, np.float32) for c in cps])
    offsets = np.asarray(offsets)
    return Suite(d["fens"], np.stack(xs), np.concatenate(rows), np.concatenate(cols), cp, offsets,
                 np.maximum.reduceat(cp, offsets[:-1]), d["baselines"])


def policy_metrics(evaluate, suite: Suite, batch: int = 512) -> dict:
    """The raw network, no search: move = most likely legal move."""
    losses, values = [], []
    for a in range(0, len(suite.x), batch):
        b = min(a + batch, len(suite.x))
        m0, m1 = suite.offsets[a], suite.offsets[b]
        legal, v = evaluate(suite.x[a:b], suite.rows[m0:m1] - a, suite.cols[m0:m1])
        values.append(v)
        for i in range(a, b):
            s, e = suite.offsets[i] - m0, suite.offsets[i + 1] - m0
            losses.append(suite.best_cp[i] - suite.cp[m0 + s + int(legal[s:e].argmax())])
    losses = np.asarray(losses)
    # is the value head ordered like Stockfish's evaluation?  (squashed so that +3 and +9 pawns are both "winning")
    v = np.concatenate(values)
    value_corr = float(np.corrcoef(v, np.tanh(suite.best_cp / 300))[0, 1]) if v.std() > 1e-6 else 0.0
    value_corr = value_corr if np.isfinite(value_corr) else 0.0      # an untrained head is constant; NaN is not valid JSON
    return {"acpl_policy": round(float(losses.mean()), 1), "top1_policy": round(float((losses == 0).mean()), 3),
            "value_corr": round(value_corr, 3)}


def search_metrics(evaluate, suite: Suite, mcts: MCTSConfig) -> dict:
    """Network plus MCTS, as it actually plays."""
    rng = np.random.default_rng(0)
    searches = [Search(chess.Board(fen), mcts, rng) for fen in suite.fens]
    run_simulations(searches, evaluate, mcts.sims, noise=False)
    losses = np.asarray([suite.loss_of(i, int(s.root.idx[s.best_action()])) for i, s in enumerate(searches)])
    return {"acpl_search": round(float(losses.mean()), 1), "top1_search": round(float((losses == 0).mean()), 3)}
