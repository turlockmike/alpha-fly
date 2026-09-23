"""An Elo scale for networks far too weak for Stockfish's own (its floor is UCI_Elo 1320).

A ladder of fixed opponents runs from a random mover, through Stockfish diluted with random
moves, up to Stockfish's calibrated levels.  `scripts/calibrate_ladder.py` plays the rungs against
each other once and fits their ratings, anchored at  Stockfish 17.1 UCI_Elo 1320 := 1320.
Node limits rather than time limits, so ratings do not depend on the machine.

Stockfish is only ever a yardstick here: nothing it produces reaches the training data.
This module must not import torch (calibration runs it in worker processes)."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import chess
import chess.engine
import numpy as np

from .selfplay import PIECE_VALUE, material_balance

MAX_PLIES = 200
ADJUDICATE = 5          # same rule as self-play: at MAX_PLIES a 5-point material lead wins
ANCHOR = ("sf_1320", 1320.0)
STYLE_RUNGS = {"greedy"}     # rated, but not used to measure Elo: it punishes hung pieces far beyond its general strength
LADDER_FILE = Path(__file__).parent.parent / "eval" / "ladder.json"

# name -> None for the built-in movers, else Stockfish settings.  `p` is the probability of
# playing Stockfish's move rather than a random one.
RUNGS: dict[str, dict | None] = {
    "random": None,
    "greedy": None,
    "sf_mix10": {"skill": 0, "nodes": 100, "p": 0.10},
    "sf_mix25": {"skill": 0, "nodes": 100, "p": 0.25},
    "sf_mix50": {"skill": 0, "nodes": 100, "p": 0.50},
    "sf_mix75": {"skill": 0, "nodes": 100, "p": 0.75},
    "sf_mix90": {"skill": 0, "nodes": 100, "p": 0.90},
    "sf_skill0": {"skill": 0, "nodes": 100},
    "sf_skill0_10k": {"skill": 0, "nodes": 10_000},
    "sf_1320": {"elo": 1320, "nodes": 100_000},
    "sf_1600": {"elo": 1600, "nodes": 100_000},
}


def find_stockfish() -> str | None:
    here = Path(__file__).parent.parent / "tools" / "stockfish.exe"
    for c in (os.environ.get("STOCKFISH"), str(here) if here.exists() else None, shutil.which("stockfish")):
        if c:
            return c
    return None


def random_mover(board: chess.Board, rng) -> chess.Move:
    moves = list(board.legal_moves)
    return moves[rng.integers(len(moves))]


def greedy_mover(board: chess.Board, rng) -> chess.Move:
    """Mates if it can, otherwise takes the most valuable piece, otherwise moves at random."""
    best, best_score = [], -1
    for m in board.legal_moves:
        board.push(m)
        mate = board.is_checkmate()
        board.pop()
        victim = board.piece_type_at(m.to_square)
        score = 100 if mate else PIECE_VALUE.get(victim, 0) + (8 if m.promotion == chess.QUEEN else 0)
        if score > best_score:
            best, best_score = [m], score
        elif score == best_score:
            best.append(m)
    return best[rng.integers(len(best))]


class Rung:
    """A ladder opponent: `rung(board, rng) -> move`.  Owns a Stockfish process if it needs one."""

    def __init__(self, name: str):
        self.name, self.spec, self.engine = name, RUNGS[name], None
        if self.spec:
            path = find_stockfish()
            if path is None:
                raise FileNotFoundError("Stockfish not found: put it at tools/stockfish.exe or set STOCKFISH")
            # own process group: a Ctrl+C meant to pause training would otherwise kill the engine mid-game
            flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
            self.engine = chess.engine.SimpleEngine.popen_uci(path, **flags)
            opts = {"Threads": 1, "Hash": 16}
            opts |= ({"UCI_LimitStrength": True, "UCI_Elo": self.spec["elo"]} if "elo" in self.spec
                     else {"Skill Level": self.spec["skill"]})
            self.engine.configure(opts)
            self.limit = chess.engine.Limit(nodes=self.spec["nodes"])

    def __call__(self, board: chess.Board, rng) -> chess.Move:
        if self.name == "greedy":
            return greedy_mover(board, rng)
        if self.spec is None or rng.random() >= self.spec.get("p", 1.0):
            return random_mover(board, rng)
        return self.engine.play(board, self.limit).move

    def close(self) -> None:
        if self.engine:
            self.engine.quit()


def game_result(board: chess.Board) -> float | None:
    """+1 / 0 / -1 for white once the game is over (same rules as self-play), else None."""
    if not any(board.legal_moves):
        return (-1.0 if board.turn == chess.WHITE else 1.0) if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    if board.ply() >= MAX_PLIES:
        m = material_balance(board)
        return float(np.sign(m)) if abs(m) >= ADJUDICATE else 0.0
    return None


def play_game(white, black, rng) -> float:
    board = chess.Board()
    while (z := game_result(board)) is None:
        board.push((white if board.turn == chess.WHITE else black)(board, rng))
    return z


# ---- ratings -------------------------------------------------------------------------------

def expected(r_a: float, r_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400))


def fit_ratings(names: list[str], results: list[tuple[str, str, float, int]]) -> dict[str, float]:
    """Maximum-likelihood Elo from (a, b, score of a, games).  A draw counts as half a win and half
    a loss; one virtual draw per pairing keeps 100% scores finite."""
    i = {n: k for k, n in enumerate(names)}
    r = np.zeros(len(names))
    pairs = [(i[a], i[b], s + 0.5, n + 1) for a, b, s, n in results]
    for _ in range(3000):                                   # damped diagonal Newton steps
        grad, curv = np.zeros_like(r), np.full(len(r), 1e-9)
        for a, b, s, n in pairs:
            e = expected(r[a], r[b])
            grad[a] += s - n * e
            grad[b] -= s - n * e
            curv[a] += n * e * (1 - e)
            curv[b] += n * e * (1 - e)
        r += 0.5 * (400 / math.log(10)) * grad / curv
    return {n: float(r[i[n]] - r[i[ANCHOR[0]]] + ANCHOR[1]) for n in names}


def estimate_elo(results: list[tuple[float, float, int]], lo=-1500.0, hi=3500.0) -> tuple[float, float]:
    """Rating and standard error of one player from (opponent rating, score, games)."""
    score = sum(s for _, s, _ in results) + 0.5 * len(results)          # same virtual draws
    games = [(r, n + 1) for r, _, n in results]
    for _ in range(60):                                                 # expected score is monotone: bisect
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sum(n * expected(mid, r) for r, n in games) < score else (lo, mid)
    info = sum(n * expected(mid, r) * (1 - expected(mid, r)) for r, n in games) * (math.log(10) / 400) ** 2
    return mid, 1 / math.sqrt(info)


def load_ladder() -> dict[str, float] | None:
    return json.loads(LADDER_FILE.read_text())["ratings"] if LADDER_FILE.exists() else None
