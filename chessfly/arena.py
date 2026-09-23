"""Strength against yardsticks that do not move with the network: games against the two ladder
rungs that bracket the current rating, turned into an Elo estimate (see ladder.py).

Two ways to run them.  `arena_process` plays them continuously in its own process while training
runs: its search leaves ride in the same GPU batches as self-play (64 games next to 1024 cost ~6%),
nothing ever waits for a match, and Elo is a rolling estimate refreshed every iteration.
`evaluate_strength` is the blocking match, used when the search runs in the training process.
This module must not import torch."""

from __future__ import annotations

import collections
import io
import queue
import signal
import threading
from pathlib import Path

import chess
import chess.pgn
import numpy as np

from .ladder import ADJUDICATE, MAX_PLIES, STYLE_RUNGS, Rung, estimate_elo, find_stockfish, load_ladder
from .mcts import MCTSConfig, Search, descend_all, finish_all, run_simulations, terminal_value
from .selfplay import material_balance


def play_match(evaluate, mcts: MCTSConfig, opponent, n_games: int, seed: int) -> dict:
    """The network (MCTS, no noise, best move) against `opponent`, alternating colours; all games run batched."""
    rng = np.random.default_rng(seed)
    games = [(Search(chess.Board(), mcts, rng), chess.WHITE if i % 2 == 0 else chess.BLACK) for i in range(n_games)]
    score, mates, live, results = 0.0, 0, list(games), {}

    def finished(s: Search, colour) -> bool:
        nonlocal score, mates
        b = s.board
        t = terminal_value(b, any(b.legal_moves))
        if t is None and b.ply() < MAX_PLIES:
            return False
        if t is not None:
            z = t if b.turn == colour else -t
            mates += z > 0
        else:
            m = material_balance(b) * (1 if colour == chess.WHITE else -1)
            z = float(np.sign(m)) if abs(m) >= ADJUDICATE else 0.0
        score += (z + 1) / 2
        results[id(s)] = z if colour == chess.WHITE else -z
        return True

    while live:
        for s, colour in live:                       # opponent replies first wherever it is to move
            if s.board.turn != colour:
                s.advance_move(opponent(s.board, rng))
        live = [g for g in live if not finished(*g)]
        if not live:
            break
        run_simulations([s for s, _ in live], evaluate, mcts.sims, noise=False)
        for s, _ in live:
            s.advance(s.best_action())
        live = [g for g in live if not finished(*g)]
    return {"score": score, "mates": mates, "games": [(s.board, colour, results[id(s)]) for s, colour in games]}


def bracket(ratings: dict[str, float], guess: float) -> list[str]:
    """The rung just below the guess and the one just above: matches near 50% carry the most information."""
    names = sorted((n for n in ratings if n not in STYLE_RUNGS), key=ratings.get)
    above = next((i for i, n in enumerate(names) if ratings[n] > guess), len(names) - 1)
    return names[max(above - 1, 0):max(above - 1, 0) + 2]


def write_pgn(path: Path, games, opponent: str, tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for board, colour, z in games:
            g = chess.pgn.Game.from_board(board)
            g.headers.update({"Event": tag, "White": "fly" if colour == chess.WHITE else opponent,
                              "Black": opponent if colour == chess.WHITE else "fly",
                              "Result": {1.0: "1-0", -1.0: "0-1"}.get(z, "1/2-1/2")})
            print(g, file=f, end="\n\n")


def evaluate_strength(evaluate, mcts: MCTSConfig, n_games: int, seed: int = 0, elo_guess: float | None = None,
                      pgn_dir: Path | None = None, tag: str = "eval") -> dict:
    """`n_games` against each of two rungs.  Without a calibrated ladder (or without Stockfish) it
    falls back to raw scores against the random and the greedy mover."""
    ratings = load_ladder()
    if ratings is None or find_stockfish() is None:
        ratings, names = None, ["random", "greedy"]
    else:
        names = bracket(ratings, ratings["random"] if elo_guess is None else elo_guess)
    out, results = {}, []
    for name in names:
        rung = Rung(name)
        try:
            r = play_match(evaluate, mcts, rung, n_games, seed)
        finally:
            rung.close()
        out[f"vs_{name}"] = round(r["score"] / n_games, 3)
        if pgn_dir:
            write_pgn(pgn_dir / f"{tag}_vs_{name}.pgn", r["games"], name, tag)
        out["mates"] = out.get("mates", 0) + r["mates"]
        if ratings:
            results.append((ratings[name], r["score"], n_games))
    if ratings:
        elo, se = estimate_elo(results)
        out |= {"elo": round(elo), "elo_se": round(se)}
    return out


def pgn_text(board: chess.Board, colour, z_white: float, opponent: str, tag: str) -> str:
    g = chess.pgn.Game.from_board(board)
    g.headers.update({"Event": tag, "White": "fly" if colour == chess.WHITE else opponent,
                      "Black": opponent if colour == chess.WHITE else "fly",
                      "Result": {1.0: "1-0", -1.0: "0-1"}.get(z_white, "1/2-1/2")})
    out = io.StringIO()
    print(g, file=out, end="\n\n")
    return out.getvalue()


# The opponents of the rolling evaluation.  FIXED, not the two rungs nearest the current estimate: the ladder is not
# transitive for this player.  The rungs were rated against each other, where Stockfish's share of the moves punishes the
# random share; the fly cannot yet convert won positions, so the diluted rungs are far closer together for it than their
# ratings say (at "Elo 720" it scored 70% / 57% / 30% against the rungs rated 335 / 661 / 879).  An adaptive bracket then has
# two self-consistent answers for the same network - ~720 between the 661 and 879 rungs, ~550 between 335 and 661 - and
# which one it reports depends on where it started.  A fixed panel gives one answer, and the direct score against each rung
# (vs_<rung>, with n_<rung> games) is the number to trust: milestones mean scoring 50% against that rung itself.
PANEL = ("sf_mix50", "sf_mix75", "sf_mix90", "sf_skill0", "sf_1320")


def arena_process(conn, mcts: MCTSConfig, n_games: int, window: int, elo_guess: float | None, seed: int, slot_names=None) -> None:
    """Plays ladder games for ever.  Protocol as the self-play workers': send (request, reports), receive
    (legal_logits, values); every finished game adds a report with the rolling Elo.  Games are out of
    phase with one another, so each keeps its own simulation count."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from .workers import PolicySlots, unpack
    slots = PolicySlots(n_games, 1, slot_names) if slot_names else None
    rng = np.random.default_rng(seed)
    ratings = load_ladder()
    elo = ratings["random"] if elo_guess is None else elo_guess
    rungs: dict[str, Rung] = {}
    results: collections.deque = collections.deque(maxlen=window)        # (rung, score, mate)
    outbox: queue.Queue = queue.Queue()                                  # see workers.py: never block in send
    threading.Thread(target=lambda: [conn.send(m) for m in iter(outbox.get, None)], daemon=True).start()
    started = 0

    def new_game():
        nonlocal started
        name = PANEL[started % len(PANEL)]
        colour = chess.WHITE if (started // len(PANEL)) % 2 == 0 else chess.BLACK
        started += 1
        if name not in rungs:
            rungs[name] = Rung(name)
        return {"search": Search(chess.Board(), mcts, rng), "colour": colour, "rung": name, "sims": 0}

    def settle(g) -> dict | None:
        """Let the opponent move if it is its turn; report the game if it is over."""
        nonlocal elo
        s, b = g["search"], g["search"].board
        for _ in range(2):
            t = terminal_value(b, any(b.legal_moves))
            if t is not None or b.ply() >= MAX_PLIES:
                if t is not None:
                    z = t if b.turn == g["colour"] else -t
                else:
                    m = material_balance(b) * (1 if g["colour"] == chess.WHITE else -1)
                    z = float(np.sign(m)) if abs(m) >= ADJUDICATE else 0.0
                results.append((g["rung"], (z + 1) / 2, t is not None and z > 0))
                per = {}
                for name, score, _ in results:
                    per.setdefault(name, []).append(score)
                elo, se = estimate_elo([(ratings[n], sum(v), len(v)) for n, v in per.items()])
                return {"elo": round(elo), "elo_se": round(se), "elo_games": len(results), "mates": sum(m for *_, m in results),
                        **{f"vs_{n}": round(sum(v) / len(v), 3) for n, v in per.items()},
                        **{f"n_{n}": len(v) for n, v in per.items()},
                        "pgn": pgn_text(b, g["colour"], z if g["colour"] == chess.WHITE else -z, g["rung"], "rolling evaluation")}
            if b.turn == g["colour"]:
                return None
            s.advance_move(rungs[g["rung"]](b, rng))
        return None

    games = [new_game() for _ in range(n_games)]
    reports = [r for g in games if (r := settle(g))]
    try:
        while True:
            pending, request = descend_all([g["search"] for g in games])
            outbox.put((request, reports))
            reports = []
            reply = conn.recv()
            if reply is None:
                break
            finish_all(pending, *unpack(reply, slots))
            for i, g in enumerate(games):
                s = g["search"]
                g["sims"] += 1
                if g["sims"] > mcts.sims:                    # one round to expand the root, then `sims` simulations
                    g["sims"] = 0
                    s.advance(s.best_action())
                    if (r := settle(g)) is not None:
                        reports.append(r)
                        games[i] = new_game()
                        if (r := settle(games[i])) is not None:
                            reports.append(r)
    finally:
        for r in rungs.values():
            r.close()
