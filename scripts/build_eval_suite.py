"""Build the fixed centipawn-loss suite (no GPU).  Run once:
    uv run python scripts/build_eval_suite.py [positions] [depth]
Positions come from games between ladder opponents of mixed strength, so they range from tidy to
the chaos a young network actually reaches.  Stockfish then evaluates every legal move of each."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import chess
import chess.engine
import numpy as np

from chessfly.ladder import Rung, find_stockfish, game_result, greedy_mover
from chessfly.suite import CP_CAP, SUITE_FILE

PLAYERS = ["random", "greedy", "sf_mix25", "sf_mix50", "sf_skill0", "sf_skill0_10k"]


def make_positions(job):
    n, depth, seed = job
    rng = np.random.default_rng(seed)
    rungs = {p: Rung(p) for p in PLAYERS}
    engine = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    engine.configure({"Threads": 1, "Hash": 64})
    out = {}
    try:
        while len(out) < n:
            white, black = (rungs[p] for p in rng.choice(PLAYERS, 2))
            board, boards = chess.Board(), []
            while game_result(board) is None and board.ply() < 160:
                board.push((white if board.turn == chess.WHITE else black)(board, rng))
                boards.append(board.copy(stack=False))
            for b in rng.choice(boards[4:-1], size=min(3, max(len(boards) - 5, 0)), replace=False) if len(boards) > 6 else []:
                moves = list(b.legal_moves)
                if len(moves) < 2 or b.fen() in out:
                    continue
                infos = engine.analyse(b, chess.engine.Limit(depth=depth), multipv=len(moves))
                evals = {i["pv"][0].uci(): int(np.clip(i["score"].pov(b.turn).score(mate_score=CP_CAP), -CP_CAP, CP_CAP))
                         for i in infos}
                if len(evals) == len(moves):
                    out[b.fen()] = evals
    finally:
        engine.quit()
        for r in rungs.values():
            r.close()
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    suite = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for part in ex.map(make_positions, [(n // 8 + 1, depth, s) for s in range(8)]):
            suite |= part
    fens = sorted(suite)[:n]
    rng = np.random.default_rng(0)
    best = [max(suite[f].values()) for f in fens]
    baselines = {
        "random": round(float(np.mean([b - np.mean(list(suite[f].values())) for f, b in zip(fens, best)])), 1),
        "greedy": round(float(np.mean([b - suite[f][greedy_mover(chess.Board(f), rng).uci()] for f, b in zip(fens, best)])), 1),
    }
    SUITE_FILE.parent.mkdir(exist_ok=True)
    SUITE_FILE.write_text(json.dumps({"engine": "Stockfish 17.1", "depth": depth, "cp_cap": CP_CAP,
                                      "baselines": baselines, "fens": fens, "evals": [suite[f] for f in fens]}))
    print(f"{len(fens)} positions, {sum(len(suite[f]) for f in fens)} evaluated moves, depth {depth}")
    print(f"average centipawn loss of the random mover {baselines['random']}, of the greedy mover {baselines['greedy']}")
    print(f"wrote {SUITE_FILE}")
