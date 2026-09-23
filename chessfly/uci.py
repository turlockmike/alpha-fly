"""The fly as a UCI engine, so any chess GUI (Arena, Cute Chess, Banksia, lichess-bot) can play a checkpoint.

    uv run python -m chessfly.uci runs/fly/last.pt                 # or runs/fly/snapshots/iter_000100.pt
    engine command for a GUI:   uv run --project C:\\workspace\\chess-fly python -m chessfly.uci C:\\workspace\\chess-fly\\runs\\fly\\last.pt

Options: Simulations (default 200; "go nodes N" overrides it per move).  Clocks are ignored: the engine
always searches the same number of simulations, which keeps it quick and its strength reproducible.
It can run while training does; both share the GPU."""

from __future__ import annotations

import sys

import chess
import numpy as np

from .mcts import MCTSConfig, Search, run_simulations
from .model import Evaluator, load_net


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/fly/last.pt"
    evaluate, board, sims = None, chess.Board(), 200
    out = lambda s: print(s, flush=True)
    for line in sys.stdin:
        words = line.split()
        if not words:
            continue
        cmd = words[0]
        if cmd == "uci":
            out("id name chess-fly (MaleCNS connectome)")
            out("id author chess-fly")
            out("option name Simulations type spin default 200 min 1 max 100000")
            out("uciok")
        elif cmd == "isready":
            if evaluate is None:                                  # load lazily: GUIs send "uci" just to list engines
                net = load_net(path)
                evaluate = Evaluator(net, next(net.parameters()).device)
            out("readyok")
        elif cmd == "setoption" and "Simulations" in words and "value" in words:
            sims = int(words[words.index("value") + 1])
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            moves = words[words.index("moves") + 1:] if "moves" in words else []
            board = chess.Board() if words[1] == "startpos" else chess.Board(" ".join(words[2:8]))
            for m in moves:
                board.push_uci(m)
        elif cmd == "go":
            if evaluate is None:
                net = load_net(path)
                evaluate = Evaluator(net, next(net.parameters()).device)
            n = int(words[words.index("nodes") + 1]) if "nodes" in words else sims
            search = Search(board.copy(), MCTSConfig(sims=n), np.random.default_rng())
            run_simulations([search], evaluate, n, noise=False)
            root = search.root
            if root.moves is None:                                # no legal moves: the GUI should not have asked
                out("bestmove 0000")
                continue
            best = search.best_action()
            q = float(root.W[best] / max(root.N[best], 1))
            cp = int(np.clip(-400 * np.log10(max(1e-6, 2 / (np.clip(q, -0.999, 0.999) + 1) - 1)), -3000, 3000))
            out(f"info depth 1 nodes {int(root.N.sum())} score cp {cp} pv {root.moves[best].uci()}")
            out(f"bestmove {root.moves[best].uci()}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
