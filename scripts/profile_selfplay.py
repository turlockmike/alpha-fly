"""Self-play throughput, and how much of the wall time the network is busy.
    uv run python scripts/profile_selfplay.py [parallel_games] [sims] [workers] [steps] [raise trainer priority 0/1]   (Gumbel + lazy search, as the main run)"""
import sys, time
import torch
from chessfly.connectome import load_malecns
from chessfly.mcts import MCTSConfig
from chessfly.model import Evaluator, FlyChessNet, ModelConfig
from chessfly.selfplay import SelfPlayConfig
from chessfly.workers import make_selfplay

if __name__ == "__main__":
    games, sims, workers, steps = (int(a) for a in (sys.argv[1:5] + ["1120", "32", "14", "6"][len(sys.argv[1:5]):]))
    prio = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    net = FlyChessNet(load_malecns("data"), ModelConfig(steps=steps)).cuda()
    ev = Evaluator(net, torch.device("cuda"))            # the real, pipelined evaluator: only throughput is measured
    cfg = SelfPlayConfig(parallel_games=games, workers=workers, max_plies=40, prioritise_trainer=prio)
    sp = make_selfplay(ev, MCTSConfig(sims=sims, gumbel=True, lazy=True), cfg)
    sp.play(max(games // 8, 1))                  # warm-up: processes started, first games finished
    t, moves_before = time.time(), sp.stats["moves"]
    sp.play(games // 2)
    total, moves = time.time() - t, sp.stats["moves"] - moves_before
    print(f"{games} games, {sims} sims, {workers} workers, {steps} steps, trainer priority {'raised' if prio else 'normal'}: "
          f"{moves / total:.0f} moves/s = {moves / total * (sims + 1):,.0f} evals/s", flush=True)
    sp.close()
