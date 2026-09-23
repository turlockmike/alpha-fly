"""Same opponent, same search, same seed: how do two (or more) checkpoints score?  Uses the blocking match.
    uv run python scripts/compare_checkpoints.py <rung> <games> <sims> ckpt1.pt ckpt2.pt@gumbel ...
(@gumbel searches with Gumbel AlphaZero instead of PUCT; @gumbel+fp16 also infers in half precision, as runs did before
the float16 problem was found)"""
import sys, time
import torch
from chessfly.arena import play_match
from chessfly.ladder import Rung
from chessfly.mcts import MCTSConfig
from chessfly.model import Evaluator, load_net

rung_name, games, sims = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for arg in sys.argv[4:]:
    path, _, mode = arg.partition("@")
    net = load_net(path); rung = Rung(rung_name); t = time.time()
    if "fp16" in mode:
        net.infer_dtype = torch.float16
    try:
        r = play_match(Evaluator(net, "cuda"), MCTSConfig(sims=sims, gumbel="gumbel" in mode), rung, games, seed=1)
    finally:
        rung.close()
    plies = [b.ply() for b, _, _ in r["games"]]
    print(f"{arg}: {r['score']:.1f} / {games} vs {rung_name} ({r['score'] / games:.0%}), mates {r['mates']}, avg plies {sum(plies) / len(plies):.0f}, {time.time() - t:.0f}s", flush=True)
    del net; torch.cuda.empty_cache()
