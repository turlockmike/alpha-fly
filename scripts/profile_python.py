"""cProfile of the tree search with the network replaced by a free stub: what does Python spend its time on?
    uv run python scripts/profile_python.py [gumbel] [lazy] [noprofile]"""
import cProfile, pstats, sys, time, numpy as np, chess
from chessfly.encoding import N_MOVES
from chessfly.mcts import MCTSConfig, Search, run_simulations

rng = np.random.default_rng(0)
cfg = MCTSConfig(sims=32, gumbel="gumbel" in sys.argv, lazy="lazy" in sys.argv)
searches = [Search(chess.Board(), cfg, rng, explore=True) for _ in range(128)]
FULL = rng.standard_normal((128, N_MOVES)).astype(np.float16)
stub = lambda x, rows, cols: (FULL[:len(x)] if rows is None else rng.standard_normal(len(rows), dtype=np.float32), rng.standard_normal(len(x), dtype=np.float32) * 0.3)

def play(plies):
    for _ in range(plies):
        run_simulations(searches, stub, 32, noise=True)
        for s in searches:
            if s.root.N is not None and s.root.terminal is None:
                s.advance(s.best_action())

play(30)                        # get out of the opening so positions are typical
leaves = 128 * 6 * 33
if "noprofile" in sys.argv:
    t = time.time(); play(6); print(f"== {(time.time() - t) / leaves * 1e6:.0f} us of Python per leaf, no profiler ({sys.argv[1:]})")
else:
    pr = cProfile.Profile(); pr.enable(); play(6); pr.disable()
    st = pstats.Stats(pr); st.sort_stats("tottime").print_stats(16)
    print(f"== {st.total_tt / leaves * 1e6:.0f} us of Python per leaf (profiler overhead included)")
