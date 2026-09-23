"""Play the ladder's rungs against each other and fit their Elo (no GPU).  Run once:
    uv run python scripts/calibrate_ladder.py [games_per_pairing]
Each rung meets the next three rungs up; writes eval/ladder.json."""

import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chessfly.ladder import ANCHOR, LADDER_FILE, RUNGS, Rung, fit_ratings, play_game


def play_pairing(job):
    a, b, n, seed = job
    rng = np.random.default_rng(seed)
    ra, rb = Rung(a), Rung(b)
    score = 0.0
    try:
        for g in range(n):
            z = play_game(ra, rb, rng) if g % 2 == 0 else -play_game(rb, ra, rng)
            score += (z + 1) / 2
    finally:
        ra.close(); rb.close()
    return a, b, score, n


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    names = list(RUNGS)
    # chunks of 10 games so the slow pairings (100k-node Stockfish on both sides) spread over all cores
    jobs = [(names[i], names[j], 10, 1000 * i + 10 * j + c)
            for i in range(len(names)) for j in range(i + 1, min(i + 4, len(names))) for c in range(n // 10)]
    totals: dict[tuple[str, str], list[float]] = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for k, (a, b, s, g) in enumerate(ex.map(play_pairing, jobs)):
            t = totals.setdefault((a, b), [0.0, 0])
            t[0] += s; t[1] += g
            print(f"\r{k + 1}/{len(jobs)} chunks", end="", flush=True)
    results = [(a, b, s, g) for (a, b), (s, g) in totals.items()]
    ratings = fit_ratings(names, results)
    print(f"\n\nanchor: {ANCHOR[0]} := {ANCHOR[1]:.0f}")
    for name in names:
        print(f"  {name:14s} {ratings[name]:7.0f}")
    print()
    for a, b, s, g in results:
        print(f"  {a:14s} vs {b:14s} {s:5.1f} / {g}")
    LADDER_FILE.parent.mkdir(exist_ok=True)
    LADDER_FILE.write_text(json.dumps({"anchor": ANCHOR, "ratings": ratings,
                                       "results": results, "rungs": RUNGS}, indent=1))
    print(f"\nwrote {LADDER_FILE}")
