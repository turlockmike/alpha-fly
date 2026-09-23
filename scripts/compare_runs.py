"""The control of the experiment: two runs (the fly's wiring and --shuffled) compared at MATCHED GAME COUNTS.

    uv run python scripts/compare_runs.py fly6_noadj shuffled6            # table every 10,000 games
    uv run python scripts/compare_runs.py fly6_noadj shuffled6 5000

Reads only runs/<name>/log.jsonl - no GPU, safe while a run is training.  At each game count it takes, from each run,
the logged row nearest below it, and reports the rolling panel Elo with its standard error, the gap (first minus
second) with the combined error, and the raw-network centipawn loss.  A gap is called only beyond two combined standard
errors.  The rolling Elo is noisy just after a restart (few games in the window: see arena_min_games), so rows whose
window holds fewer than MIN_GAMES are skipped.  For a verdict on two particular checkpoints use the fixed-seed
head-to-head instead: scripts/compare_checkpoints.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MIN_GAMES = 200


def rows(run: str) -> list[dict]:
    out = []
    for line in (ROOT / "runs" / run / "log.jsonl").read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:                      # a row half-written at a crash
            continue
        if r.get("type") == "iter" and r.get("elo") is not None and r.get("elo_games", 0) >= MIN_GAMES:
            out.append(r)
    return out


def at(rs: list[dict], games: int) -> dict | None:
    below = [r for r in rs if r["games_total"] <= games]
    return below[-1] if below and games - below[-1]["games_total"] < 5000 else None


def main() -> None:
    a, b = sys.argv[1], sys.argv[2]
    every = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    ra, rb = rows(a), rows(b)
    if not ra or not rb:
        sys.exit(f"no Elo rows with >= {MIN_GAMES} games in the window: {a}: {len(ra)}, {b}: {len(rb)}")
    last = min(ra[-1]["games_total"], rb[-1]["games_total"])
    print(f"{'games':>8} | {a:>18} | {b:>18} | {'gap':>12} | verdict   | CPL {a[:8]:>8} {b[:8]:>8}")
    for g in list(range(every, last, every)) + [last]:
        x, y = at(ra, g), at(rb, g)
        if x is None or y is None:
            continue
        gap, se = x["elo"] - y["elo"], (x["elo_se"] ** 2 + y["elo_se"] ** 2) ** 0.5
        verdict = "tie" if abs(gap) < 2 * se else (a if gap > 0 else b)[:9]
        print(f"{g:>8,} | {x['elo']:>11.0f} +/- {x['elo_se']:<3.0f}| {y['elo']:>11.0f} +/- {y['elo_se']:<3.0f}| {gap:>+6.0f} +/- {se:<3.0f}| {verdict:<9} |"
              f"     {x.get('acpl_policy') or float('nan'):>8.1f} {y.get('acpl_policy') or float('nan'):>8.1f}")


if __name__ == "__main__":
    main()
