"""The training curve of a run as a small dependency-free SVG for the README: rolling Elo against games played,
with the ladder rungs as reference lines, and the raw-policy centipawn loss below it.

    uv run python scripts/plot_training.py runs/fly6_noadj/log.jsonl docs/assets/training.svg
"""
import json
import sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.startswith("{")]
it = [r for r in rows if r.get("type") == "iter" and r.get("elo") is not None and r.get("elo_games", 0) >= 200]
cpl = [r for r in rows if r.get("type") == "iter" and r.get("acpl_policy")]

W, H, L, R, T, B, GAP, H2 = 880, 470, 62, 175, 24, 40, 40, 120
PW = W - L - R
top_h = H - T - B - GAP - H2
xmax = max(r["games_total"] for r in it)
X = lambda g: L + PW * g / xmax
INK, INK2, MUTED, GRID, ACC, TEAL = "#131a17", "#4b5753", "#7b8783", "#e3e6e2", "#b8236e", "#177566"
font = 'font-family="Instrument Sans, Helvetica Neue, Arial, sans-serif"'

# --- Elo panel ---
y0, y1 = 300, 1400
Y = lambda e: T + top_h * (1 - (e - y0) / (y1 - y0))
parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" {font} font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>']
for e in range(400, 1401, 200):
    parts.append(f'<line x1="{L}" x2="{L + PW}" y1="{Y(e):.1f}" y2="{Y(e):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{Y(e) + 4:.1f}" text-anchor="end" fill="{MUTED}">{e}</text>')
for elo, label in ((661, "Stockfish, 75% random"), (879, "Stockfish, 90% random"), (1179, "Stockfish skill 0"), (1320, "Stockfish Elo 1320")):
    parts.append(f'<line x1="{L}" x2="{L + PW}" y1="{Y(elo):.1f}" y2="{Y(elo):.1f}" stroke="{MUTED}" stroke-dasharray="3 4"/>'
                 f'<text x="{L + PW + 6}" y="{Y(elo) + 4:.1f}" fill="{INK2}" font-size="11">{label}</text>')
# smoothed (5-row) Elo line, with the +/- 1 se band
sm = [sum(x["elo"] for x in it[max(0, i - 2):i + 3]) / len(it[max(0, i - 2):i + 3]) for i in range(len(it))]
band = "M" + " L".join(f"{X(r['games_total']):.1f},{Y(sm[i] + r['elo_se']):.1f}" for i, r in enumerate(it)) + \
       " L" + " L".join(f"{X(r['games_total']):.1f},{Y(sm[i] - r['elo_se']):.1f}" for i, r in reversed(list(enumerate(it)))) + " Z"
parts.append(f'<path d="{band}" fill="{ACC}" opacity="0.12"/>')
line = "M" + " L".join(f"{X(r['games_total']):.1f},{Y(v):.1f}" for r, v in zip(it, sm))
parts.append(f'<path d="{line}" fill="none" stroke="{ACC}" stroke-width="2" stroke-linejoin="round"/>')
last = it[-1]
parts.append(f'<circle cx="{X(last["games_total"]):.1f}" cy="{Y(sm[-1]):.1f}" r="4" fill="{ACC}"/>')
parts.append(f'<text x="{L}" y="{T - 8}" fill="{INK}" font-weight="600" font-size="13">Rolling Elo on the frozen ladder (32-simulation search, ±1 standard error)</text>')

# --- centipawn-loss panel ---
c0, c1 = 60, 200
cT = T + top_h + GAP
CY = lambda c: cT + H2 * (1 - (c - c0) / (c1 - c0))
for c in (100, 150, 200):
    parts.append(f'<line x1="{L}" x2="{L + PW}" y1="{CY(c):.1f}" y2="{CY(c):.1f}" stroke="{GRID}"/>'
                 f'<text x="{L - 8}" y="{CY(c) + 4:.1f}" text-anchor="end" fill="{MUTED}">{c}</text>')
for c, label in ((194, "random mover"), (123, "material grabber")):
    parts.append(f'<line x1="{L}" x2="{L + PW}" y1="{CY(c):.1f}" y2="{CY(c):.1f}" stroke="{MUTED}" stroke-dasharray="3 4"/>'
                 f'<text x="{L + PW + 6}" y="{CY(c) + 4:.1f}" fill="{INK2}" font-size="11">{label}</text>')
csm = [sum(x["acpl_policy"] for x in cpl[max(0, i - 4):i + 5]) / len(cpl[max(0, i - 4):i + 5]) for i in range(len(cpl))]
cline = "M" + " L".join(f"{X(r['games_total']):.1f},{CY(min(v, c1)):.1f}" for r, v in zip(cpl, csm))
parts.append(f'<path d="{cline}" fill="none" stroke="{TEAL}" stroke-width="2" stroke-linejoin="round"/>')
parts.append(f'<text x="{L}" y="{cT - 8}" fill="{INK}" font-weight="600" font-size="13">Centipawn loss of the raw network, no search (lower is better)</text>')

# --- x axis ---
for g in range(0, xmax + 1, 50_000):
    parts.append(f'<text x="{X(g):.1f}" y="{H - 14}" text-anchor="middle" fill="{MUTED}">{g // 1000}k</text>')
parts.append(f'<text x="{L + PW}" y="{H - 2}" text-anchor="end" fill="{INK2}" font-size="11">self-play games</text>')
parts.append("</svg>")
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text("\n".join(parts), encoding="utf-8")
print(f"{dst}: {len(it)} Elo rows, {len(cpl)} CPL rows, final smoothed Elo {sm[-1]:.0f}")
