#!/usr/bin/env bash
# Event stream for a training run: one line per thing worth knowing.
#   bash scripts/watch_run.sh [run] [heartbeat_seconds]
# Emits: errors from the console log as they appear, STALLED when the log goes quiet, STOPPED on
# pause/finish, MILESTONE when a full Elo window first clears a ladder rung of the goal (661 / 879 / 1179 / 1320;
# GOAL when 1320 has held over 3 consecutive non-overlapping 200-game windows), and a one-line status on every
# heartbeat boundary of the wall clock (default 45 min), so the cadence survives restarts of this script.
cd "$(dirname "$0")/.." || exit 1
RUN=${1:-fly}; BEAT=${2:-2700}
LOG=runs/$RUN/log.jsonl; CON=runs/$RUN/console.log
ERR='Traceback|CUDA error|out of memory|Error:'     # not "Exception": PowerShell tags every stderr line RemoteException
console() { [ -f "$CON" ] && tr -d '\000' < "$CON"; }          # PowerShell's Tee-Object writes UTF-16

status() {
  python - "$LOG" <<'PY'
import json, sys, datetime
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
it = [r for r in rows if r.get("type") == "iter"]
if not it:
    print("STATUS no iteration finished yet"); sys.exit()
r, e = it[-1], next((x for x in reversed(it) if "elo" in x), None)
age = (datetime.datetime.now() - datetime.datetime.fromisoformat(r["time"])).total_seconds() / 60
g = max(r.get("games", 1), 1)
f = lambda k, d=3: "-" if r.get(k) is None else f"{r[k]:.{d}f}"
print(f"STATUS iter {r['iter']} | {r['games_total']:,} games | {r['elapsed_s'] / 3600:.1f} h | "
      f"Elo {'-' if e is None else str(e['elo']) + ' +/- ' + str(e['elo_se']) + ' @' + str(e['iter'])} | "
      f"{' '.join(k[3:].replace('sf_', '') + ' ' + format(v, '.0%') for k, v in sorted((e or {}).items()) if k.startswith('vs_'))} | "
      f"CPL net {f('acpl_policy', 1)} search {'-' if e is None or 'acpl_search' not in e else e['acpl_search']} | top1 {f('top1_policy')} | "
      f"value corr {f('value_corr')} | policy {f('policy_loss')}/{f('val_policy_loss')} value {f('value_loss')}/{f('val_value_loss')} | "
      f"draws {r.get('draws', 0) / g:.0%} mates {(r.get('white_wins', 0) + r.get('black_wins', 0) - r.get('adjudicated', 0)) / g:.0%} | "
      f"{r.get('selfplay_positions_per_s')} pos/s | last row {age:.0f} min ago")
PY
}

milestones() {            # prints every milestone reached so far, one per line; the loop reports the new ones
  python - "$LOG" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
it = [r for r in rows if r.get("type") == "iter"]
# a milestone = scoring at least 50% against that rung itself over >= 60 games of the window (the ladder is not
# transitive for the fly, so an Elo number that mixes rungs is not trusted for this)
for rung, name in ((661, "sf_mix75"), (879, "sf_mix90"), (1179, "sf_skill0"), (1320, "sf_1320")):
    hit = next((r for r in it if r.get(f"n_{name}", 0) >= 60 and r.get(f"vs_{name}", 0) >= 0.5), None)
    if hit:
        print(f"MILESTONE {rung} ({name}): scored {hit['vs_' + name]:.0%} over {hit['n_' + name]} games at iteration {hit['iter']}, {hit['games_total']:,} games, {hit['elapsed_s'] / 3600:.1f} h")
# the target: >= 50% against Stockfish UCI_Elo 1320 in 3 consecutive windows that share no games
streak, since = 0, 0
for r in it:
    since += r.get("eval_games_finished", 0)
    if since >= r.get("elo_games", 10**9) and r.get("n_sf_1320", 0) >= 60:
        since = 0
        streak = streak + 1 if r.get("vs_sf_1320", 0) >= 0.5 else 0
        if streak == 3:
            print(f"GOAL parity with Stockfish 1320 held over 3 consecutive windows, at iteration {r['iter']} ({r['vs_sf_1320']:.0%})"); break
PY
}

seen=$(console | grep -cE "$ERR"); seen=${seen:-0}
reported=$(milestones 2>/dev/null)
slot=$(( $(date +%s) / BEAT )); stale=0; stopped=0
echo "WATCHING $RUN (heartbeat every $(( BEAT / 60 )) min)"; status
while true; do
  now=$(date +%s)
  n=$(console | grep -cE "$ERR"); n=${n:-0}
  if [ "$n" -gt "$seen" ]; then
    echo "ERROR in console.log: $(console | grep -E "$ERR" | tail -n 3 | tr '\n' ' ' | cut -c1-400)"; seen=$n
  fi
  if [ -f "$LOG" ]; then
    if tail -n 1 "$LOG" | grep -qE '"event": "(pause|finish)"'; then       # report a stop as soon as it is logged, once
      if [ "$stopped" != 1 ]; then stopped=1; echo "STOPPED: $(tail -n 1 "$LOG" | cut -c1-160)"; fi
    else stopped=0; fi
    age=$(( now - $(stat -c %Y "$LOG") ))
    if [ "$age" -gt 1500 ] && [ "$stale" = 0 ]; then
      stale=1
      if tail -n 1 "$LOG" | grep -qE '"event": "(pause|finish)"'; then :
      else echo "STALLED: no log line for $(( age / 60 )) min; python processes: $(tasklist | grep -ci python.exe)"; fi
    fi
    [ "$age" -le 1500 ] && stale=0
  fi
  now_reached=$(milestones 2>/dev/null)
  if [ "$now_reached" != "$reported" ]; then
    comm -13 <(echo "$reported") <(echo "$now_reached"); reported=$now_reached
  fi
  s=$(( now / BEAT ))
  if [ "$s" != "$slot" ]; then slot=$s; status; fi
  sleep 30
done
