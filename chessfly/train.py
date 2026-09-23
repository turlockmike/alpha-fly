"""AlphaZero loop on one GPU: self-play -> replay buffer -> train on (MCTS policy, game result) -> repeat.

    uv run python -m chessfly.train --run fly
    uv run python -m chessfly.train --run shuffled --shuffled      # rewired control, same budget

Pause with Ctrl+C (once) or by creating runs/<name>/PAUSE: the current step finishes, everything is
saved, the process exits.  The same command picks the run up again, with the settings it was
started with unless a flag says otherwise.  runs/<name>/log.jsonl has one JSON object per line:
"type": "iter" rows with every metric, and "type": "event" rows for start / resume / pause.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime
import json
import os
import pickle
import random
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .arena import evaluate_strength
from .ladder import find_stockfish, load_ladder
from .connectome import load_malecns
from .encoding import N_FEATURES, N_MOVES, upgrade_features
from .mcts import MCTSConfig
from .model import Evaluator, FlyChessNet, ModelConfig, load_weights
from .selfplay import Sample, SelfPlayConfig, write_selfplay_pgn
from .suite import load_suite, policy_metrics, search_metrics
from .workers import make_selfplay


@dataclasses.dataclass
class TrainConfig:
    games_per_iter: int = 256
    buffer_size: int = 2_000_000     # about an hour of self-play at 500 positions/s; ~5 GB of RAM
    batch_size: int = 256
    sample_reuse: float = 1.0        # average number of times a position is trained on.  lc0 trains on ~1 in 32
                                     # positions of a 100k-game window; at 4 this net memorised its MCTS targets
                                     # (held-out policy loss 8.3 against 3.1 on the training set)
    min_buffer: int = 50_000
    lr_shared: float = 3e-5          # the policy head's shared from-square / to-square terms.  1e-3 was tried and broke the policy
                                     # within 4 iterations (held-out policy loss 2.87 -> 3.71, Elo 404 -> 364): Adam moves each
                                     # of a term's 2,333 input weights by ~lr per step, and every move touching the square pushes
                                     # the same way, so the term swung whole groups of logits by ~2 per step.
    q_ratio: float = 0.5             # value loss = (1-r) CE(game result) + r MSE(expected value, Sample.q): lc0's q_ratio.
                                     # Sample.q is the TD(lambda) average of later search values (selfplay.td_targets), a dense
                                     # target that is not shared by every position of a game
    policy_target_offset: float = 0.0   # visits subtracted from every move before the policy target is normalised (KataGo's
                                     # policy target pruning, lc0's distOffset).  At 32 simulations over ~30 legal moves the
                                     # search gives almost every move its one exploratory visit, the raw target is ~3/32 for
                                     # the best move against 1/32 for the rest, and the policy learns next to nothing; 1.0
                                     # removes those forced visits.  Applies to positions already in the buffer too.
                                     # TRIED at iterations 22-27 and reverted: Elo fell ~205 -> ~165, a head-to-head against
                                     # the same rung went 61% -> 53%, and held-out policy loss rose.  With 32 simulations,
                                     # *which* moves get a second visit is mostly noise, so pruning sharpened noise.
    lr_brain: float = 3e-3           # per-synapse log-gains, per-neuron bias and leak.  Rates are constant; drop them by hand on
                                     # resume.  TRIED at iteration 159 of fly6 (strength flat at ~720 Elo for 60 iterations): all
                                     # rates / 3 for an hour took held-out policy loss 2.93 -> 2.33, value-vs-Stockfish 0.80 ->
                                     # 0.83 and closed the train/held-out gap, but playing strength did not move (42.1% -> 43.5%
                                     # over 600 games, +-2.8; runs/ab3).  Kept: nothing got worse.  The plateau was not the rate.
    lr_io: float = 3e-4              # encoder and heads
    io_decay: float = 1e-2
    gain_decay: float = 1e-4         # pulls log-gains back towards the measured synapse counts
    val_fraction: float = 0.05       # of finished positions, held out to expose memorisation of game results
    arena_games: int = 64            # ladder games played continuously next to self-play, in the same GPU batches (~6% of
                                     # the load); Elo is then logged every iteration.  0 = a blocking match every eval_every
    arena_sims: int = 32             # Elo is always measured with a 32-simulation Gumbel search, whatever self-play uses,
                                     # so that readings stay comparable when the training search changes
    arena_window: int = 200          # Elo is estimated from this many most recent evaluation games (resume with 400 for the
                                     # five-rung panel: 80 games per rung)
    arena_min_games: int = 100       # after a restart the window refills with the games that finish first - the quick losses -
                                     # so Elo is only logged once the window holds this many (it read 38 +/- 60, then 164)
    eval_every: int = 10             # Elo match and search centipawn loss; policy centipawn loss is logged every iteration
    eval_games: int = 32             # against each of the two ladder rungs that bracket the current rating
    save_every: int = 10             # network + optimiser + counters (runs/<name>/last.pt)
    buffer_save_every: int = 50      # the replay buffer is several GB; it is also saved on every pause
    snapshot_every: int = 100        # keep the network of this iteration in runs/<name>/snapshots/


def shared_group(net: FlyChessNet, cfg: TrainConfig) -> dict:
    return {"params": [*net.policy_from.parameters(), *net.policy_to.parameters()], "lr": cfg.lr_shared, "weight_decay": cfg.io_decay}


def make_optimizer(net: FlyChessNet, cfg: TrainConfig, with_shared: bool = True) -> torch.optim.Optimizer:
    """with_shared=False rebuilds the optimiser of a checkpoint saved before the shared policy terms existed, so that its
    Adam state loads; the caller then adds the group."""
    brain = net.brain
    return torch.optim.AdamW([
        {"params": [brain.log_gain], "lr": cfg.lr_brain, "weight_decay": cfg.gain_decay},
        {"params": [brain.bias, brain.alpha_logit], "lr": cfg.lr_brain, "weight_decay": 0.0},
        {"params": [*net.encoder.parameters(), *net.policy.parameters(), *net.value.parameters()],
         "lr": cfg.lr_io, "weight_decay": cfg.io_decay},
    ] + ([shared_group(net, cfg)] if with_shared else []))


def prune(pi: np.ndarray, offset: float) -> np.ndarray:
    """pi minus `offset` visits, renormalised.  One visit is the smallest non-zero entry of a visit distribution."""
    pi = pi.astype(np.float32)
    if offset <= 0:
        return pi
    t = np.maximum(pi - offset * pi[pi > 0].min(), 0)
    return t / t.sum() if t.sum() > 0 else pi


def collate(batch: list[Sample], device, policy_target_offset: float = 0.0):
    x = torch.from_numpy(np.stack([s.x for s in batch]).astype(np.float32)).to(device)
    pi = np.zeros((len(batch), N_MOVES), dtype=np.float32)
    legal = np.zeros((len(batch), N_MOVES), dtype=bool)
    for i, s in enumerate(batch):
        pi[i, s.idx] = prune(s.pi, policy_target_offset)
        legal[i, s.idx] = True
    wdl = torch.tensor([s.wdl for s in batch], device=device)
    q = torch.tensor([s.q for s in batch], dtype=torch.float32, device=device)
    return x, torch.from_numpy(pi).to(device), torch.from_numpy(legal).to(device), wdl, q


def train_step(net, opt, batch, device, q_ratio: float = 0.0, policy_target_offset: float = 0.0) -> dict:
    x, pi, legal, wdl, q = collate(batch, device, policy_target_offset)
    logits, wdl_logits = net(x)
    logp = logits.masked_fill(~legal, -1e9).log_softmax(-1)      # illegal moves are masked, as in the search
    policy_loss = -(pi * logp).sum(-1).mean()
    value_ce = F.cross_entropy(wdl_logits, wdl)
    value_loss = (1 - q_ratio) * value_ce + q_ratio * F.mse_loss(net.expected_value(wdl_logits), q)
    opt.zero_grad(set_to_none=True)
    (policy_loss + value_loss).backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    opt.step()
    # value_ce is the part comparable with val_value_loss (the held-out loss is the plain cross-entropy)
    return {"policy_loss": policy_loss.item(), "value_loss": value_loss.item(), "value_ce": value_ce.item()}


@torch.no_grad()
def validation_loss(net, val: list[Sample], device, policy_target_offset: float = 0.0) -> dict:
    net.eval()
    tot = {"val_policy_loss": 0.0, "val_value_loss": 0.0}
    for i in range(0, len(val), 256):
        x, pi, legal, wdl, _ = collate(val[i:i + 256], device, policy_target_offset)
        logits, wdl_logits = net(x)
        logp = logits.float().masked_fill(~legal, -1e9).log_softmax(-1)
        tot["val_policy_loss"] += -(pi * logp).sum().item()
        tot["val_value_loss"] += F.cross_entropy(wdl_logits.float(), wdl, reduction="sum").item()
    return {k: round(v / len(val), 4) for k, v in tot.items()}


CONFIGS = {"model": ModelConfig, "mcts": MCTSConfig, "selfplay": SelfPlayConfig, "train": TrainConfig}


def parse_config(run: Path, argv=None):
    """Settings = dataclass defaults <- the run's config.json (when resuming) <- flags given on this command line."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fly")
    ap.add_argument("--iters", type=int, default=1_000_000, help="stop after this iteration")
    ap.add_argument("--data", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init_from", default=None, help="new run only: start from this checkpoint's weights (any step count)")
    ap.add_argument("--init_buffer", default=None, help="new run only: start with this replay buffer (runs/<name>/buffer.pkl)")
    for cls in CONFIGS.values():
        for f in dataclasses.fields(cls):
            kind = {"action": "store_true"} if f.type == "bool" else {"type": type(f.default)}
            ap.add_argument(f"--{f.name}", default=argparse.SUPPRESS, **kind)
    args = vars(ap.parse_args(argv))
    run = run / args["run"]
    saved = json.loads((run / "config.json").read_text()) if (run / "config.json").exists() else {}
    cfgs, overrides = {}, {}
    for section, cls in CONFIGS.items():
        values = {f.name: f.default for f in dataclasses.fields(cls)} | saved.get(section, {})
        for k in values:
            if k in args:
                if section in saved and saved[section].get(k) != args[k]:
                    overrides[k] = [saved[section].get(k), args[k]]
                values[k] = args[k]
        cfgs[section] = cls(**{k: v for k, v in values.items() if k in {f.name for f in dataclasses.fields(cls)}})
    return args, run, cfgs, overrides


def atomic_write(path: Path, write) -> None:
    """A pause or a crash in the middle of a save must not cost the previous good file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    write(tmp)
    os.replace(tmp, path)


class Pause:
    """First Ctrl+C (or a runs/<name>/PAUSE file) asks for a clean stop; a second Ctrl+C kills."""

    def __init__(self, run: Path):
        self.file, self.requested = run / "PAUSE", False
        self.file.unlink(missing_ok=True)
        signal.signal(signal.SIGINT, self._on_sigint)

    def _on_sigint(self, *_):
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True
        print("\npausing: finishing the current step and saving (Ctrl+C again to quit without saving)", flush=True)

    def __call__(self) -> bool:
        self.requested = self.requested or self.file.exists()
        return self.requested


def grow_optimizer_state(opt_state: dict, opt: torch.optim.Optimizer) -> dict:
    """Adam moments of a parameter that gained input columns (the encoder, when features were added): pad with zeros."""
    params = [p for g in opt.param_groups for p in g["params"]]
    for i, st in opt_state["state"].items():
        for k, v in st.items():
            if torch.is_tensor(v) and v.ndim == 2 and v.shape[0] == params[i].shape[0] and v.shape[1] < params[i].shape[1]:
                st[k] = torch.cat([v, v.new_zeros(v.shape[0], params[i].shape[1] - v.shape[1])], dim=1)
    return opt_state


def upgrade_buffer(samples: list[Sample]) -> list[Sample]:
    old = [s for s in samples if len(s.x) != N_FEATURES]
    if old:
        print(f"adding the attack planes to {len(old):,} stored positions ...", flush=True)
        for s in old:
            s.x = upgrade_features(s.x)
    return samples


def main():
    args, run, cfgs, overrides = parse_config(Path("runs"))
    mcfg, scfg, pcfg, tcfg = cfgs["model"], cfgs["mcts"], cfgs["selfplay"], cfgs["train"]
    torch.manual_seed(args["seed"]); random.seed(args["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run.mkdir(parents=True, exist_ok=True)

    def log(row: dict) -> None:
        row = {"type": row.pop("type", "iter"), "time": datetime.datetime.now().isoformat(timespec="seconds")} | row
        print(json.dumps(row), flush=True)
        with open(run / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    net = FlyChessNet(load_malecns(args["data"], mcfg.min_syn), mcfg).to(device)
    opt = make_optimizer(net, tcfg)
    buffer: deque[Sample] = deque(maxlen=tcfg.buffer_size)
    val: deque[Sample] = deque(maxlen=4096)
    state = {"iter": 0, "owed": 0.0, "elo": None, "elapsed_s": 0.0, "positions_total": 0, "train_steps_total": 0,
             "stats": None}
    resumed = (run / "last.pt").exists()
    if resumed:
        ck = torch.load(run / "last.pt", map_location=device, weights_only=False)
        load_weights(net, ck["model"])
        if len(ck["opt"]["param_groups"]) == 3:                  # saved before the shared policy terms existed
            opt = make_optimizer(net, tcfg, with_shared=False)
            opt.load_state_dict(grow_optimizer_state(ck["opt"], opt))
            opt.add_param_group(shared_group(net, tcfg))
        else:
            opt.load_state_dict(grow_optimizer_state(ck["opt"], opt))
        state |= {k: ck[k] for k in state if k in ck}
        for group, lr in zip(opt.param_groups, (tcfg.lr_brain, tcfg.lr_brain, tcfg.lr_io, tcfg.lr_shared)):
            group["lr"] = lr                                     # a learning rate given on resume takes effect
        if (run / "buffer.pkl").exists():
            saved = pickle.loads((run / "buffer.pkl").read_bytes())
            buffer.extend(upgrade_buffer(saved["buffer"])); val.extend(upgrade_buffer(saved["val"]))
    elif args["init_from"] or args["init_buffer"]:
        # Bootstrapping a run from another: the parameters are the same at any number of steps (the same synapses,
        # unrolled further), and self-play positions are valid training data for any network.  4-step weights run
        # for 6 steps kept value-vs-Stockfish correlation 0.635 of 0.654 once the output normaliser was refreshed.
        if args["init_from"]:
            load_weights(net, torch.load(args["init_from"], map_location=device, weights_only=False)["model"])
        if args["init_buffer"]:
            saved = pickle.loads(Path(args["init_buffer"]).read_bytes())
            buffer.extend(upgrade_buffer(saved["buffer"])); val.extend(upgrade_buffer(saved["val"]))
        if args["init_from"] and len(buffer) >= 2048:           # the readout statistics belong to the old depth: refresh them
            net.train(); net.norm.initialised.fill_(False)
            with torch.no_grad():
                pool = list(buffer)
                for _ in range(8):
                    net(collate(random.sample(pool, 256), device)[0])
    atomic_write(run / "config.json", lambda p: p.write_text(json.dumps({k: dataclasses.asdict(v) for k, v in cfgs.items()}, indent=1)))
    suite = load_suite()
    log({"type": "event", "event": "resume" if resumed else "start", "iter": state["iter"], "buffer": len(buffer),
         "overrides": overrides, "config": {k: dataclasses.asdict(v) for k, v in cfgs.items()},
         "neurons": net.brain.n, "synapses": net.brain.pre.numel(), "parameters": sum(p.numel() for p in net.parameters()),
         "device": str(device), "centipawn_suite": suite is not None,
         "init_from": None if resumed else args["init_from"], "init_buffer": None if resumed else args["init_buffer"]})
    if suite is None:
        print("no eval/suite.json: centipawn loss is not logged (scripts/build_eval_suite.py builds it)")

    def save(with_buffer: bool) -> None:
        atomic_write(run / "last.pt", lambda p: torch.save(
            {"model": net.state_dict(), "opt": opt.state_dict(), "config": dataclasses.asdict(mcfg)} | state, p))
        if with_buffer:
            atomic_write(run / "buffer.pkl", lambda p: p.write_bytes(
                pickle.dumps({"buffer": list(buffer), "val": list(val)}, protocol=5)))

    pause = Pause(run)
    rolling = tcfg.arena_games > 0 and pcfg.workers > 0 and load_ladder() is not None and find_stockfish() is not None
    eval_mcts = dataclasses.replace(scfg, sims=tcfg.arena_sims, gumbel=True)
    arena = {"games": tcfg.arena_games, "window": tcfg.arena_window, "elo_guess": state["elo"], "mcts": eval_mcts} if rolling else None
    selfplay = make_selfplay(Evaluator(net, device), scfg, pcfg, seed=args["seed"] + state["iter"], arena=arena)
    if state["stats"]:
        selfplay.stats.update(state["stats"])
    try:
        while state["iter"] < args["iters"] and not pause():
            t0 = time.time()
            before = dict(selfplay.stats)
            new = selfplay.play(tcfg.games_per_iter, should_stop=pause)
            if pause():
                buffer.extend(new)                               # keep the finished games, do not count an iteration
                break
            state["iter"] += 1
            it = state["iter"]
            held = [random.random() < tcfg.val_fraction for _ in new]
            val.extend(s for s, h in zip(new, held) if h)
            new = [s for s, h in zip(new, held) if not h]
            buffer.extend(new)
            t1 = time.time()

            losses = []
            if len(buffer) >= tcfg.min_buffer:
                net.train()
                state["owed"] += len(new) * tcfg.sample_reuse / tcfg.batch_size
                pool = list(buffer)
                while state["owed"] >= 1 and not pause():
                    state["owed"] -= 1
                    losses.append(train_step(net, opt, random.sample(pool, tcfg.batch_size), device, tcfg.q_ratio,
                                             tcfg.policy_target_offset))
            t2 = time.time()

            d = {k: selfplay.stats[k] - before[k] for k in before}
            if selfplay.stats.records:
                write_selfplay_pgn(run / "selfplay" / f"iter_{it:06d}.pgn", selfplay.stats.records, run.name, it)
                selfplay.stats.records = []
            state["positions_total"] += len(new)
            state["train_steps_total"] += len(losses)
            row = {"iter": it, "games_total": selfplay.stats["games"], "positions_total": state["positions_total"],
                   "train_steps_total": state["train_steps_total"], "buffer": len(buffer),
                   "games": d["games"], "positions": len(new), "avg_plies": round(d["plies"] / max(d["games"], 1), 1),
                   "white_wins": d["white"], "draws": d["draw"], "black_wins": d["black"], "adjudicated": d["adjudicated"],
                   "selfplay_s": round(t1 - t0, 1), "train_s": round(t2 - t1, 1), "train_steps": len(losses),
                   "selfplay_positions_per_s": round(d["moves"] / max(t1 - t0, 1e-9))}       # moves played, not games finished
            if losses:
                row |= {k: round(float(np.mean([l[k] for l in losses])), 4) for k in losses[0]}
                if len(val) >= 256:
                    row |= validation_loss(net, list(val), device, tcfg.policy_target_offset)
            evaluator = Evaluator(net, device)
            if suite:
                row |= policy_metrics(evaluator, suite)
            if it % tcfg.eval_every == 0 and not pause():
                if suite:
                    row |= search_metrics(evaluator, suite, eval_mcts)
                if not rolling:
                    row |= evaluate_strength(evaluator, eval_mcts, tcfg.eval_games, seed=it, elo_guess=state["elo"],
                                             pgn_dir=run / "games", tag=f"iter_{it:06d}")
            if rolling and (reports := selfplay.take_arena_reports()):
                (run / "games").mkdir(exist_ok=True)
                (run / "games" / f"iter_{it:06d}_rolling.pgn").write_text("".join(r.pop("pgn") for r in reports), encoding="utf-8")
                latest = reports[-1] | {"eval_games_finished": len(reports)}
                if latest["elo_games"] < tcfg.arena_min_games:
                    latest = {k: v for k, v in latest.items() if k not in ("elo", "elo_se")}
                row |= latest
            state["elo"] = row.get("elo", state["elo"])
            state["elapsed_s"] += time.time() - t0
            state["stats"] = dict(selfplay.stats)
            log(row | {"eval_s": round(time.time() - t2, 1), "elapsed_s": round(state["elapsed_s"])})
            if it % tcfg.save_every == 0:
                save(with_buffer=it % tcfg.buffer_save_every == 0)
            if it % tcfg.snapshot_every == 0:
                (run / "snapshots").mkdir(exist_ok=True)
                torch.save({"model": net.state_dict(), "config": dataclasses.asdict(mcfg), "iter": it, "elo": state["elo"]},
                           run / "snapshots" / f"iter_{it:06d}.pt")
    finally:
        selfplay.close()
    state["stats"] = dict(selfplay.stats)
    print(f"saving {len(buffer):,} positions ...", flush=True)
    save(with_buffer=True)
    pause.file.unlink(missing_ok=True)
    log({"type": "event", "event": "pause" if pause.requested else "finish", "iter": state["iter"], "buffer": len(buffer)})


if __name__ == "__main__":
    main()
