# Training notes

The engineering notes behind [alpha-fly](README.md): what the model is, how to run and resume training, what is logged, how strength is measured, what it costs, and what differs from AlphaZero and why.
AlphaZero-style self-play training of a chess network whose hidden computation is the
**complete male fruit-fly connectome** (MaleCNS v1.0: 166,700 neurons, 10.5M connections at
≥3 synapses, from HHMI Janelia / Google Research / Cambridge).

No human games, no Stockfish labels: the only teacher is the network's own MCTS.

## Model

```
board (838 features, side-to-move view)
  └─ linear ─► 17,937 sensory neurons
                 │  4 recurrent steps through the fixed wiring
                 │  h ← (1-α)h + α·relu(W h + b + input)
                 ▼
               2,333 output neurons (motor / efferent / descending)
  ├─ linear ─► 4,168 move logits
  └─ linear ─► win / draw / loss
```

Fixed by the fly: who connects to whom, excitatory/inhibitory sign per neuron (Dale's law, from
the predicted transmitter), and the initial weight (synapse count, normalised per target neuron).
Trained: one positive gain per connection (10.5M), a bias and a leak per neuron, and the linear
encoder and heads (35.6M parameters in total). Encoder and heads are single linear maps, so all
nonlinear computation happens inside the connectome.

Three synapses connect sensory neurons to 99.7% of the output neurons, hence 4 steps.

`--shuffled` trains the control: same neurons, degrees and signs, random partners. Run it with
the same budget before claiming that the fly's wiring matters. `scripts/compare_runs.py <fly run>
<shuffled run>` reads the two logs (no GPU) and prints the Elo gap, with its error, at matched game counts.

## Run

```
uv sync --extra dev
uv run pytest                                   # kernel gradients vs dense, encoding, MCTS, workers, ratings, pause/resume
uv run python -m chessfly.train --run fly6 --steps 6 --gumbel --lazy --sample_reuse 2     # the main run's settings; downloads MaleCNS on first use
uv run python -m chessfly.train --run shuffled --shuffled
uv run python -m chessfly.dashboard             # live charts on http://127.0.0.1:8765
```

Every dataclass field in `model.py`, `mcts.py`, `selfplay.py` and `train.py` is a flag.

### Pause, resume, weights

* **Pause:** Ctrl+C once, or create `runs/<name>/PAUSE` from anywhere. The current step finishes,
  everything is saved, the process exits (a second Ctrl+C quits without saving). Games that were
  still in progress are dropped; finished ones are kept.
* **Bootstrap:** `--init_from <checkpoint> --init_buffer <buffer.pkl>` start a *new* run from another run's weights and
  positions. The parameters are the same at any step count: 4-step weights run for 6 steps kept 0.635 of their 0.654
  value-vs-Stockfish correlation once the output normaliser was refreshed, which is how the 6-step main run began.
* **Resume:** the same command. Settings come from `runs/<name>/config.json`, so
  `--run fly` alone continues with whatever the run was started with; a flag given on resume
  overrides it and the change is recorded in the log (`--sims 64` is how you raise the search
  budget mid-run). Counters, optimiser state, the replay buffer and the Elo estimate carry over.
* **Weights:** `runs/<name>/last.pt` (network + optimiser, every `--save_every` iterations and on
  pause; written atomically) and `runs/<name>/snapshots/iter_*.pt` (network only, 140 MB, every
  `--snapshot_every` iterations). `chessfly.model.load_net(path)` returns a ready network from either.
* **Playing a checkpoint:** `python -m chessfly.uci <checkpoint>` is a UCI engine, so any chess GUI
  (Arena, Cute Chess, Banksia) or lichess-bot can load it. Engine command:
  `uv run --project <path to this repository> python -m chessfly.uci runs/fly/last.pt`. It searches a
  fixed number of simulations per move (option `Simulations`, default 200; about 1 s per 100 while
  training shares the GPU) and ignores the clock. Two snapshots in a GUI tournament is the simplest
  way to see how much a day of training bought.

### What is logged

`runs/<name>/log.jsonl`, one JSON object per line. `"type": "event"` rows mark start / resume /
pause / finish with the full configuration; `"type": "iter"` rows carry, per iteration:

| | fields |
| --- | --- |
| progress | `iter`, `time`, `elapsed_s`, `games_total`, `positions_total`, `train_steps_total`, `buffer` |
| self-play | `games`, `positions`, `avg_plies`, `white_wins`, `draws`, `black_wins`, `adjudicated`, `selfplay_positions_per_s` (moves actually played, not games finished) |
| learning | `policy_loss`, `value_loss`, and the same on held-out positions: `val_policy_loss`, `val_value_loss` |
| centipawn loss, every iteration | `acpl_policy`, `top1_policy` (raw network), `value_corr` (value head vs Stockfish's evaluation) |
| Elo, every iteration | `elo`, `elo_se`, `elo_games`, `vs_<opponent>`, `mates` over the rolling window; `eval_games_finished` |
| every `--eval_every` iterations | `acpl_search`, `top1_search` (network + MCTS) |

Games are kept as PGN: every evaluation game in `runs/<name>/games/`, and a random
`--record_fraction` (1%) of the self-play games in `runs/<name>/selfplay/`, one file per iteration,
with how each game ended in its `Termination` header. The dashboard shows the ten most recent of
either kind on a board you can step through (arrow keys, Home / End, autoplay); the server replays
the PGN with python-chess, so the page needs no chess library and works offline.

`http://127.0.0.1:8765/play` (linked from the dashboard header) lets you play the newest saved network
of any run: pick a colour and a search budget (1 simulation = the raw network, 32 = the budget the Elo
is measured at), click a piece and a highlighted square. It uses the same Gumbel search as training,
reloads `last.pt` whenever the trainer saves a newer one, shows the moves the fly considered with their
visits and values, and copies the game as PGN. It shares the GPU with a running trainer (about 0.5 s a
move at 100 simulations); the first move waits ~8 s for the network to load.

### Elo and centipawn loss

Stockfish is a yardstick only; nothing it produces reaches the training data. Put the binary at
`tools/stockfish.exe` or set `STOCKFISH`.

* **Centipawn loss** (`eval/suite.json`, built by `scripts/build_eval_suite.py`): Stockfish 17.1's
  depth-10 evaluation of *every legal move* in 1,000 positions taken from games between players of
  mixed strength. Scoring a network is then a table lookup, cheap enough for every iteration.
  For scale, a random mover loses 194 centipawns per move on this suite and a material grabber 123.
* **Elo** (`eval/ladder.json`, built by `scripts/calibrate_ladder.py`): Stockfish's own scale
  starts at 1320, far above a young network, so a ladder of fixed opponents fills the gap: a
  random mover, Stockfish diluted with 90-10% random moves, Stockfish skill 0, then UCI_Elo 1320
  and 1600. The rungs were rated against each other once, anchored at UCI_Elo 1320 := 1320.
  An arena worker (`--arena_games 64`) plays the two rungs that bracket the current rating
  *continuously, next to self-play*: its search leaves ride in the same GPU batches (~6% of the
  load) and its Stockfish time never holds up a self-play round, so nothing waits for a match.
  `elo` is then a rolling maximum-likelihood estimate over the last `--arena_window` (200)
  evaluation games, logged every iteration with its standard error; consecutive points share most
  of their games, and a window spans a few network versions. Re-calibrating the ladder changes the
  scale: do it before a run, not during one.

## Cost (RTX 5080)

| | 4 steps | 6 steps |
| --- | --- | --- |
| synapses that can carry the position to an output neuron | 22.8% | 91.6% |
| network inference, float16, positions / s | 80,000 | 19,000 |
| self-play, 1120 games, 14 search processes, 32 simulations per move, positions / s | 1,150 (CPU-bound, before lazy search) | 735 (GPU-bound) |

The first version of this code played 36 positions a second. What bought the rest, largest first:

* **The light cone** (`ConnectomeRNN.forward_io`, `LightCone` in `brain.py`). The state starts at zero and the position
  enters only through the sensory neurons, so after *t* steps only neurons within *t*-1 synapses of an input can differ
  between positions; the rest follow one trajectory, computed once per weight version. And only neurons within *T*-*t*
  synapses of an output can still matter. Per position the network needs a sub-matrix of W per step; the rest of W
  enters as a constant drive. At 4 steps that is 3.0M synapse visits instead of 42M (the first step needs none: it
  multiplies a zero state) - 9x faster inference, 3.8x faster training, *the same function*: a test holds outputs and
  all gradients equal to the literal recurrence. It is also why the step count matters: at 4 steps three quarters of
  the fly's synapses are only trainable constants, hence the main run's 6.
* **State stored (neurons, batch), not (batch, neurons).** The recurrent step is a cuSPARSE SpMM; with the state the
  other way round every synapse reads strided memory and the step is 15x slower. Its cost steps every 128 batch
  columns (any batch of 513-640 costs the same), so pools are sized to fill a step: 8 x 70 or 14 x 40 leaves + the arena's 64.
* **Tree search in worker processes** (`workers.py`): python-chess costs more per leaf than the network does.
  Processes own the games, each searching one pool while its other pool is evaluated; the trainer launches a batch
  on the GPU and assembles the next one while it computes.
* **Lazy expansion** (`--lazy`): half of a leaf's CPU time is generating legal moves, and at tens of simulations ~70%
  of leaves are never visited again. A leaf is evaluated for its value and whole policy vector (returned through
  shared memory: 8 KB per leaf is too much for Windows' 8 KB pipes); its moves are generated only if the search
  returns. 151 -> 84 us per leaf, identical trees (tested).
* **Per-synapse gradient through cuSPARSE SDDMM** (`torch.sparse.sampled_addmm`, in 64-column blocks) instead of
  two 10 GB gathers per step.

`scripts/` has the measurements behind each of these: `benchmark.py`, `bench_kernel.py`, `bench_evaluator.py`,
`bench_light_cone.py`, `profile_selfplay.py`, `profile_python.py`.

## What differs from AlphaZero, and why

Borrowed from [lc0](https://github.com/LeelaChessZero/lc0) (values read from its source): self-play
search settings (cpuct 1.2 constant, unvisited children valued at the parent's Q, Dirichlet 0.3 at
25%), WDL value head, `--temp_visit_offset` so a move with a single exploratory visit is never
played, and `--q_ratio`, which mixes the search value into the value target.

Forced by a budget of tens of simulations per move instead of 800:

* **Gumbel AlphaZero search and targets** (`--gumbel`; Danihelka et al., ICLR 2022). With 32 simulations over ~25
  legal moves, PUCT's visit counts are nearly flat (the top move got 28% of the target) and the policy head learned
  nothing in 9,000 games: all strength came from the value head plus search. Gumbel spends the root's simulations by
  sequential halving and trains the policy on softmax(logits + sigma(completed Q)), i.e. on the children's *values*.
  Measured before switching: such a target has expected centipawn loss 162 against 193 for the policy as it was,
  and the Gumbel search scored 49% where PUCT scored 44% with the same network. Subtracting a visit from every
  move instead (`--policy_target_offset`, KataGo's pruning) was tried first and made things worse: at this budget,
  *which* moves get a second visit is mostly noise.

* **Moves are sampled for the whole game** (`--late_temperature 0.5` after ply 30) instead of
  playing the most visited move. With 32 simulations most visit counts tie, argmax kept choosing
  the same low-index moves, and decisive games fell 46, 31, 8 of 128 over three iterations as
  self-play collapsed into threefold repetitions. With sampling they stay near 44 of 64.
* **Random openings** (`--opening_plies 6`, KataGo's trick). Each game's first 0-6 plies are sampled from the raw
  policy at temperature 2, unsearched and unrecorded; the searched game that follows is trained on as usual. Gumbel
  self-play explores only through its root noise, and with a peaked prior that repeats itself: over 127 recorded games
  of iterations 868-911, 1.d4 was answered by ...d5 in 43 of 44, 1.e4 was played four times, and the games diverged
  only after move 3. Switched on at iteration 921 of `fly6_noadj`; every earlier run played without it.
* **`--sample_reuse 1`.** With 35M parameters and tens of thousands of positions, training four
  times on each position memorised the MCTS targets. The log carries `val_policy_loss` and
  `val_value_loss` on held-out positions so that this is visible.
* **Output activity is standardised with running statistics in both modes** (`RunningNorm`), not
  BatchNorm, whose lagging averages made self-play see a far worse network than training did.
  `scripts/probe_generalisation.py runs/<name>` separates that failure from memorisation.
* **Games that reach `--max_plies` are adjudicated by material** (lead ≥ 5 points wins). Untrained
  networks rarely deliver mate, and without this the value head has little to learn.
  `--adjudicate_material 0` turns it off.

## Layout

```
chessfly/connectome.py   MaleCNS download + parsing, shuffled control
chessfly/brain.py        sparse recurrent core with custom backward
chessfly/model.py        encoder / connectome / heads
chessfly/encoding.py     board features, move indexing
chessfly/mcts.py         PUCT and Gumbel AlphaZero search, batched across games; lazy expansion
chessfly/selfplay.py     game pools, training samples
chessfly/workers.py      search worker processes
chessfly/arena.py        matches against the ladder -> Elo
chessfly/ladder.py       ladder opponents, rating maths
chessfly/suite.py        centipawn loss on the fixed suite
chessfly/dashboard.py    live dashboard and /play (+ dashboard.html, play.html, pieces/)
chessfly/uci.py          a checkpoint as a UCI engine
eval/                    ladder.json, suite.json: the yardsticks (keep them fixed across runs)
chessfly/train.py        the loop
scripts/                 benchmarks, self-play profiler; initialisation, learnability and generalisation probes
```

## Credits

Connectome: Berg et al., *Sexual dimorphism in the complete connectome of the Drosophila male
central nervous system*, Cell 2026 (data CC-BY 4.0, https://male-cns.janelia.org).
The loader and the rate-RNN formulation follow [nfly](https://github.com/zhengxuyu/nfly) (MIT).
