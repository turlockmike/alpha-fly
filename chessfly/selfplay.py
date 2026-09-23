"""Self-play with a fixed pool of concurrent games; finished games are replaced immediately,
so the pool (and the GPU batch) stays full and long games are not under-sampled.

This module must not import torch: it runs inside the search worker processes."""

from __future__ import annotations

import dataclasses

import chess
import numpy as np

from .encoding import encode
from .mcts import MCTSConfig, Search, descend_all, finish_all, terminal_value

PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


@dataclasses.dataclass
class SelfPlayConfig:
    parallel_games: int = 1120       # cuSPARSE's SpMM costs the same for any batch of 513-640 columns (it steps every 128,
    workers: int = 8                 # scripts/bench_evaluator.py).  Each worker alternates two pools, so a round carries half the
                                     # games: 8 x 70 = 560 leaves, 624 with the arena's 64 - just inside the 640 step.  The old
                                     # 1024 games paid for 640 columns and used 512-576.  (workers = 0: search in the trainer.)
    prioritise_trainer: int = 1      # trainer process above normal priority, search workers below (see workers.set_priority)
    max_plies: int = 200
    temperature_plies: int = 30      # sample moves in proportion to visits for this long ...
    temp_visit_offset: float = -1.0  # lc0's --temp-visit-offset: moves are sampled from max(0, visits + offset), so a
                                     # move that only got its one exploratory visit is never played
    late_temperature: float = 0.5    # ... then in proportion to visits^(1/t).  AlphaZero plays the most visited move
                                     # (t -> 0), but at tens of simulations visit counts tie, argmax keeps picking the
                                     # same low-index moves, and self-play collapses into threefold repetitions.
    full_search_prob: float = 1.0    # KataGo's playout cap randomisation: each move gets the full search with this probability and
    fast_sims: int = 8               # is recorded for training; otherwise `fast_sims` simulations just keep the game moving (their
                                     # search values still feed the TD(lambda) targets).  0.25 / 8 = ~2.2x more games per GPU-hour:
                                     # independent game results are what the value head is short of.  1.0 = every move in full.
                                     # TRIED (A/B fork from iteration 67 of fly6, one hour of wall clock per arm): 0.25 / 8 played
                                     # 2.7x more games and finished behind - 39.6% against 43.4% over 600 games versus the same two
                                     # ladder rungs (about -28 Elo, 1.3 standard errors; runs/ab/head_to_head.txt).  Self-play became
                                     # sloppier and more drawn (7% -> 23% draws) and only a quarter of the positions were recorded.
                                     # Left at 1.0; an hour may be too short to show a value-learning gain, but it bought nothing here.
    td_lambda: float = 0.9           # Sample.q = (1-l) q_t - l G_{t+1}, ending in the game result: an average of the *future*
                                     # search values (KataGo's TD(lambda) target).  At this strength the final result says little
                                     # about a position 60 plies earlier (held-out value loss sits at the entropy of results),
                                     # while the search value a few plies on already knows about the blunder.  1.0 = result only.
    record_fraction: float = 0.01    # share of self-play games kept as PGN in runs/<name>/selfplay/ (~1 KB each)
    opening_plies: int = 6           # KataGo-style random openings: each game starts with 0..N plies (uniform) played straight
    opening_temperature: float = 2.0 # from the raw policy at this temperature, unsearched and not recorded as samples.  Gumbel
                                     # self-play explores only through its root noise, and with a peaked prior that is not much:
                                     # over iterations 868-911 of fly6_noadj (127 recorded games) 1.d4 was answered by ...d5 in
                                     # 43 of 44 games, 1.e4 was played 4 times in 127, and 9 first moves / 22 two-ply openings
                                     # covered everything - the games only diverged after move 3.  Switched on at iteration 921
                                     # (before: snapshots/before_openings_iter_000921.pt).  0 = off, as every earlier run played.
                                     # RESULT over the next 100 iterations (rows with a full Elo window): rolling Elo 1242 (range
                                     # 1204-1314) against 1209 (1176-1237) over the 70 before, raw-policy CPL 75.3 against 79.0,
                                     # 47% against 39% versus sf_1320 - about +30 Elo, kept.
    adjudicate_material: int = 5     # at max_plies, a material lead of this many points wins (0 = always draw).
                                     # Not in AlphaZero; early nets never mate, and without it every game is a draw.
                                     # IT MUST BE SWITCHED OFF ONCE THE NETWORK MATES.  Left on, it teaches that holding material
                                     # until the clock runs out *is* winning: over fly6's first 190 iterations checkmates fell from
                                     # 49.5% of self-play games to 34.1% while adjudicated wins rose from 34% to 58%, the winner
                                     # sat a queen or more ahead for a median of 95 plies without mating, and playing strength
                                     # against the ladder stopped improving at iteration ~100 - through a playout cap, attack-plane
                                     # inputs, a learning-rate drop and 64-simulation self-play alike (runs/ab .. runs/ab4).


@dataclasses.dataclass
class Sample:
    x: np.ndarray        # (N_FEATURES,) float16
    idx: np.ndarray      # (k,) int16 policy indices of the legal moves
    pi: np.ndarray       # (k,) float16 visit distribution
    q: float = 0.0       # search value of the root for the side to move (lc0's root_q)
    wdl: int = 1         # 0 win / 1 draw / 2 loss for the side to move


@dataclasses.dataclass
class FinishedGame:
    samples: list[Sample]
    result: float        # for white
    plies: int
    adjudicated: bool
    termination: str = ""
    san: str = ""        # movetext, written by the worker that played the game


def material_balance(board: chess.Board) -> int:
    """White minus black."""
    return sum(v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
               for pt, v in PIECE_VALUE.items())


def result_white(board: chess.Board, root_terminal: float | None, cfg: SelfPlayConfig) -> float | None:
    """+1 / 0 / -1 for white if the game is over, else None."""
    if root_terminal is not None:
        return root_terminal if board.turn == chess.WHITE else -root_terminal
    if board.ply() >= cfg.max_plies:
        m = material_balance(board)
        return float(np.sign(m)) if cfg.adjudicate_material and abs(m) >= cfg.adjudicate_material else 0.0
    return None


def termination(board: chess.Board, root_terminal: float | None, z: float) -> str:
    if root_terminal is None:
        return "material adjudication at the move limit" if z else "move limit"
    if board.is_checkmate():
        return "checkmate"
    if not any(board.legal_moves):
        return "stalemate"
    if board.is_insufficient_material():
        return "insufficient material"
    return "fifty-move rule" if board.halfmove_clock >= 100 else "threefold repetition"


def td_targets(samples: list[Sample], z_last: float, lam: float) -> None:
    """Replace each sample's search value q by the lambda-return of the search values that followed, from the
    point of view of its own side to move.  Every ply is a sample, so the point of view alternates.
    z_last: the game result for the side to move in the last sample."""
    g = -z_last                                  # the "return" after the last sample, seen by the other side
    for smp in reversed(samples):
        g = (1 - lam) * smp.q - lam * g
        smp.q = float(g)


class _Game:
    def __init__(self, mcts: MCTSConfig, rng):
        self.search = Search(chess.Board(), mcts, rng, explore=True)
        self.samples: list[tuple[Sample, chess.Color, bool]] = []       # (sample, side to move, recorded for training?)
        self.rounds, self.budget, self.full = 0, mcts.sims, True
        self.opening = 0                  # plies of this game that are played from the raw policy (SelfPlayConfig.opening_plies)


class GamePool:
    """Games driven from outside, one simulation round at a time: `request()` descends one leaf per game,
    `deliver()` takes the network's answer.  A game plays its move once its search budget is spent (the
    same for every game unless the playout cap is randomised); finished games pile up in `finished`."""

    def __init__(self, n_games: int, mcts: MCTSConfig, cfg: SelfPlayConfig, rng: np.random.Generator):
        self.mcts, self.cfg, self.rng = mcts, cfg, rng
        self.games = [self._new_game() for _ in range(n_games)]
        self.finished: list[FinishedGame] = []
        self.moves_played = 0                     # throughput that does not depend on when games happen to finish
        self._pending = []

    def _new_game(self) -> _Game:
        g = _Game(self.mcts, self.rng)
        if self.cfg.opening_plies > 0:
            g.opening = int(self.rng.integers(0, self.cfg.opening_plies + 1))
        self._budget(g)
        return g

    def _in_opening(self, g: _Game) -> bool:
        return g.search.board.ply() < g.opening

    def _budget(self, g: _Game) -> None:
        """Decide how hard the next move is searched."""
        g.rounds = 0
        if self._in_opening(g):                    # one round evaluates the root; its policy is then sampled, no search
            g.full, g.budget = False, 0
            g.search.move_sims = g.search.move_considered = None
            return
        g.full = self.cfg.full_search_prob >= 1 or self.rng.random() < self.cfg.full_search_prob
        g.budget = self.mcts.sims if g.full else self.cfg.fast_sims
        g.search.move_sims = None if g.full else g.budget
        g.search.move_considered = None if g.full else min(self.mcts.max_considered, max(g.budget, 2))

    def request(self):
        self._pending, req = descend_all([g.search for g in self.games])
        return req

    def deliver(self, legal_logits: np.ndarray, values: np.ndarray) -> None:
        finish_all(self._pending, legal_logits, values)
        for i, g in enumerate(self.games):
            g.rounds += 1
            if g.rounds == 1 and not self.mcts.gumbel and g.search.root.terminal is None:      # PUCT explores by root noise
                if g.search.root.moves is not None:
                    g.search.add_root_noise()
                else:
                    g.search.noise_pending = True      # lazy search: the root is expanded by the next round
            if g.rounds > g.budget:                    # one round to evaluate the root, then `budget` simulations
                self._play_move(i, g)

    def _play_move(self, i: int, g: _Game) -> None:
        s, board = g.search, g.search.board
        if s.root.terminal is None and self._in_opening(g):
            s.expand_root()
            z = s.root.logits.astype(np.float64) / self.cfg.opening_temperature
            p = np.exp(z - z.max())
            s.advance(self.rng.choice(len(p), p=p / p.sum()))
            self.moves_played += 1
        elif s.root.terminal is None:
            pi = s.policy_target()
            g.samples.append((Sample(encode(board).astype(np.float16), s.root.idx.astype(np.int16),
                                     pi.astype(np.float16), s.root_value()), board.turn, g.full))
            if self.mcts.gumbel:                    # the halving winner is already a sample from the improved policy
                s.advance(s.best_action())
            else:
                p = np.maximum(s.root.N + self.cfg.temp_visit_offset, 0).astype(np.float64)
                if not p.any():
                    p = pi.astype(np.float64)
                if board.ply() >= self.cfg.temperature_plies:
                    p = p ** (1 / self.cfg.late_temperature)
                s.advance(self.rng.choice(len(p), p=p / p.sum()))
            self.moves_played += 1
            if s.root.moves is None and s.root.terminal is None and s.root.raw is None:     # unexplored child: settle it now
                s.root.terminal = terminal_value(board, any(board.legal_moves))
        z = result_white(board, s.root.terminal, self.cfg)
        if z is None:
            self._budget(g)
            return
        for sample, turn, _ in g.samples:
            sample.wdl = 1 - int(z if turn == chess.WHITE else -z)
        td_targets([smp for smp, _, _ in g.samples], z if g.samples and g.samples[-1][1] == chess.WHITE else -z,
                   self.cfg.td_lambda)                                   # over every ply, recorded or not
        self.finished.append(FinishedGame(
            [smp for smp, _, keep in g.samples if keep], z, board.ply(), s.root.terminal is None and z != 0,
            termination(board, s.root.terminal, z),
            chess.Board().variation_san(board.move_stack) if self.rng.random() < self.cfg.record_fraction else ""))
        self.games[i] = self._new_game()

    def take_finished(self) -> list[FinishedGame]:
        out, self.finished = self.finished, []
        return out

    def take_moves(self) -> int:
        n, self.moves_played = self.moves_played, 0
        return n


class SelfPlayStats(dict):
    def __init__(self):
        super().__init__(games=0, white=0, draw=0, black=0, adjudicated=0, plies=0, moves=0)
        self.records: list[FinishedGame] = []        # games with movetext, waiting to be written out

    def absorb(self, games: list[FinishedGame]) -> list[Sample]:
        out = []
        for g in games:
            self["games"] += 1
            self["plies"] += g.plies
            self["white" if g.result > 0 else "black" if g.result < 0 else "draw"] += 1
            self["adjudicated"] += g.adjudicated
            out.extend(g.samples)
            if g.san:
                g.samples = []
                self.records.append(g)
        return out


class SelfPlay:
    """Search in the calling process."""

    def __init__(self, evaluate, mcts: MCTSConfig, cfg: SelfPlayConfig, seed: int = 0):
        self.evaluate = evaluate
        self.pool = GamePool(cfg.parallel_games, mcts, cfg, np.random.default_rng(seed))
        self.stats = SelfPlayStats()

    def play(self, n_games: int, should_stop=lambda: False) -> list[Sample]:
        """Advance every game until `n_games` more have finished; return their samples."""
        out, target = [], self.stats["games"] + n_games
        while self.stats["games"] < target and not should_stop():
            x, rows, cols = self.pool.request()
            legal, values = self.evaluate(x, rows, cols) if len(x) else (np.zeros(0, np.float32),) * 2
            self.pool.deliver(legal, values)
            out.extend(self.stats.absorb(self.pool.take_finished()))
            self.stats["moves"] += self.pool.take_moves()
        return out

    def close(self) -> None:
        pass


def write_selfplay_pgn(path, games: list[FinishedGame], run: str, iteration: int) -> None:
    """Plain PGN, one file per iteration; the movetext was rendered by the workers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for n, g in enumerate(games, 1):
            result = {1.0: "1-0", -1.0: "0-1"}.get(g.result, "1/2-1/2")
            f.write(f'[Event "chess-fly self-play: {run}"]\n[Round "{iteration}.{n}"]\n[White "fly"]\n[Black "fly"]\n'
                    f'[Result "{result}"]\n[PlyCount "{g.plies}"]\n[Termination "{g.termination}"]\n\n{g.san} {result}\n\n')
