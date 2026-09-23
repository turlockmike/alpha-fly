"""Tree search, batched across games: every simulation round descends one leaf per game and
evaluates all leaves in a single network call, so the GPU batch is the number of parallel games.

Two searches share the tree.  PUCT (AlphaZero / lc0) learns the policy from visit counts, which
needs simulations >> legal moves: at 32 simulations over ~25 moves the target is nearly flat and
the policy head learned nothing in the first 9,000 games.  `--gumbel` is Gumbel AlphaZero
(Danihelka et al., ICLR 2022, "Policy improvement by planning with Gumbel"; schedule as in
DeepMind's mctx), made for small budgets: the root spends its simulations by sequential halving
over the Gumbel-top-k moves, and the policy target is  softmax(logits + sigma(completed Q)),
built from the children's *values* rather than from how often they happened to be visited."""

from __future__ import annotations

import dataclasses
import math

import chess
import numpy as np

from .encoding import N_FEATURES, encode, legal_move_indices


@dataclasses.dataclass
class MCTSConfig:
    sims: int = 64                 # (fly6 trains at 32.  64 was tried for 90 minutes from iteration 189: 40.8% against 43.5% over
                                   # 600 games, no gain at half the games per hour - runs/ab4.  The plateau had another cause.)
    c_puct: float = 1.2            # lc0's self-play settings: constant cpuct, unvisited children valued at the
    fpu_reduction: float = 0.0     # parent's Q (no reduction), Dirichlet(0.3) mixed in at 25%
    dirichlet_alpha: float = 0.3
    dirichlet_eps: float = 0.25
    gumbel: bool = False           # Gumbel AlphaZero search and targets instead of PUCT
    max_considered: int = 16       # root moves entering sequential halving
    c_visit: float = 50.0          # sigma(q) = (c_visit + max visits) * c_scale * (q + 1) / 2   (paper defaults; with
    lazy: bool = False             # see Search.descend: legal moves are generated on a leaf's second visit, not its first
    c_scale: float = 1.0           # this value head the target's expected centipawn loss is 162 against 193 for the
                                   # raw policy, and keeps falling to 157 at c_scale 2: scripts/probe_gumbel_target.py)


class Node:
    __slots__ = ("moves", "idx", "P", "N", "W", "children", "terminal", "logits", "v", "raw")

    def __init__(self):
        self.moves = None        # list[chess.Move]; None until expanded
        self.idx = None          # policy indices of `moves`
        self.P = self.N = self.W = None
        self.children = None
        self.terminal = None     # value for the side to move if the game is over here
        self.logits = None       # legal-move logits (max 0) and the network's value here, for the side to move
        self.v = 0.0
        self.raw = None          # lazy search: the network's whole policy vector, kept until the node is expanded

    def expand(self, moves, idx, priors, logits=None, v: float = 0.0):
        self.moves, self.idx, self.P, self.logits, self.v = moves, idx, priors, logits, v
        self.N = np.zeros(len(moves), dtype=np.float32)
        self.W = np.zeros(len(moves), dtype=np.float32)
        self.children = [None] * len(moves)

    def select(self, cfg: MCTSConfig) -> int:
        total = self.N.sum()
        visited = self.N > 0
        parent_q = self.W.sum() / total if total else 0.0
        q = np.where(visited, self.W / np.maximum(self.N, 1), parent_q - cfg.fpu_reduction)
        return int(np.argmax(q + cfg.c_puct * self.P * math.sqrt(total + 1) / (1 + self.N)))

    # ---- Gumbel AlphaZero ----------------------------------------------------------------
    def completed_q(self) -> np.ndarray:
        """Q of every move for the side to move: the search value where visited, and elsewhere v_mix, the
        paper's estimate of this node's value from the network's v and the policy-weighted visited Qs."""
        visited = self.N > 0
        if not visited.any():
            return np.full(len(self.N), self.v, dtype=np.float32)
        q = self.W[visited] / self.N[visited]
        total = self.N.sum()
        v_mix = (self.v + total * float((self.P[visited] * q).sum() / self.P[visited].sum())) / (1 + total)
        out = np.full(len(self.N), v_mix, dtype=np.float32)
        out[visited] = q
        return out

    def sigma_q(self, cfg: MCTSConfig) -> np.ndarray:
        return (cfg.c_visit + self.N.max()) * cfg.c_scale * (self.completed_q() + 1) / 2      # q rescaled to [0, 1]

    def improved_policy(self, cfg: MCTSConfig) -> np.ndarray:
        z = self.logits + self.sigma_q(cfg)
        p = np.exp(z - z.max())
        return p / p.sum()

    def select_gumbel(self, cfg: MCTSConfig) -> int:
        """Below the root: deterministic, so that visit counts track the improved policy."""
        return int(np.argmax(self.improved_policy(cfg) - self.N / (1 + self.N.sum())))


def considered_visits(max_considered: int, sims: int) -> tuple[int, ...]:
    """mctx's sequential-halving schedule: entry t is the visit count a root move must have to be
    eligible at simulation t.  16 moves, 32 simulations -> 16 x [0], 8 x [1], 4 x [2], 4 x [3]."""
    if max_considered <= 1:
        return tuple(range(sims))
    log2max = int(math.ceil(math.log2(max_considered)))
    seq, visits, considered = [], [0] * max_considered, max_considered
    while len(seq) < sims:
        for _ in range(max(1, int(sims / (log2max * considered)))):
            seq.extend(visits[:considered])
            for i in range(considered):
                visits[i] += 1
        considered = max(2, considered // 2)
    return tuple(seq[:sims])


def terminal_value(board: chess.Board, has_moves: bool) -> float | None:
    """Value for the side to move if the game has ended, else None."""
    if not has_moves:
        return -1.0 if board.is_check() else 0.0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0.0
    return None


class Search:
    """One game's tree.  `descend` walks to a leaf and returns its features (or None if the
    leaf was terminal and already backed up); `finish` expands the leaf with the network output."""

    def __init__(self, board: chess.Board, cfg: MCTSConfig, rng: np.random.Generator, explore: bool = False):
        """explore: sample Gumbel noise at the root (self-play).  Without it the Gumbel search is deterministic (matches)."""
        self.board, self.cfg, self.rng, self.explore = board, cfg, rng, explore
        self.root = Node()
        self._g = self._root_visits = self._seq = None      # Gumbel root state, rebuilt for every move
        self.noise_pending = False
        self.move_sims = self.move_considered = None      # per-move overrides of cfg.sims / cfg.max_considered (playout cap)
        self._t = 0
        self._path: list[tuple[Node, int]] = []
        self._leaf: Node | None = None

    def _expand_from_cache(self, node: Node, board: chess.Board) -> None:
        moves, idx = legal_move_indices(board)
        z = node.raw[idx].astype(np.float32)
        z -= z.max()
        p = np.exp(z)
        node.expand(moves, idx, p / p.sum(), z, node.v)
        node.raw = None
        if node is self.root and self.noise_pending:
            self.noise_pending = False
            self.add_root_noise()

    def expand_root(self) -> None:
        """Lazy search: generate the root's moves from its cached policy vector (it has been evaluated once)."""
        if self.root.moves is None and self.root.raw is not None:
            self._expand_from_cache(self.root, self.board)

    def descend(self) -> np.ndarray | None:
        """Walk to a leaf.  Eager search generates a leaf's legal moves at once, to ask the network for their
        logits.  That is half of the search's CPU time, and at tens of simulations ~70% of leaves are never
        visited again.  Lazy search asks for the leaf's value and whole policy vector, and generates the
        moves only when a later simulation returns to the node - the same tree, the same numbers."""
        node, board, path = self.root, self.board, []
        while node.terminal is None:
            if node.moves is None:
                if node.raw is None:
                    break                                   # a new leaf
                self._expand_from_cache(node, board)        # seen once before: expand it now and keep walking
            if not self.cfg.gumbel:
                a = node.select(self.cfg)
            elif node is self.root:
                a = self._gumbel_root_action()
            else:
                a = node.select_gumbel(self.cfg)
            path.append((node, a))
            board.push(node.moves[a])
            if node.children[a] is None:
                node.children[a] = Node()
            node = node.children[a]
        x = None
        if node.terminal is None and self.cfg.lazy:
            node.terminal = terminal_value(board, any(board.legal_moves))      # any(): stops at the first legal move
            if node.terminal is None:
                x = encode(board)
        elif node.terminal is None:
            moves, idx = legal_move_indices(board)
            node.terminal = terminal_value(board, bool(moves))
            if node.terminal is None:
                node.moves, node.idx = moves, idx
                x = encode(board)
        for _ in path:
            board.pop()
        if x is None:
            self._backup(path, node.terminal)
            return None
        self._path, self._leaf = path, node
        return x

    def finish_lazy(self, logits: np.ndarray, value: float) -> None:
        self._leaf.raw, self._leaf.v = logits.copy(), value      # a row of a shared buffer: keep our own copy
        self._backup(self._path, value)

    def finish(self, legal_logits: np.ndarray, value: float) -> None:
        leaf = self._leaf
        z = (legal_logits - legal_logits.max()).astype(np.float32)
        p = np.exp(z)
        leaf.expand(leaf.moves, leaf.idx, p / p.sum(), z, value)
        self._backup(self._path, value)

    @staticmethod
    def _backup(path, v: float) -> None:
        for node, a in reversed(path):
            v = -v
            node.N[a] += 1
            node.W[a] += v

    def add_root_noise(self) -> None:
        r = self.root
        noise = self.rng.dirichlet([self.cfg.dirichlet_alpha] * len(r.moves))
        r.P = (1 - self.cfg.dirichlet_eps) * r.P + self.cfg.dirichlet_eps * noise

    def visit_policy(self) -> np.ndarray:
        return self.root.N / self.root.N.sum()

    def _gumbel_root_action(self) -> int:
        r = self.root
        if self._seq is None:                         # first simulation of this move (the root may carry a reused subtree)
            n = len(r.moves)
            self._g = self.rng.gumbel(size=n).astype(np.float32) if self.explore else np.zeros(n, np.float32)
            self._root_visits = np.zeros(n, dtype=np.int64)       # visits of *this* move's search: the schedule counts these
            self._seq, self._t = considered_visits(min(self.move_considered or self.cfg.max_considered, n),
                                                   (self.move_sims or self.cfg.sims) + 1), 0
        want = self._seq[min(self._t, len(self._seq) - 1)]
        self._t += 1
        score = self._g + r.logits + r.sigma_q(self.cfg)
        eligible = self._root_visits == want
        if not eligible.any():                        # fewer legal moves than the schedule assumes
            eligible = self._root_visits == self._root_visits.min()
        a = int(np.argmax(np.where(eligible, score, -np.inf)))
        self._root_visits[a] += 1
        return a

    def policy_target(self) -> np.ndarray:
        return self.root.improved_policy(self.cfg) if self.cfg.gumbel else self.visit_policy()

    def best_action(self) -> int:
        """The move to play when no exploration beyond the search's own is wanted: Gumbel's winner of the
        sequential halving (a sample from the improved policy if `explore`), or PUCT's most visited move."""
        r = self.root
        if not self.cfg.gumbel or self._root_visits is None:
            return int((r.N + self.rng.random(len(r.N)) * 0.5).argmax())                      # ties at random
        score = self._g + r.logits + r.sigma_q(self.cfg)
        return int(np.argmax(np.where(self._root_visits == self._root_visits.max(), score, -np.inf)))

    def root_value(self) -> float:
        return float(self.root.W.sum() / max(self.root.N.sum(), 1))

    def advance(self, a: int) -> None:
        """Play root move `a` on the board and keep its subtree."""
        self.board.push(self.root.moves[a])
        self.root = self.root.children[a] or Node()
        self._g = self._root_visits = self._seq = None

    def advance_move(self, move: chess.Move) -> None:
        if self.root.moves is None:
            self.board.push(move)
            self.root = Node()
            self._g = self._root_visits = self._seq = None
        else:
            self.advance(self.root.moves.index(move))


def descend_all(searches: list[Search]):
    """One leaf per game.  Returns the searches awaiting an evaluation and the request
    (x (B, F), rows (M,), cols (M,)): logits[rows, cols] are the legal moves of all leaves, concatenated."""
    pending, xs = [], []
    for s in searches:
        x = s.descend()
        if x is not None:
            pending.append(s)
            xs.append(x)
    lazy = bool(searches) and searches[0].cfg.lazy
    if not pending:
        return pending, (np.zeros((0, N_FEATURES), np.float16), None, None) if lazy else \
            (np.zeros((0, N_FEATURES), np.float16), np.zeros(0, np.int64), np.zeros(0, np.int64))
    if lazy:
        return pending, (np.stack(xs).astype(np.float16), None, None)       # no legal-move gather: the whole policy comes back
    counts = [len(s._leaf.idx) for s in pending]
    return pending, (np.stack(xs).astype(np.float16), np.repeat(np.arange(len(pending)), counts),      # half the bytes on the pipe
                     np.concatenate([s._leaf.idx for s in pending]))


def finish_all(pending: list[Search], legal_logits: np.ndarray, values: np.ndarray) -> None:
    """legal_logits: eager search - the legal moves' logits of all leaves, concatenated; lazy search - (B, N_MOVES)."""
    if pending and pending[0].cfg.lazy:
        for i, s in enumerate(pending):
            s.finish_lazy(legal_logits[i], float(values[i]))
        return
    a = 0
    for s, v in zip(pending, values):
        b = a + len(s._leaf.idx)
        s.finish(legal_logits[a:b], float(v))
        a = b


def run_simulations(searches: list[Search], evaluate, sims: int, noise: bool) -> None:
    """`evaluate(x, rows, cols) -> (logits[rows, cols] (M,), values (B,))` as numpy arrays."""
    def round_():
        pending, request = descend_all(searches)
        if pending:
            finish_all(pending, *evaluate(*request))

    if any(s.root.moves is None for s in searches):
        round_()                                  # expand fresh roots
    if noise:
        for s in searches:
            if not s.cfg.gumbel and s.root.moves is not None and s.root.terminal is None:
                s.add_root_noise()                # Gumbel explores through its own noise (Search(explore=True))
    for _ in range(sims):
        round_()
