"""Tree search in worker processes, network in the calling process.

Python move generation costs ~100 us per leaf, as much as the network at batch 256, and it is
single-threaded: so the games live in `workers` processes and only leaves travel.  Each worker
keeps two pools in flight and descends one while the other is being evaluated, which hides the
round trip.  This module must not import torch (it would be loaded once per worker)."""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
from multiprocessing import shared_memory
import signal
import threading

import numpy as np

from .encoding import N_MOVES
from .mcts import MCTSConfig
from .selfplay import GamePool, Sample, SelfPlay, SelfPlayConfig, SelfPlayStats


class PolicySlots:
    """Lazy search gets the network's whole policy vector per leaf (8 KB): far too much for Windows' 8 KB pipes at
    ~70 rounds a second, so it travels through shared memory and the pipe carries only (slot, rows, values).
    A worker has two requests in flight, hence two slots, used alternately: a slot is only rewritten after the
    worker has consumed it and asked again."""

    def __init__(self, capacity: int, n_slots: int, names: list[str] | None = None):
        size = max(capacity, 1) * N_MOVES * 2
        self.blocks = [shared_memory.SharedMemory(name=n) for n in names] if names else \
            [shared_memory.SharedMemory(create=True, size=size) for _ in range(n_slots)]
        self.views = [np.ndarray((max(capacity, 1), N_MOVES), dtype=np.float16, buffer=b.buf) for b in self.blocks]
        self.names, self.turn = [b.name for b in self.blocks], 0

    def write(self, logits: np.ndarray) -> tuple[int, int]:
        slot, self.turn = self.turn, (self.turn + 1) % len(self.views)
        self.views[slot][:len(logits)] = logits
        return slot, len(logits)

    def read(self, slot: int, k: int) -> np.ndarray:
        return self.views[slot][:k]

    def close(self) -> None:
        self.views = []
        for b in self.blocks:
            b.close()


def unpack(reply, slots: "PolicySlots | None"):
    """(legal logits, values) over the pipe, or ((slot, rows), values) pointing into shared memory."""
    first, values = reply
    return (slots.read(*first), values) if isinstance(first, tuple) else (first, values)


def set_priority(above_normal: bool) -> None:
    """The trainer launches ~40 CUDA kernels per evaluation.  With every core busy in the search workers, its thread
    was descheduled between launches and the GPU starved (an evaluation took 40 ms inside the loop, 21 ms alone).
    Windows only; elsewhere this does nothing."""
    if os.name == "nt":
        import ctypes
        k = ctypes.windll.kernel32
        k.SetPriorityClass(k.GetCurrentProcess(), 0x00008000 if above_normal else 0x00004000)      # ABOVE_NORMAL / BELOW_NORMAL


def _worker(conn, n_games: int, mcts: MCTSConfig, cfg: SelfPlayConfig, seed: int, slot_names=None) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)      # Ctrl+C reaches every process of the console; only the trainer acts on it
    if cfg.prioritise_trainer:
        set_priority(above_normal=False)
    slots = PolicySlots(n_games, 2, slot_names) if slot_names else None
    rng = np.random.default_rng(seed)
    pools = [GamePool(n_games // 2, mcts, cfg, rng), GamePool(n_games - n_games // 2, mcts, cfg, rng)]
    # Sends go through a thread so that this process is always able to receive.  Windows pipes
    # buffer 8 KB: with both ends in a blocking send of something larger, neither ever reads.
    outbox: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [conn.send(m) for m in iter(outbox.get, None)], daemon=True).start()
    for p in pools:
        outbox.put((p.request(), [], 0))
    i = 0
    while (reply := conn.recv()) is not None:          # replies come back in the order requests went out
        pool = pools[i % 2]
        pool.deliver(*unpack(reply, slots))
        outbox.put((pool.request(), pool.take_finished(), pool.take_moves()))
        i += 1


class ParallelSelfPlay:
    def __init__(self, evaluate, mcts: MCTSConfig, cfg: SelfPlayConfig, seed: int = 0, arena: dict | None = None):
        """arena: {"games", "window", "elo_guess"} starts the continuous evaluation worker (arena.py)."""
        self.evaluate, self.stats = evaluate, SelfPlayStats()
        if cfg.prioritise_trainer:
            set_priority(above_normal=True)
        ctx = mp.get_context("spawn")
        self.arena, self.arena_reports, self.slots = None, [], {}
        if arena:
            from .arena import arena_process
            self.arena, there = ctx.Pipe()
            amcts = arena.get("mcts", mcts)
            if amcts.lazy:
                self.slots[self.arena] = PolicySlots(arena["games"], 1)
            self.arena_proc = ctx.Process(target=arena_process, daemon=True,
                                          args=(there, amcts, arena["games"], arena["window"], arena["elo_guess"], seed + 7,
                                                self.slots[self.arena].names if amcts.lazy else None))
            self.arena_proc.start()
        self.conns, self.procs = [], []
        for i in range(cfg.workers):
            n = cfg.parallel_games // cfg.workers + (i < cfg.parallel_games % cfg.workers)
            here, there = ctx.Pipe()
            if mcts.lazy:
                self.slots[here] = PolicySlots(max(n, 2), 2)
            p = ctx.Process(target=_worker, daemon=True,
                            args=(there, max(n, 2), mcts, cfg, seed * 1000 + i, self.slots[here].names if mcts.lazy else None))
            p.start()
            self.conns.append(here)
            self.procs.append(p)

    def play(self, n_games: int, should_stop=lambda: False) -> list[Sample]:
        """Advance every game until `n_games` more have finished; return their samples.

        Pipelined: round k is launched on the GPU, then round k+1 is received and assembled while it
        computes, and only then are round k's results fetched and sent out.  (Workers keep two pools
        in flight, so the next requests are already in the pipes.)  Unpickling and concatenating used
        to leave the GPU idle for a quarter of every round."""
        submit = getattr(self.evaluate, "submit", None) or (lambda *req: self.evaluate(*req))
        result = getattr(self.evaluate, "result", None) or (lambda handle: handle)
        out, target, inflight = [], self.stats["games"] + n_games, None

        def finish(job) -> None:
            handle, conns, reqs = job
            legal, values = result(handle) if handle is not None else (np.zeros((0, N_MOVES), np.float16), np.zeros(0, np.float32))
            a = b = 0
            for c, (x, rows, _) in zip(conns, reqs):
                if rows is None:                                # lazy: one policy row per leaf, through shared memory
                    c.send((self.slots[c].write(legal[b:b + len(x)]), values[b:b + len(x)]))
                else:
                    c.send((legal[a:a + len(rows)], values[b:b + len(x)]))
                    a += len(rows)
                b += len(x)

        while self.stats["games"] < target and not should_stop():
            msgs = [c.recv() for c in self.conns]           # one request per worker: a lockstep round
            reqs, conns = [m[0] for m in msgs], list(self.conns)
            for _, finished, moves in msgs:
                out.extend(self.stats.absorb(finished))
                self.stats["moves"] += moves
            # the arena is slower than a lockstep round whenever Stockfish is thinking: take its request only if it is there
            if self.arena is not None and self.arena.poll():
                areq, reports = self.arena.recv()
                self.arena_reports.extend(reports)
                reqs.append(areq)
                conns.append(self.arena)
            sizes = [len(x) for x, _, _ in reqs]
            offsets = np.cumsum([0] + sizes[:-1])
            if not sum(sizes):
                batch = None
            elif reqs[0][1] is None:
                batch = (np.concatenate([r[0] for r in reqs]), None, None)
            else:
                batch = (np.concatenate([r[0] for r in reqs]), np.concatenate([r[1] + o for r, o in zip(reqs, offsets)]),
                         np.concatenate([r[2] for r in reqs]))
            if inflight is not None:
                finish(inflight)
            inflight = (submit(*batch) if batch else None, conns, reqs)
        if inflight is not None:                            # nothing may stay on the GPU across a training phase
            finish(inflight)
        return out

    def take_arena_reports(self) -> list[dict]:
        out, self.arena_reports = self.arena_reports, []
        return out

    def close(self) -> None:
        for c in self.conns + ([self.arena] if self.arena is not None else []):
            try:
                c.send(None)
            except OSError:
                pass
        for p in self.procs + ([self.arena_proc] if self.arena is not None else []):
            p.join(timeout=5)
        for s in self.slots.values():
            s.close()
            if p.is_alive():
                p.terminate()


def make_selfplay(evaluate, mcts: MCTSConfig, cfg: SelfPlayConfig, seed: int = 0, arena: dict | None = None):
    if cfg.workers > 0:
        return ParallelSelfPlay(evaluate, mcts, cfg, seed, arena)
    return SelfPlay(evaluate, mcts, cfg, seed)
