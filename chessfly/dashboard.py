"""Live dashboard over runs/<name>/log.jsonl.

    uv run python -m chessfly.dashboard            # then open http://127.0.0.1:8765   (/play: play the newest network)

Standard library only, read-only, bound to localhost.  The page polls /api/log with a byte
offset, so a long log is sent once and then only its new lines."""

from __future__ import annotations

import argparse
import functools
import json
import re
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import chess.pgn

ROOT = Path(__file__).parent.parent
RUNS = ROOT / "runs"
PAGE = Path(__file__).with_name("dashboard.html")
PLAY_PAGE = Path(__file__).with_name("play.html")
PIECES = Path(__file__).with_name("pieces")


def run_dir(name: str) -> Path | None:
    """Only a plain directory name directly under runs/ is accepted."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name or "") or name in (".", ".."):
        return None
    d = RUNS / name
    return d if d.is_dir() else None


@functools.lru_cache(maxsize=64)
def _games_of(path: str, mtime: float) -> list[dict]:
    """Every game of a PGN file, replayed here so the page needs no chess logic: one FEN and one
    last-move per ply.  Cached on (path, mtime): finished files never change."""
    out = []
    with open(path, encoding="utf-8") as f:
        while (game := chess.pgn.read_game(f)) is not None:
            board, fens, sans, moves = game.board(), [game.board().board_fen()], [], [None]
            for move in game.mainline_moves():
                sans.append(board.san(move))
                board.push(move)
                fens.append(board.board_fen())
                moves.append([move.from_square, move.to_square])
            h = game.headers
            out.append({"white": h.get("White", "?"), "black": h.get("Black", "?"), "result": h.get("Result", "*"),
                        "termination": h.get("Termination", ""), "round": h.get("Round", ""), "file": Path(path).name,
                        "fens": fens, "sans": sans, "moves": moves})
    return out


def recent_games(d: Path, kind: str, n: int) -> list[dict]:
    """Newest first.  kind: "selfplay" (the sampled self-play games) or "eval" (matches against the ladder)."""
    folder = d / ("selfplay" if kind == "selfplay" else "games")
    files = sorted(folder.glob("*.pgn"), key=lambda p: p.stat().st_mtime, reverse=True) if folder.is_dir() else []
    games = []
    for p in files:
        m = re.search(r"iter_(\d+)", p.name)
        for k, g in enumerate(reversed(_games_of(str(p), p.stat().st_mtime))):
            games.append(g | {"iter": int(m.group(1)) if m else None, "id": f"{p.name}#{k}"})
            if len(games) >= n:
                return games
    return games


class Engine:
    """The newest checkpoint of a run, on the GPU, for the /play page.  torch is imported on first use, so the
    dashboard stays light until somebody plays; the checkpoint is reloaded when training has saved a newer one."""

    def __init__(self):
        self.lock, self.key, self.net, self.info = threading.Lock(), None, None, {}

    def load(self, d: Path) -> dict:
        import torch
        from .model import Evaluator, load_net
        path = d / "last.pt"
        key = (str(path), path.stat().st_mtime)
        if key != self.key:
            self.net = None                                   # free the old copy first
            net = load_net(path, data_dir=ROOT / "data", device="cuda" if torch.cuda.is_available() else "cpu")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.net, self.key = Evaluator(net, next(net.parameters()).device), key
            self.info = {"run": d.name, "iter": ck.get("iter"), "elo": ck.get("elo"), "steps": ck["config"].get("steps"),
                         "saved": time.strftime("%H:%M", time.localtime(path.stat().st_mtime))}
        return self.info

    def think(self, d: Path, board: "chess.Board", sims: int) -> dict:
        import numpy as np
        from .mcts import MCTSConfig, Search, run_simulations
        with self.lock:
            self.load(d)
            t = time.time()
            search = Search(board.copy(), MCTSConfig(sims=sims, gumbel=True, max_considered=16), np.random.default_rng())
            run_simulations([search], self.net, sims, noise=False)
            root, a = search.root, search.best_action()
            q = root.W / np.maximum(root.N, 1)
            order = np.argsort(-root.N)[:5]
            return {"move": root.moves[a].uci(), "value": float(q[a]), "sims": int(root.N.sum()), "ms": round((time.time() - t) * 1000),
                    "top": [{"san": board.san(root.moves[i]), "visits": int(root.N[i]), "q": round(float(q[i]), 3),
                             "prior": round(float(root.P[i]), 3)} for i in order if root.N[i] > 0]}


ENGINE = Engine()


def game_status(board: "chess.Board") -> tuple[str, str | None]:
    if board.is_checkmate():
        return "checkmate", "0-1" if board.turn == chess.WHITE else "1-0"
    if board.is_stalemate():
        return "stalemate", "1/2-1/2"
    if board.is_insufficient_material():
        return "insufficient material", "1/2-1/2"
    if board.is_fifty_moves() or board.halfmove_clock >= 100:
        return "fifty-move rule", "1/2-1/2"
    if board.is_repetition(3):
        return "threefold repetition", "1/2-1/2"
    return "playing", None


def play(d: Path, moves: list[str], human: str, sims: int) -> dict:
    """Stateless: the page sends the whole game; the reply holds the engine's answer if it is the engine's turn."""
    board = chess.Board()
    for u in moves:
        m = chess.Move.from_uci(u)
        if m not in board.legal_moves:
            raise ValueError(f"illegal move {u}")
        board.push(m)
    engine = None
    if game_status(board)[0] == "playing" and board.turn != (chess.WHITE if human == "white" else chess.BLACK):
        engine = ENGINE.think(d, board, max(1, min(int(sims), 3200)))
        engine["san"] = board.san(chess.Move.from_uci(engine["move"]))
        board.push(chess.Move.from_uci(engine["move"]))
    else:
        with ENGINE.lock:
            ENGINE.load(d)
    status, result = game_status(board)
    replay, sans = chess.Board(), []
    for m in board.move_stack:
        sans.append(replay.san(m)); replay.push(m)
    last = board.move_stack[-1] if board.move_stack else None
    return {"fen": board.board_fen(), "turn": "white" if board.turn == chess.WHITE else "black", "moves": [m.uci() for m in board.move_stack],
            "sans": sans, "legal": [m.uci() for m in board.legal_moves] if status == "playing" else [], "check": board.is_check(),
            "last": [last.from_square, last.to_square] if last else None, "status": status, "result": result,
            "engine": engine, "checkpoint": ENGINE.info}


class Handler(BaseHTTPRequestHandler):
    def _send(self, body: bytes, kind: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, status: int = 200) -> None:
        self._send(json.dumps(obj).encode(), "application/json", status)

    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        if url.path == "/":
            return self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/play":
            return self._send(PLAY_PAGE.read_bytes(), "text/html; charset=utf-8")
        if m := re.fullmatch(r"/pieces/([wb][kqrbnp])\.svg", url.path):
            self.send_response(200)
            body = (PIECES / f"{m.group(1)}.svg").read_bytes()
            self.send_header("Content-Type", "image/svg+xml")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400")
            self.end_headers()
            return self.wfile.write(body)
        if url.path == "/api/runs":
            runs = sorted((d for d in RUNS.iterdir() if (d / "log.jsonl").exists()),
                          key=lambda d: (d / "log.jsonl").stat().st_mtime, reverse=True) if RUNS.exists() else []
            return self._json([d.name for d in runs])
        if url.path == "/api/play/runs":                          # runs that have a network to play, newest first
            runs = sorted((d for d in RUNS.iterdir() if (d / "last.pt").exists()),
                          key=lambda d: (d / "last.pt").stat().st_mtime, reverse=True) if RUNS.exists() else []
            return self._json([d.name for d in runs])
        if url.path == "/api/meta":
            ladder = ROOT / "eval" / "ladder.json"
            suite = ROOT / "eval" / "suite.json"
            return self._json({
                "ladder": json.loads(ladder.read_text())["ratings"] if ladder.exists() else {},
                # the baselines sit at the front of the file; do not parse 450 KB for two numbers
                "baselines": json.loads(re.search(r'"baselines": (\{.*?\})', suite.read_text()[:400]).group(1)) if suite.exists() else {},
            })
        if url.path == "/api/games":
            d = run_dir(q.get("run", ""))
            if d is None:
                return self._json({"error": "unknown run"}, 404)
            return self._json(recent_games(d, q.get("kind", "selfplay"), min(int(q.get("n", 10)), 50)))
        if url.path == "/api/log":
            d = run_dir(q.get("run", ""))
            if d is None:
                return self._json({"error": "unknown run"}, 404)
            offset = max(int(q.get("offset", 0)), 0)
            with open(d / "log.jsonl", "rb") as f:
                f.seek(0, 2)
                if offset > f.tell():
                    offset = 0                               # the log was replaced by a shorter one
                f.seek(offset)
                data = f.read()
            data = data[:data.rfind(b"\n") + 1]              # never hand out half a line
            rows = []
            for line in data.splitlines():
                try:
                    rows.append(json.loads(line, parse_constant=lambda _: None))     # NaN is not JSON to a browser
                except json.JSONDecodeError:
                    pass
            return self._json({"rows": rows, "offset": offset + len(data), "paused_file": (d / "PAUSE").exists()})
        self._json({"error": "not found"}, 404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/play":
            return self._json({"error": "not found"}, 404)
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
            d = run_dir(str(body.get("run", "")))
            if d is None or not (d / "last.pt").exists():
                return self._json({"error": "this run has no saved network yet"}, 404)
            moves = [str(m) for m in body.get("moves", [])][:1000]
            return self._json(play(d, moves, "black" if body.get("human") == "black" else "white", int(body.get("sims", 100))))
        except Exception as e:                                    # a bad request must not take the dashboard down
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"dashboard on {url}  (Ctrl+C to stop; training is not affected)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
