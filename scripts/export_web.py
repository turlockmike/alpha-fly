"""Export a checkpoint for the browser engine in docs/ (GitHub Pages): the effective synapse weights in CSR order,
the neuron constants, the encoder columns the network reads, the heads and the read-out normaliser, as little-endian
binaries in chunks of <= 24 MB (GitHub refuses files over 100 MB and warns over 50), plus a manifest and a few test
positions with the float32 logits and values the JavaScript port must reproduce.

    uv run python scripts/export_web.py runs/fly6_noadj/snapshots/goal_iter_001237.pt docs/model
"""
import json
import sys
from pathlib import Path

import chess
import numpy as np
import torch

from chessfly.encoding import N_BASE, encode, legal_move_indices
from chessfly.model import load_net

CHUNK = 24 * 2**20
ckpt, out = Path(sys.argv[1]), Path(sys.argv[2])
(out / "weights").mkdir(parents=True, exist_ok=True)
net = load_net(ckpt, device="cuda" if torch.cuda.is_available() else "cpu")
net.infer_dtype = torch.float32
b, cfg = net.brain, net.cfg
manifest = {"neurons": b.n, "edges": int(b.pre.numel()), "steps": cfg.steps, "h_max": b.h_max, "input_gain": cfg.input_gain,
            "n_base": N_BASE, "norm_eps": net.norm.eps, "norm_clip": net.norm.clip, "checkpoint": ckpt.name, "tensors": {}}


def put(name, arr: np.ndarray, kind: str):
    raw = arr.tobytes()
    files = []
    for i in range(0, max(len(raw), 1), CHUNK):
        f = f"weights/{name}.{len(files)}.bin"
        (out / f).write_bytes(raw[i:i + CHUNK])
        files.append(f)
    manifest["tensors"][name] = {"kind": kind, "shape": list(arr.shape), "files": files, "bytes": len(raw)}


with torch.no_grad():
    w = b.edge_weights().float().cpu().numpy()
    put("values", w.astype("<f2"), "f16")                                              # CSR values, (post, pre) order
    put("crow", b.crow.cpu().numpy().astype("<u4"), "u32")
    col = b.pre.cpu().numpy().astype("<u4")
    put("col", (col[:, None] >> np.array([0, 8, 16], dtype=np.uint32)).astype(np.uint8).reshape(-1), "u24")   # 3 bytes each
    put("bias", b.bias.float().cpu().numpy().astype("<f4"), "f32")
    put("alpha", torch.sigmoid(b.alpha_logit).float().cpu().numpy().astype("<f4"), "f32")
    put("in_idx", net.in_idx.cpu().numpy().astype("<u4"), "u32")
    put("out_idx", net.out_idx.cpu().numpy().astype("<u4"), "u32")
    put("encoder_wT", net.encoder.weight[:, :N_BASE].T.contiguous().float().cpu().numpy().astype("<f2"), "f16")   # (838, 17937)
    put("encoder_b", net.encoder.bias.float().cpu().numpy().astype("<f4"), "f32")
    put("norm_mean", net.norm.mean.float().cpu().numpy().astype("<f4"), "f32")
    put("norm_std", (net.norm.var + net.norm.eps).sqrt().float().cpu().numpy().astype("<f4"), "f32")
    put("policy_w", net.policy.weight.float().cpu().numpy().astype("<f2"), "f16")     # (4168, 2333)
    put("policy_b", net.policy.bias.float().cpu().numpy().astype("<f4"), "f32")
    put("from_w", net.policy_from.weight.float().cpu().numpy().astype("<f4"), "f32")
    put("from_b", net.policy_from.bias.float().cpu().numpy().astype("<f4"), "f32")
    put("to_w", net.policy_to.weight.float().cpu().numpy().astype("<f4"), "f32")
    put("to_b", net.policy_to.bias.float().cpu().numpy().astype("<f4"), "f32")
    put("value_w", net.value.weight.float().cpu().numpy().astype("<f4"), "f32")
    put("value_b", net.value.bias.float().cpu().numpy().astype("<f4"), "f32")

    # test vectors: float32 logits of the legal moves and the value, from the reference implementation
    fens = [chess.STARTING_FEN,
            "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
            "rnbqkb1r/pp2pppp/3p1n2/2p5/3PP3/2N5/PPP2PPP/R1BQKBNR w KQkq c6 0 4",
            "r3k2r/ppp2ppp/2n1bn2/3qp3/8/2NP1N2/PPP1BPPP/R2QK2R b KQkq - 4 8",
            "8/5pk1/6p1/8/3K4/8/8/4R3 w - - 0 60",
            "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 40",
            "r1b1k2r/ppppqppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 3 6",
            "4k3/8/8/8/8/8/3P4/4K3 b - - 0 1",
            "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2",
            "r2q1rk1/pp1nbppp/2p1pn2/3p4/2PP4/1PN1PN2/P4PPP/R2QKB1R w KQ - 0 9"]
    tests = []
    for fen in fens:
        board = chess.Board(fen)
        moves, idx = legal_move_indices(board)
        x = torch.from_numpy(encode(board)).to(next(net.parameters()).device)[None]
        logits, wdl = net(x)
        tests.append({"fen": fen, "moves": [m.uci() for m in moves], "idx": idx.tolist(),
                      "logits": [round(float(v), 4) for v in logits[0, idx].cpu()],
                      "value": round(float(net.expected_value(wdl)[0]), 5), "features": encode(board)[:N_BASE].nonzero()[0].tolist()})
    manifest["tests"] = tests
(out / "manifest.json").write_text(json.dumps(manifest))
total = sum(t["bytes"] for t in manifest["tensors"].values())
print(f"exported {len(manifest['tensors'])} tensors, {total / 2**20:.1f} MB, {sum(len(t['files']) for t in manifest['tensors'].values())} files")
