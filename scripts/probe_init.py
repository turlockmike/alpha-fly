"""At initialisation, do different chess positions produce different activity at the output neurons?"""
import sys
import chess, numpy as np, torch
from chessfly.connectome import load_malecns
from chessfly.encoding import encode
from chessfly.model import FlyChessNet, ModelConfig

rng = np.random.default_rng(0)
boards = []
while len(boards) < 128:
    b = chess.Board()
    for _ in range(rng.integers(2, 60)):
        if b.is_game_over(): break
        ms = list(b.legal_moves); b.push(ms[rng.integers(len(ms))])
    boards.append(b)
x = torch.from_numpy(np.stack([encode(b) for b in boards])).cuda()
conn = load_malecns("data")
for scale, bias, gain in [(1, 0, 1), (1, 0.1, 1), (2, 0.1, 1), (2, 0.1, 5), (3, 0.2, 5), (1, 0.5, 5)]:
    torch.manual_seed(0)
    net = FlyChessNet(conn, ModelConfig(input_gain=gain)).cuda()
    net.brain.w0 *= scale; net.brain.bias.data.fill_(bias)
    with torch.no_grad():
        net.infer_dtype = torch.float32
        u = torch.zeros(len(x), net.brain.n, device="cuda"); u[:, net.in_idx] = net.encoder(x) * gain
        h = net.brain(u, net.cfg.steps)
    r = h[:, net.out_idx]
    sd = r.std(0)
    print(f"scale {scale} bias {bias} in_gain {gain}: all-neuron active {float((h > 0).float().mean()):.2f}  "
          f"readout mean {float(r.mean()):.4f}  position-dependent outputs {float((sd > 1e-4).float().mean()):.2f}  "
          f"median sd/mean {float((sd / r.mean(0).clamp_min(1e-9)).median()):.3f}  saturated {float((h >= 10).float().mean()):.3f}")
