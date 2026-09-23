"""Sanity check before spending GPU-days: can the network learn a simple chess rule, supervised?
Target = the greedy mover's choice (mate > best capture) on positions that have a capture."""
import sys, time
import chess, numpy as np, torch
from chessfly.arena import greedy_mover
from chessfly.connectome import load_malecns
from chessfly.encoding import encode, legal_move_indices, move_index
from chessfly.model import FlyChessNet, ModelConfig
from chessfly.selfplay import Sample
from chessfly.train import TrainConfig, make_optimizer, train_step, collate

rng = np.random.default_rng(0)
samples = []
while len(samples) < 6000:
    b = chess.Board()
    for _ in range(80):
        if b.is_game_over(): break
        if any(b.is_capture(m) for m in b.legal_moves):
            moves, idx = legal_move_indices(b)
            pi = np.zeros(len(moves), np.float16); pi[moves.index(greedy_mover(b, rng))] = 1
            samples.append(Sample(encode(b).astype(np.float16), idx.astype(np.int16), pi, 1))
        ms = list(b.legal_moves); b.push(ms[rng.integers(len(ms))])
train, test = samples[:5000], samples[5000:]
cfg = ModelConfig(shuffled="shuffled" in sys.argv, steps=0 if "linear" in sys.argv else 4)
net = FlyChessNet(load_malecns("data"), cfg).cuda()
opt = make_optimizer(net, TrainConfig())

def accuracy(data):
    net.eval(); hit = 0
    with torch.no_grad():
        for i in range(0, len(data), 250):
            x, pi, legal, _, _ = collate(data[i:i + 250], "cuda")
            hit += (net(x)[0].float().masked_fill(~legal, -1e9).argmax(-1) == pi.argmax(-1)).sum().item()
    return hit / len(data)

t = time.time()
for step in range(1, 301):
    net.train()
    l = train_step(net, opt, [train[i] for i in rng.integers(len(train), size=128)], "cuda")
    if step % 100 == 0:
        print(f"step {step} loss {l['policy_loss']:.3f} train acc {accuracy(train[:1000]):.3f} test acc {accuracy(test):.3f}  {time.time() - t:.0f}s", flush=True)
g = net.brain.log_gain
print(f"log-gain moved: mean |dg| {g.abs().mean():.4f}, max {g.abs().max():.2f}")
