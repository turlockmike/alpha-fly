"""Policy/value network whose only hidden computation is the fly connectome.

    board features --linear--> sensory (afferent) neurons
                               ... `steps` recurrent updates through the fixed wiring ...
    output neurons (efferent + descending) --linear--> move logits, value

The encoder and the heads are single linear maps: with `steps=0` wiring removed the model is
linear in the board, so any nonlinear chess knowledge has to live in the connectome's gains,
biases and leaks.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from .brain import ConnectomeRNN
from .connectome import Connectome
from .encoding import N_BASE, N_FEATURES, N_MOVES, move_squares

READOUT_SUPERCLASSES = ("descending_neuron",)   # plus every efferent neuron


@dataclasses.dataclass
class ModelConfig:
    steps: int = 4              # 3 synapses already connect sensory neurons to 99.7% of the outputs
    min_syn: int = 3
    shuffled: bool = False      # control: randomly rewired connectome
    attack_planes: bool = False # feed the rule-derived attack planes (encoding.py); off = the network never sees them
    input_gain: float = 5.0     # these three were picked with scripts/probe_init.py: 96% of neurons active and
    weight_scale: float = 2.0   # 99% of output neurons position-dependent at initialisation
    bias_init: float = 0.1      # tonic drive, so that inhibition is visible through the ReLU


class RunningNorm(nn.Module):
    """Per-neuron standardisation with running statistics in *both* modes.  It removes the resting
    pattern, which is ~30x larger than the position-dependent part of the output activity.
    BatchNorm was wrong here: the activity statistics move quickly while the gains train, its
    running averages lagged, and self-play (eval mode) saw a policy loss of 4.7 where training
    (batch statistics) saw 2.3 on the same positions."""

    def __init__(self, n: int, momentum: float = 0.05, eps: float = 1e-8, clip: float = 8.0):
        super().__init__()
        self.momentum, self.eps, self.clip = momentum, eps, clip
        self.register_buffer("mean", torch.zeros(n))
        self.register_buffer("var", torch.ones(n))
        self.register_buffer("initialised", torch.tensor(False))

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        if self.training and r.shape[0] > 1:
            with torch.no_grad():
                m = 1.0 if not self.initialised else self.momentum
                self.mean.lerp_(r.mean(0), m)
                self.var.lerp_(r.var(0), m)
                self.initialised.fill_(True)
        return ((r - self.mean) / (self.var + self.eps).sqrt()).clamp(-self.clip, self.clip)


class FlyChessNet(nn.Module):
    def __init__(self, conn: Connectome, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # Inference precision.  PLAIN float16 was the default until iteration 290 of fly6_noadj and it was WRONG: as the gains
        # grew (some to e^5) and the output normaliser divides by standard deviations of ~2e-3, half-precision rounding in the
        # last recurrent steps swamped the signal - the best legal move agreed with float32 on 59.6% of 1,000 positions and the
        # value was off by 0.06 on average (sd 0.43), so self-play and every Elo measurement ran on a noisy copy of the trained
        # network (same checkpoint, same seeds: 74% / 42% against two ladder rungs in float16, 80% / 45% in float32).  It had
        # agreed on 98% when checked at iteration ~35.  float16 now means the deviation form (brain.FUSED_DEVIATION): 100.0%
        # agreement, value error 0.0004, 15.9 ms per 624 positions against 21.6 ms for float32.  A test holds it to float32.
        self.infer_dtype = torch.float16
        if cfg.shuffled:
            conn = conn.shuffled()
        self.brain = ConnectomeRNN(conn, bias_init=cfg.bias_init, global_scale=cfg.weight_scale)
        self.register_buffer("in_idx", conn.nodes("afferent"), persistent=False)
        out = torch.cat([conn.nodes("efferent"), conn.nodes_of_superclass(*READOUT_SUPERCLASSES)]).unique()
        self.register_buffer("out_idx", out, persistent=False)
        self.encoder = nn.Linear(N_FEATURES, len(self.in_idx))
        self.norm = RunningNorm(len(out))
        self.policy = nn.Linear(len(out), N_MOVES)
        self.value = nn.Linear(len(out), 3)     # win / draw / loss logits for the side to move
        # logit(move) = policy[move] + policy_from[its from-square] + policy_to[its to-square], all linear in the output
        # neurons.  Each of the 4,168 rows of `policy` is rarely the target and learns alone; the two shared terms are
        # trained by every move that touches the square.  Zero-initialised, so adding them to a trained network changes nothing.
        self.policy_from, self.policy_to = nn.Linear(len(out), 64), nn.Linear(len(out), 64)
        frm, to = move_squares()
        self.register_buffer("move_from", torch.as_tensor(frm), persistent=False)
        self.register_buffer("move_to", torch.as_tensor(to), persistent=False)
        for layer in (self.policy_from, self.policy_to):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)
        nn.init.zeros_(self.policy.weight)
        nn.init.zeros_(self.value.weight)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, N_FEATURES) -> policy logits (B, N_MOVES), WDL logits (B, 3)."""
        dtype = self.infer_dtype if x.is_cuda and not torch.is_grad_enabled() else torch.float32
        if self.cfg.attack_planes:
            enc = self.encoder(x)
        else:                                   # the attack planes are never seen and their columns never trained
            enc = nn.functional.linear(x[:, :N_BASE], self.encoder.weight[:, :N_BASE], self.encoder.bias)
        u_in = (enc * self.cfg.input_gain).to(dtype).T.contiguous()                 # (input neurons, batch)
        r = self.norm(self.brain.forward_io(u_in, self.in_idx, self.out_idx, self.cfg.steps).T.float())
        logits = self.policy(r) + self.policy_from(r)[:, self.move_from] + self.policy_to(r)[:, self.move_to]
        return logits, self.value(r)

    @staticmethod
    def expected_value(wdl_logits: torch.Tensor) -> torch.Tensor:
        p = wdl_logits.softmax(-1)
        return p[..., 0] - p[..., 2]


class Evaluator:
    """The `evaluate` callable of the search: numpy in, numpy out.  Legal-move logits are gathered on
    the GPU, so tens of kilobytes leave the card per batch instead of B x 4168 floats."""

    def __init__(self, net: FlyChessNet, device):
        self.net, self.device = net, device

    @torch.no_grad()
    def submit(self, x, rows, cols):
        """Launch the evaluation and return at once (CUDA runs asynchronously); `result` waits for it."""
        self.net.eval()
        logits, wdl = self.net(torch.from_numpy(x).to(self.device, non_blocking=True).float())
        if rows is None:                                  # lazy search: the whole policy vector, half precision
            return logits.half(), self.net.expected_value(wdl.float())
        legal = logits[torch.from_numpy(rows).to(self.device, non_blocking=True), torch.from_numpy(cols).to(self.device, non_blocking=True)]
        return legal.float(), self.net.expected_value(wdl.float())

    @staticmethod
    def result(handle):
        return handle[0].cpu().numpy(), handle[1].float().cpu().numpy()

    def __call__(self, x, rows, cols):
        return self.result(self.submit(x, rows, cols))


NEW_HEADS = ("policy_from.", "policy_to.")


def load_weights(net: FlyChessNet, state: dict) -> None:
    """Checkpoints from before the factorised policy head lack its (zero-initialised) terms; nothing else may be missing."""
    state = dict(state)
    w = state.get("encoder.weight")
    if w is not None and w.shape[1] < net.encoder.weight.shape[1]:      # saved before the attack planes: their columns start at
        pad = w.new_zeros(w.shape[0], net.encoder.weight.shape[1] - w.shape[1])      # zero, so the network is unchanged
        state["encoder.weight"] = torch.cat([w, pad], dim=1)
    missing, unexpected = net.load_state_dict(state, strict=False)
    bad = [k for k in missing if not k.startswith(NEW_HEADS)] + list(unexpected)
    if bad:
        raise RuntimeError(f"checkpoint does not match the network: {bad}")


def load_net(path, data_dir="data", device="cuda") -> FlyChessNet:
    """The network of a checkpoint (runs/<name>/last.pt) or snapshot (runs/<name>/snapshots/iter_*.pt), ready to evaluate."""
    from .connectome import load_malecns
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ck["config"])
    net = FlyChessNet(load_malecns(data_dir, cfg.min_syn), cfg).to(device)
    load_weights(net, ck["model"])
    return net.eval()
