import chess
import numpy as np
import pandas as pd
import torch

from chessfly.brain import ConnectomeRNN
from chessfly.connectome import Connectome
from chessfly.encoding import N_FEATURES, N_MOVES, encode, legal_move_indices
from chessfly.mcts import MCTSConfig, Search, run_simulations


def tiny_connectome(n=30, e=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    pairs = torch.randperm(n * n, generator=g)[:e].sort().values     # unique, (post, pre) order
    post, pre = pairs // n, pairs % n
    nt_sign = torch.where(torch.rand(n, generator=g) < 0.6, 1.0, -1.0)
    flow = ["afferent"] * 5 + ["intrinsic"] * (n - 10) + ["efferent"] * 5
    neurons = pd.DataFrame({"root_id": np.arange(n), "super_class": flow, "cell_type": "", "nt": "", "flow": flow})
    return Connectome(neurons, pre, post, torch.randint(1, 20, (e,), generator=g).float(), nt_sign[pre])


def dense_reference(rnn, u, steps):
    W = torch.zeros(rnn.n, rnn.n, dtype=u.dtype).index_put_((rnn.post, rnn.pre), rnn.edge_weights().to(u.dtype))
    alpha, h = torch.sigmoid(rnn.alpha_logit).to(u.dtype), torch.zeros_like(u)
    for _ in range(steps):
        h = (1 - alpha) * h + alpha * torch.relu(h @ W.T + rnn.bias.to(u.dtype) + u).clamp(max=rnn.h_max)
    return h


def test_sparse_rnn_matches_dense_forward_and_gradients():
    rnn = ConnectomeRNN(tiny_connectome(), global_scale=3.0).double()
    u = torch.rand(4, rnn.n, dtype=torch.double, requires_grad=True)
    target = torch.rand(4, rnn.n, dtype=torch.double)
    grads = []
    for fn in (lambda: rnn(u.T.contiguous(), 5).T, lambda: dense_reference(rnn, u, 5)):
        rnn.zero_grad(); u.grad = None
        out = fn()
        ((out - target) ** 2).sum().backward()
        grads.append((out.detach(), u.grad.clone(), rnn.log_gain.grad.clone(), rnn.bias.grad.clone()))
    for a, b in zip(*grads):
        assert a.abs().sum() > 0
        torch.testing.assert_close(a, b)
    with torch.no_grad():                                   # the in-place inference path computes the same thing
        torch.testing.assert_close(rnn(u.detach().T.contiguous(), 5).T, grads[1][0])


def test_cuda_sampled_gradient_matches_gather():
    if not torch.cuda.is_available():
        return
    import chessfly.brain as brain
    rnn = ConnectomeRNN(tiny_connectome(200, 3000), global_scale=3.0).cuda()
    u = torch.rand(200, 16, device="cuda")
    grads = []
    for flag in (True, False):
        brain.SAMPLED_ADDMM = flag
        rnn.zero_grad()
        rnn(u.clone(), 4).square().sum().backward()
        grads.append(rnn.log_gain.grad.clone())
    brain.SAMPLED_ADDMM = True
    assert grads[0].abs().sum() > 0
    torch.testing.assert_close(grads[0], grads[1], rtol=1e-4, atol=1e-6)


def test_shuffled_keeps_degrees_and_dale():
    c = tiny_connectome()
    s = c.shuffled()
    assert torch.equal(torch.bincount(s.post, minlength=30), torch.bincount(c.post, minlength=30))
    assert torch.equal(torch.bincount(s.pre, minlength=30), torch.bincount(c.pre, minlength=30))
    nt_sign = torch.ones(30).index_put_((c.pre,), c.sign)
    assert torch.equal(s.sign, nt_sign[s.pre])
    assert torch.all(s.post[1:] >= s.post[:-1])


def test_move_indices_unique_and_in_range():
    for fen in (chess.STARTING_FEN, "r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPpP/R3K2R b KQkq - 0 1",
                "8/P6k/8/8/8/8/p6K/8 w - - 0 1"):
        board = chess.Board(fen)
        for b in (board, board.mirror()):
            moves, idx = legal_move_indices(b)
            assert len(set(idx.tolist())) == len(moves) and idx.min() >= 0 and idx.max() < N_MOVES


def test_encoding_is_colour_symmetric():
    board = chess.Board("r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPpP/R3K2R b KQkq e3 0 1")
    assert encode(board).shape == (N_FEATURES,)
    np.testing.assert_array_equal(encode(board), encode(board.mirror()))
    m = next(iter(board.legal_moves))
    mm = chess.Move(chess.square_mirror(m.from_square), chess.square_mirror(m.to_square), m.promotion)
    from chessfly.encoding import move_index
    assert move_index(m, board.turn) == move_index(mm, not board.turn)


def uniform_net(x, rows, cols):
    if rows is None:                                     # lazy search asks for the whole policy vector
        return np.zeros((len(x), N_MOVES), np.float16), np.zeros(len(x), np.float32)
    return np.zeros(len(rows), np.float32), np.zeros(len(x), np.float32)


def test_mcts_finds_mate_in_one_with_uniform_net():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    s = Search(board, MCTSConfig(sims=200), np.random.default_rng(0))
    run_simulations([s], uniform_net, 200, noise=False)
    assert s.root.moves[int(s.root.N.argmax())] == chess.Move.from_uci("a1a8")
    assert board.fen() == "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"      # search leaves the board untouched


def test_selfplay_training_and_arena_smoke():
    """Whole loop on a tiny random 'connectome', CPU."""
    from chessfly.arena import evaluate_strength
    from chessfly.model import Evaluator, FlyChessNet, ModelConfig
    from chessfly.selfplay import SelfPlay, SelfPlayConfig
    from chessfly.train import TrainConfig, make_optimizer, train_step, validation_loss
    net = FlyChessNet(tiny_connectome(60, 600), ModelConfig(steps=3))
    mcts = MCTSConfig(sims=4)
    sp = SelfPlay(Evaluator(net, "cpu"), mcts, SelfPlayConfig(parallel_games=4, workers=0, max_plies=12))
    samples = sp.play(4)
    assert len(samples) >= 4 * 12 * 0.5 and all(abs(s.pi.astype(np.float32).sum() - 1) < 1e-2 for s in samples)
    opt = make_optimizer(net, TrainConfig())
    net.train()
    first = train_step(net, opt, samples[:32], "cpu")
    for _ in range(30):
        last = train_step(net, opt, samples[:32], "cpu")
    assert last["policy_loss"] < first["policy_loss"]
    assert validation_loss(net, samples[32:], "cpu")["val_policy_loss"] > 0
    r = evaluate_strength(Evaluator(net, "cpu"), mcts, n_games=2)
    assert any(k.startswith("vs_") and 0 <= v <= 1 for k, v in r.items())


def test_worker_processes_play_complete_games(tmp_path):
    from chessfly.selfplay import SelfPlayConfig
    from chessfly.workers import ParallelSelfPlay
    sp = ParallelSelfPlay(uniform_net, MCTSConfig(sims=3), SelfPlayConfig(parallel_games=8, workers=2, max_plies=10, record_fraction=1.0))
    try:
        samples = sp.play(8)
    finally:
        sp.close()
    assert sp.stats["games"] >= 8 and len(samples) >= 8 * 10 * 0.5
    import io, chess.pgn
    from chessfly.selfplay import write_selfplay_pgn
    assert len(sp.stats.records) == sp.stats["games"] and all(g.termination for g in sp.stats.records)
    write_selfplay_pgn(tmp_path / "selfplay" / "iter_000001.pgn", sp.stats.records, "t", 1)
    pgn = open(tmp_path / "selfplay" / "iter_000001.pgn")
    game = chess.pgn.read_game(pgn)
    assert game.errors == [] and game.end().ply() == int(game.headers["PlyCount"]) == sp.stats.records[0].plies
    pgn.close()
    assert all(len(s.idx) == len(s.pi) and abs(float(s.pi.astype(np.float32).sum()) - 1) < 1e-2 for s in samples)


def test_rating_fit_recovers_known_gaps():
    from chessfly.ladder import ANCHOR, estimate_elo, expected, fit_ratings
    true = {"a": 0.0, "b": 200.0, ANCHOR[0]: 500.0}
    names = list(true)
    results = [(x, y, 1000 * expected(true[x], true[y]), 1000) for i, x in enumerate(names) for y in names[i + 1:]]
    fit = fit_ratings(names, results)
    assert fit[ANCHOR[0]] == ANCHOR[1]
    assert abs((fit["b"] - fit["a"]) - 200) < 5 and abs((fit[ANCHOR[0]] - fit["b"]) - 300) < 5
    elo, se = estimate_elo([(1000.0, 64 * expected(1100, 1000), 64), (1200.0, 64 * expected(1100, 1200), 64)])
    assert abs(elo - 1100) < 10 and 30 < se < 80


def test_suite_metrics_reward_the_best_move():
    from chessfly.suite import Suite, policy_metrics, search_metrics
    fens = [chess.STARTING_FEN, "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"]
    xs, rows, cols, cps, offsets = [], [], [], [], [0]
    for i, fen in enumerate(fens):
        b = chess.Board(fen)
        moves, idx = legal_move_indices(b)
        xs.append(encode(b)); rows.append(np.full(len(moves), i)); cols.append(idx)
        cps.append(np.arange(len(moves), dtype=np.float32) * 10)            # the last legal move is "best"
        offsets.append(offsets[-1] + len(moves))
    cp, offsets = np.concatenate(cps), np.asarray(offsets)
    suite = Suite(fens, np.stack(xs), np.concatenate(rows), np.concatenate(cols), cp, offsets,
                  np.maximum.reduceat(cp, offsets[:-1]), {})
    oracle = lambda x, r, c: (cp[:len(r)] if len(x) == 2 else np.zeros(len(r), np.float32), np.zeros(len(x), np.float32))
    m = policy_metrics(oracle, suite)
    assert m["acpl_policy"] == 0 and m["top1_policy"] == 1
    worst = lambda x, r, c: (-cp[:len(r)], np.zeros(len(x), np.float32))
    assert policy_metrics(worst, suite)["acpl_policy"] > 50
    assert search_metrics(uniform_net, suite, MCTSConfig(sims=8))["acpl_search"] >= 0


def test_pause_flag_and_resume_config(tmp_path):
    import json, signal
    from chessfly.train import Pause, parse_config
    (tmp_path / "r").mkdir()
    (tmp_path / "r" / "PAUSE").touch()                       # a stale request must not stop the new session
    pause = Pause(tmp_path / "r")
    try:
        assert not pause()
        signal.raise_signal(signal.SIGINT)                   # first Ctrl+C: finish the step, save, exit
        assert pause()
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
    (tmp_path / "r" / "PAUSE").touch()
    assert Pause(tmp_path / "r")() is False
    signal.signal(signal.SIGINT, signal.default_int_handler)

    (tmp_path / "r" / "config.json").write_text(json.dumps({"mcts": {"sims": 16}, "train": {"lr_io": 0.5}}))
    _, _, cfgs, overrides = parse_config(tmp_path, ["--run", "r"])
    assert cfgs["mcts"].sims == 16 and cfgs["train"].lr_io == 0.5 and overrides == {}
    _, _, cfgs, overrides = parse_config(tmp_path, ["--run", "r", "--sims", "64"])
    assert cfgs["mcts"].sims == 64 and cfgs["train"].lr_io == 0.5 and overrides == {"sims": [16, 64]}


def test_arena_worker_reports_a_rolling_elo(tmp_path):
    from chessfly.ladder import find_stockfish, load_ladder
    if load_ladder() is None or find_stockfish() is None:
        return
    from chessfly.selfplay import SelfPlayConfig
    from chessfly.workers import ParallelSelfPlay
    sp = ParallelSelfPlay(uniform_net, MCTSConfig(sims=1), SelfPlayConfig(parallel_games=4, workers=1, max_plies=6),
                          arena={"games": 4, "window": 50, "elo_guess": None})
    try:
        reports = []
        for _ in range(400):                                 # the arena's games are 200 plies; self-play's are 6
            sp.play(4)
            reports += sp.take_arena_reports()
            if reports:
                break
    finally:
        sp.close()
    assert reports and {"elo", "elo_se", "elo_games", "pgn"} <= set(reports[0]) and any(k.startswith("vs_") for k in reports[0])
    assert sp.stats["moves"] > 0


def test_policy_target_pruning_removes_single_visits():
    from chessfly.train import prune
    visits = np.array([1, 1, 3, 1, 0, 2], dtype=np.float32)
    pi = (visits / visits.sum()).astype(np.float16)
    np.testing.assert_allclose(prune(pi, 0.0), pi.astype(np.float32))
    np.testing.assert_allclose(prune(pi, 1.0), np.array([0, 0, 2, 0, 0, 1]) / 3, atol=2e-3)
    flat = np.full(4, 0.25, dtype=np.float16)                  # every move visited once: nothing to prune towards
    np.testing.assert_allclose(prune(flat, 1.0), flat.astype(np.float32))


def material_net(x, rows, cols):
    """A stub whose value is a function of the position itself, so a misrouted reply is detectable."""
    x = np.asarray(x, dtype=np.float32)
    w = np.repeat(np.array([1, 3, 3, 5, 9, 0], dtype=np.float32), 64)
    diff = x[:, :384] @ w - x[:, 384:768] @ w
    legal = np.zeros((len(x), N_MOVES), np.float16) if rows is None else np.zeros(len(rows), np.float32)
    return legal, np.tanh(diff / 3).astype(np.float32)


class AsyncMaterialNet:
    """Same stub through the submit / result interface the pipelined trainer uses."""
    def submit(self, x, rows, cols): return material_net(x, rows, cols)
    def result(self, handle): return handle
    def __call__(self, x, rows, cols): return material_net(x, rows, cols)


def test_pipelined_workers_route_every_reply_to_its_own_game():
    from chessfly.selfplay import SelfPlay, SelfPlayConfig
    from chessfly.workers import ParallelSelfPlay
    w = np.repeat(np.array([1, 3, 3, 5, 9, 0], dtype=np.float32), 64)

    def agreement(samples):
        x = np.stack([s.x for s in samples]).astype(np.float32)
        own = np.tanh((x[:, :384] @ w - x[:, 384:768] @ w) / 3)
        q = np.array([s.q for s in samples])
        keep = np.abs(own) > 0.3                     # positions where one side is clearly ahead
        return float(np.mean(np.sign(q[keep]) == np.sign(own[keep]))), int(keep.sum())

    cfg = dict(parallel_games=12, max_plies=60, temperature_plies=60, td_lambda=0.0)     # lambda 0: Sample.q is the position's own search value
    base = SelfPlay(material_net, MCTSConfig(sims=6), SelfPlayConfig(workers=0, **cfg)).play(12)
    sp = ParallelSelfPlay(AsyncMaterialNet(), MCTSConfig(sims=6), SelfPlayConfig(workers=3, **cfg))
    try:
        piped = sp.play(12)
    finally:
        sp.close()
    (a0, n0), (a1, n1) = agreement(base), agreement(piped)
    assert n0 > 30 and n1 > 30
    assert a0 > 0.9 and a1 > 0.9, (a0, a1)           # the search value of a position reflects *that* position's material


def test_gumbel_schedule_is_sequential_halving():
    from chessfly.mcts import considered_visits
    seq = considered_visits(16, 32)
    assert seq == (0,) * 16 + (1,) * 8 + (2,) * 4 + (3,) * 4
    assert considered_visits(1, 5) == (0, 1, 2, 3, 4) and len(considered_visits(5, 33)) == 33


def test_gumbel_search_finds_mate_and_targets_it():
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    cfg = MCTSConfig(sims=32, gumbel=True)
    for explore in (False, True):                      # a mate is worth far more than any Gumbel draw
        s = Search(board.copy(), cfg, np.random.default_rng(3), explore=explore)
        run_simulations([s], uniform_net, cfg.sims, noise=True)
        mate = s.root.moves.index(chess.Move.from_uci("a1a8"))
        assert s.best_action() == mate
        pi = s.policy_target()
        assert abs(pi.sum() - 1) < 1e-5 and pi.min() >= 0 and pi[mate] > 0.9
        assert s._root_visits.sum() == cfg.sims and s._root_visits[mate] == s._root_visits.max()
    # completed Q: unvisited moves take the node's mixed value, visited ones their search value
    q = s.root.completed_q()
    assert q[mate] == 1.0 and np.all(q[s.root.N == 0] == q[s.root.N == 0][0])


def test_gumbel_selfplay_produces_value_based_targets():
    from chessfly.selfplay import SelfPlay, SelfPlayConfig
    sp = SelfPlay(material_net, MCTSConfig(sims=8, gumbel=True), SelfPlayConfig(parallel_games=6, workers=0, max_plies=30))
    samples = sp.play(6)
    assert len(samples) > 60
    assert all(abs(float(s.pi.astype(np.float32).sum()) - 1) < 2e-2 and len(s.pi) == len(s.idx) for s in samples)
    first_moves = {tuple(s.x[:768].astype(np.int8).tolist()) for s in samples}       # Gumbel noise keeps games apart
    assert len(first_moves) > 40


def test_light_cone_equals_the_full_recurrence():
    """forward_io must be the same function as forward: outputs and every gradient, with and without autograd."""
    conn = tiny_connectome(120, 700, seed=3)
    rnn = ConnectomeRNN(conn, global_scale=3.0, bias_init=0.1).double()
    with torch.no_grad():
        rnn.bias.add_(torch.randn(rnn.n, dtype=torch.double) * 0.2)
        rnn.alpha_logit.add_(torch.randn(rnn.n, dtype=torch.double))
        rnn.log_gain.add_(torch.randn(conn.n_edges, dtype=torch.double) * 0.3)
    in_idx, out_idx = torch.arange(0, 9), torch.arange(100, 120)
    for steps in (1, 2, 4, 6):
        u_in = torch.randn(9, 5, dtype=torch.double, requires_grad=True)
        target = torch.randn(20, 5, dtype=torch.double)
        results = []
        for mode in ("full", "cone"):
            rnn.zero_grad(); u_in.grad = None
            if mode == "full":
                u = torch.zeros(rnn.n, 5, dtype=torch.double).index_add(0, in_idx, u_in)
                out = rnn(u.contiguous(), steps)[out_idx]
            else:
                out = rnn.forward_io(u_in, in_idx, out_idx, steps)
            ((out - target) ** 2).sum().backward()
            results.append([out.detach()] + [g.grad.clone() if g.grad is not None else torch.zeros_like(g)
                                             for g in (u_in, rnn.log_gain, rnn.bias, rnn.alpha_logit)])
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)
        assert steps == 1 or results[0][2].abs().sum() > 0        # one step never touches W: it multiplies a zero state
        with torch.no_grad():
            torch.testing.assert_close(rnn.forward_io(u_in.detach(), in_idx, out_idx, steps), results[0][0], rtol=1e-9, atol=1e-11)
    cone = rnn.light_cone(in_idx, out_idx, 4)
    assert cone.edge_visits < 4 * conn.n_edges and len(cone.rows[0]) == 0


_PROJ = np.random.default_rng(7).standard_normal((N_FEATURES, N_MOVES)).astype(np.float32) * 0.3


def projected_net(x, rows, cols):
    """A deterministic 'network': logits and value are fixed functions of the position."""
    x = np.asarray(x, dtype=np.float32)
    logits = (x @ _PROJ).astype(np.float16)
    values = np.tanh(x @ _PROJ[:, 0]).astype(np.float32)
    return (logits, values) if rows is None else (logits[rows, cols].astype(np.float32), values)


def test_lazy_search_builds_the_same_tree_as_eager_search():
    for fen in (chess.STARTING_FEN, "r3k2r/pPppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPpP/R3K2R b KQkq - 0 1"):
        for gumbel in (True, False):
            out = []
            for lazy in (False, True):
                cfg = MCTSConfig(sims=40, gumbel=gumbel, lazy=lazy)
                s = Search(chess.Board(fen), cfg, np.random.default_rng(5))
                run_simulations([s], projected_net, cfg.sims, noise=False)
                out.append((s.root.N.copy(), s.policy_target(), s.best_action(), [m.uci() for m in s.root.moves]))
            assert out[0][3] == out[1][3] and out[0][2] == out[1][2]
            np.testing.assert_array_equal(out[0][0], out[1][0])                      # identical visit counts
            np.testing.assert_allclose(out[0][1], out[1][1], atol=2e-3)
            assert out[0][0].sum() == 40


def test_lazy_workers_use_shared_memory(tmp_path):
    from chessfly.selfplay import SelfPlayConfig
    from chessfly.workers import ParallelSelfPlay
    sp = ParallelSelfPlay(projected_net, MCTSConfig(sims=6, gumbel=True, lazy=True),
                          SelfPlayConfig(parallel_games=10, workers=2, max_plies=16, record_fraction=0.0))
    try:
        samples = sp.play(10)
    finally:
        sp.close()
    assert len(samples) > 80 and all(abs(float(s.pi.astype(np.float32).sum()) - 1) < 2e-2 for s in samples)
    assert max(float(s.pi.max()) for s in samples) > 0.2          # targets are shaped by the projected logits, not flat


def test_td_lambda_targets():
    from chessfly.selfplay import Sample, td_targets
    mk = lambda q: Sample(np.zeros(1, np.float16), np.zeros(1, np.int16), np.ones(1, np.float16), q)
    s = [mk(0.2), mk(-0.4), mk(0.6)]                       # sides alternate; the side to move in the last sample wins
    td_targets(s, 1.0, 0.5)
    g2 = 0.5 * 0.6 + 0.5 * 1.0                             # last sample: its own search value and the result
    g1 = 0.5 * -0.4 - 0.5 * g2                             # the ply before is the opponent's: the future counts with the sign flipped
    g0 = 0.5 * 0.2 - 0.5 * g1
    np.testing.assert_allclose([x.q for x in s], [g0, g1, g2], atol=1e-6)
    s = [mk(0.3), mk(0.1)]
    td_targets(s, -1.0, 1.0)                               # lambda = 1: the result alone, from each side's point of view
    np.testing.assert_allclose([x.q for x in s], [1.0, -1.0])


def test_factorised_policy_head_is_a_no_op_until_trained_and_loads_old_checkpoints():
    from chessfly.encoding import move_index, move_squares
    from chessfly.model import FlyChessNet, ModelConfig, load_weights
    frm, to = move_squares()
    for fen in ("8/P6k/8/8/8/8/p6K/8 w - - 0 1", "8/P6k/8/8/8/8/p6K/8 b - - 0 1", chess.STARTING_FEN):
        b = chess.Board(fen)
        for m in b.legal_moves:
            f, t_ = (m.from_square, m.to_square) if b.turn == chess.WHITE else (m.from_square ^ 56, m.to_square ^ 56)
            i = move_index(m, b.turn)
            assert (frm[i], to[i]) == (f, t_), (fen, m)
    net = FlyChessNet(tiny_connectome(60, 600), ModelConfig(steps=3))
    old = {k: v for k, v in net.state_dict().items() if not k.startswith(("policy_from.", "policy_to."))}
    load_weights(net, old)                                  # a checkpoint from before the shared terms
    x = torch.rand(3, N_FEATURES)
    with torch.no_grad():
        nn_only = net.policy(net.norm(net.brain.forward_io((net.encoder(x) * net.cfg.input_gain).T.contiguous(), net.in_idx, net.out_idx, 3).T))
        torch.testing.assert_close(net.eval()(x)[0], nn_only)
    try:
        load_weights(net, {k: v for k, v in old.items() if k != "value.bias"})
        assert False, "a genuinely incomplete checkpoint must be rejected"
    except RuntimeError:
        pass


def test_playout_cap_records_only_full_searches():
    from chessfly.selfplay import SelfPlay, SelfPlayConfig
    cfg = SelfPlayConfig(parallel_games=16, workers=0, max_plies=40, full_search_prob=0.25, fast_sims=4)
    sp = SelfPlay(material_net, MCTSConfig(sims=16, gumbel=True, lazy=True), cfg, seed=2)
    samples = sp.play(16)
    share = len(samples) / sp.stats["plies"]                    # recorded positions / moves played in the finished games
    assert 0.15 < share < 0.37, share
    assert all(abs(float(s.pi.astype(np.float32).sum()) - 1) < 2e-2 for s in samples)
    everything = SelfPlay(material_net, MCTSConfig(sims=16, gumbel=True, lazy=True),
                          SelfPlayConfig(parallel_games=16, workers=0, max_plies=40), seed=2)
    assert len(everything.play(16)) == everything.stats["plies"]          # default: every move searched in full and recorded


def test_attack_planes_and_their_reconstruction():
    from chessfly.encoding import N_BASE, upgrade_features
    rng = np.random.default_rng(0)
    b = chess.Board()
    for ply in range(60):
        x = encode(b)
        assert x.shape == (N_FEATURES,)
        np.testing.assert_array_equal(upgrade_features(x[:N_BASE].astype(np.float16)).astype(np.float32)[N_BASE:], x[N_BASE:])
        np.testing.assert_array_equal(encode(b.mirror()), x)                  # colour symmetry holds for the new planes
        us_att, them_att = x[N_BASE:N_BASE + 64], x[N_BASE + 64:N_BASE + 128]
        for sq in chess.SQUARES:                                              # against python-chess, in the side-to-move frame
            f = sq if b.turn == chess.WHITE else sq ^ 56
            assert us_att[f] == b.is_attacked_by(b.turn, sq) and them_att[f] == b.is_attacked_by(not b.turn, sq)
        assert x[-1] == b.is_check()
        ms = list(b.legal_moves)
        if not ms:
            break
        b.push(ms[rng.integers(len(ms))])


def test_old_checkpoints_grow_the_encoder_without_changing_the_network():
    from chessfly.encoding import N_BASE
    from chessfly.model import FlyChessNet, ModelConfig, load_weights
    from chessfly.train import TrainConfig, grow_optimizer_state, make_optimizer
    net = FlyChessNet(tiny_connectome(60, 600), ModelConfig(steps=3))
    with torch.no_grad():
        net.policy.weight.normal_(0, 0.05)
    old = {k: (v[:, :N_BASE].clone() if k == "encoder.weight" else v.clone()) for k, v in net.state_dict().items()}
    fresh = FlyChessNet(tiny_connectome(60, 600), ModelConfig(steps=3, attack_planes=True))
    load_weights(fresh, old)
    assert fresh.encoder.weight.shape[1] == N_FEATURES and float(fresh.encoder.weight[:, N_BASE:].abs().sum()) == 0
    x = torch.rand(4, N_FEATURES)
    x_old = x.clone(); x_old[:, N_BASE:] = 0
    with torch.no_grad():
        torch.testing.assert_close(fresh.eval()(x)[0], fresh(x_old)[0])       # the new inputs do nothing until trained
    opt = make_optimizer(fresh, TrainConfig())
    state = {"state": {3: {"step": torch.tensor(5.0), "exp_avg": torch.ones(old["encoder.weight"].shape), "exp_avg_sq": torch.ones(old["encoder.weight"].shape)}},
             "param_groups": opt.state_dict()["param_groups"]}
    grown = grow_optimizer_state(state, opt)
    assert grown["state"][3]["exp_avg"].shape == fresh.encoder.weight.shape and float(grown["state"][3]["exp_avg"][:, N_BASE:].sum()) == 0
    opt.load_state_dict(grown)


def test_attack_planes_are_invisible_unless_switched_on():
    from chessfly.encoding import N_BASE
    from chessfly.model import FlyChessNet, ModelConfig
    off = FlyChessNet(tiny_connectome(60, 600), ModelConfig(steps=3)).eval()
    on = FlyChessNet(tiny_connectome(60, 600), ModelConfig(steps=3, attack_planes=True)).eval()
    on.load_state_dict(off.state_dict())
    with torch.no_grad():
        for net in (off, on):
            net.encoder.weight[:, N_BASE:].normal_(0, 1.0); net.policy.weight.normal_(0, 0.05)
        on.load_state_dict(off.state_dict())
        x = torch.rand(4, N_FEATURES); y = x.clone(); y[:, N_BASE:] = torch.rand(4, N_FEATURES - N_BASE)
        torch.testing.assert_close(off(x)[0], off(y)[0])                     # off: the planes cannot influence anything
        assert not torch.allclose(on(x)[0], on(y)[0])                        # on: they do
    off.train(); off(x)[0].sum().backward()
    assert float(off.encoder.weight.grad[:, N_BASE:].abs().sum()) == 0       # and their weights receive no gradient


def test_half_precision_inference_matches_float32_on_cuda():
    """The float16 path (deviations from the resting trajectory) must reproduce float32 even when the position-dependent
    signal is a small fraction of the resting activity - the situation in which plain float16 silently broke a 300-iteration run."""
    if not torch.cuda.is_available():
        return
    import chessfly.brain as bm
    assert bm.FUSED_DEVIATION is not None and bm.FUSED_UPDATE is not None
    rnn = ConnectomeRNN(tiny_connectome(400, 9000, seed=5), global_scale=2.0, bias_init=0.6).cuda()
    with torch.no_grad():
        rnn.log_gain.add_(torch.randn_like(rnn.log_gain) * 0.8)
        rnn.bias.add_(torch.randn_like(rnn.bias) * 0.7)             # some rows sit on the clamp, below zero and above h_max
        rnn.bias[::17] += 12.0
        rnn.alpha_logit.add_(torch.randn_like(rnn.alpha_logit))
        in_idx, out_idx = torch.arange(0, 40, device="cuda"), torch.arange(300, 400, device="cuda")
        u = torch.randn(40, 64, device="cuda") * 0.05              # a small signal on top of a large resting state
        for steps in (1, 3, 6):
            ref = rnn.forward_io(u, in_idx, out_idx, steps)
            half = rnn.forward_io(u.half(), in_idx, out_idx, steps)
            assert half.dtype == torch.float32
            signal = (ref - ref.mean(1, keepdim=True)).abs().mean()
            assert float((half - ref).abs().max()) < 0.02 * float(signal) + 1e-4, (steps, float((half - ref).abs().max()), float(signal))


def test_random_openings_are_played_unsearched_and_unrecorded():
    """opening_plies: a game's first 0..N plies come from the raw policy; they are not training samples, and the game
    that follows is searched and recorded as usual."""
    from chessfly.selfplay import GamePool, SelfPlayConfig
    from chessfly.mcts import MCTSConfig
    cfg = SelfPlayConfig(parallel_games=16, workers=0, max_plies=20, opening_plies=4, record_fraction=1.0)
    pool = GamePool(cfg.parallel_games, MCTSConfig(sims=4, gumbel=True, lazy=True), cfg, np.random.default_rng(3))
    assert {g.opening for g in pool.games} <= set(range(5)) and any(g.opening > 0 for g in pool.games)
    for _ in range(400):
        x, rows, cols = pool.request()
        legal, values = uniform_net(x, rows, cols) if len(x) else (np.zeros(0, np.float32),) * 2
        pool.deliver(legal, values)
        if len(pool.finished) >= 8:
            break
    done = pool.finished[:8]
    assert done, "no game finished"
    for g in done:
        n_moves = len(g.san.split()) - sum(tok.endswith(".") for tok in g.san.split())
        assert g.plies == n_moves
    openings = [len(pool.games[i].samples) for i in range(len(pool.games))]
    assert all(g.search.board.ply() - len(g.samples) == min(g.opening, g.search.board.ply()) for g in pool.games), openings
