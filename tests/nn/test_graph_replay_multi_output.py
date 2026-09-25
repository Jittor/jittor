"""Auto-replay must survive a module that returns more than one Var.

The warm-up call that runs before a capture used to call ``.sync()`` on the
return value, which only a single Var answers, so every multi-output module
raised ``AttributeError: 'tuple' object has no attribute 'sync'`` from inside
auto-replay. ``nn.RNN``, ``nn.LSTM`` and ``nn.GRU`` all return
``(output, hidden)``. A capture now replays the whole structure.

Auto-replay arms itself only under ``no_grad`` with repeated calls of one
signature (see ``auto_replay_for``), which is why this failed in an inference
loop and nowhere else.
"""
import numpy as np
import pytest

import jittor as jt
from jittor import nn

BUILDERS = [
    ("RNN", lambda: nn.RNN(3, 4, nonlinearity="tanh")),
    ("LSTM", lambda: nn.LSTM(3, 4)),
    ("GRU", lambda: nn.GRU(3, 4)),
]


@pytest.mark.parametrize("name,build", BUILDERS, ids=[n for n, _ in BUILDERS])
def test_a_multi_output_module_runs_under_auto_replay(name, build):
    x = jt.array(np.arange(6, dtype=np.float32).reshape(2, 1, 3) / 8)
    with jt.flag_scope(use_cuda=0):
        model = build()
        model.eval()
        with jt.no_grad():
            reference, _ = model(x)
            reference = reference.numpy().copy()

        # `no_grad` plus a repeated signature is what arms auto-replay; two
        # calls are its threshold, so run past it.
        with jt.flag_scope(auto_graph_replay=1), jt.no_grad():
            for _ in range(4):
                output, hidden = model(x)
                jt.sync_all()

        assert isinstance(hidden, (jt.Var, tuple)), type(hidden)
        np.testing.assert_allclose(output.numpy(), reference,
                                   rtol=1e-5, atol=1e-6)


def test_the_replay_answers_every_part_of_a_tuple():
    """Both outputs are replayed, for each new input -- not one of the two."""
    from jittor._runtime.graph_replay import GraphReplay

    rng = np.random.RandomState(0)
    feeds = [jt.array(rng.randn(2, 1, 3).astype(np.float32)) for _ in range(4)]
    with jt.flag_scope(use_cuda=0), jt.no_grad():
        model = nn.RNN(3, 4, nonlinearity="tanh")
        model.eval()
        expected = []
        for x in feeds:
            out, hidden = model(x)
            expected.append((out.numpy().copy(), hidden.numpy().copy()))
        replay = GraphReplay(model, measure=False, weak=True)
        for x, (want_out, want_hidden) in zip(feeds + feeds, expected + expected):
            result = replay(x)
            assert isinstance(result, tuple) and len(result) == 2, type(result)
            np.testing.assert_allclose(result[0].numpy(), want_out, rtol=1e-5, atol=1e-6)
            np.testing.assert_allclose(result[1].numpy(), want_hidden, rtol=1e-5, atol=1e-6)
        assert replay.refused is None, replay.refused
        assert replay.stats["replayed"] >= 6, replay.stats
