"""Auto-replay must survive a module that returns more than one Var.

``GraphReplay._capture_now`` already refuses such a module and sends it down
the eager path -- that is the design, and it is what should happen here. But
the warm-up call that runs *before* that decision called ``.sync()`` on the
return value, which only a single Var answers, so every multi-output module
raised ``AttributeError: 'tuple' object has no attribute 'sync'`` from inside
auto-replay. ``nn.RNN``, ``nn.LSTM`` and ``nn.GRU`` all return
``(output, hidden)``.

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


def test_the_replay_declines_rather_than_replaying_a_tuple():
    """It must fall back, not silently capture one of the two outputs."""
    from jittor._runtime.graph_replay import GraphReplay

    x = jt.array(np.arange(6, dtype=np.float32).reshape(2, 1, 3) / 8)
    with jt.flag_scope(use_cuda=0), jt.no_grad():
        model = nn.RNN(3, 4, nonlinearity="tanh")
        model.eval()
        replay = GraphReplay(model, measure=False, weak=True)
        out, hidden = replay(x)
        assert replay.refused is not None, "a tuple return must be refused"
        assert "not a single Var" in replay.refused, replay.refused
        assert out.shape[0] == 2
