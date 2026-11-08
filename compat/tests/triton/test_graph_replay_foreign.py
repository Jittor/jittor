"""A kept Jittor graph must not omit opaque Triton launches on replay."""

import numpy as np
import pytest

from _helpers import capability


@pytest.mark.parametrize('automatic', [False, True])
def test_foreign_triton_launch_recomputes_changed_inputs(automatic):
    import jittor as jt
    if not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('real CUDA and Triton are required')
    triton = pytest.importorskip('triton')
    global tl
    import triton.language as tl
    from jittor.compat.triton import backend

    @triton.jit
    def external_copy(x, output, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        tl.store(output + i, tl.load(x + i) * 2)

    class Foreign(jt.nn.Module):
        def execute(self, x):
            output = jt.empty(x.shape, x.dtype)
            backend.run(external_copy, (x, output), {'BLOCK': 32}, (1,))
            return output + 1

    with jt.flag_scope(use_cuda=1, auto_graph_replay=int(automatic)), jt.no_grad():
        model = Foreign()
        call = model if automatic else jt.graph_replay(model)
        for value in (10, 20, 30, 40, 50):
            x = jt.full((32,), value, dtype='float32')
            actual = call(x).numpy()
            np.testing.assert_array_equal(actual, np.full(32, value * 2 + 1),
                                          err_msg='foreign launch was omitted on replay')
