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



def test_eager_triton_batch_submits_fresh_dependency_chain(monkeypatch):
    import jittor as jt
    if not capability.check_accelerator('cuda', backend=jt).enabled:
        pytest.skip('real CUDA and Triton are required')
    triton = pytest.importorskip('triton')
    global tl
    import triton.language as tl
    from jittor.compat.triton import backend, batch_launches

    @triton.jit
    def scale(src, dst, factor: tl.constexpr):
        i = tl.arange(0, 64)
        tl.store(dst + i, tl.load(src + i) * factor)

    monkeypatch.setenv("JITTOR_TRITON_GUARD_ENABLE", "0")
    monkeypatch.setenv("JITTOR_TRITON_FAST_SYNC", "1")
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0), jt.no_grad():
        for value in (3., 11.):
            src, intermediate, result = jt.full((64,), value), jt.empty((64,)), jt.empty((64,))
            with batch_launches((scale,)) as batch:
                backend.run(scale, (src, intermediate), {"factor": 2}, (1,))
                backend.run(scale, (intermediate, result), {"factor": 5}, (1,))
            assert batch.submitted_launches == 2
            jt.sync([result], True)
            assert result.location() == "device"
            np.testing.assert_array_equal(result.numpy(), np.full(64, value * 10))


@pytest.mark.parametrize("allow_strided", [False, True])
def test_eager_triton_batch_can_write_padded_storage(monkeypatch, allow_strided):
    import jittor as jt
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("real CUDA and Triton are required")
    triton = pytest.importorskip("triton")
    global tl
    import triton.language as tl
    from jittor.compat.triton import backend
    from jittor.kernels import batch_triton_launches

    @triton.jit
    def store(dst, WIDTH: tl.constexpr, PITCH: tl.constexpr, VALUE: tl.constexpr):
        lane = tl.arange(0, 16)
        row = tl.program_id(0)
        tl.store(dst + row * PITCH + lane, VALUE, mask=lane < WIDTH)

    monkeypatch.setenv("JITTOR_TRITON_GUARD_ENABLE", "0")
    monkeypatch.setenv("JITTOR_TRITON_FAST_SYNC", "1")
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback="error"), jt.no_grad():
        raw = jt.zeros((3, 16), dtype="float32")
        dst = raw[:, :13]
        assert tuple(dst._storage_strides()) == (16, 1)
        jt.sync([dst], True)
        with batch_triton_launches((store,), allow_strided=allow_strided) as batch:
            backend.run(store, (dst,), {"WIDTH": 13, "PITCH": 16, "VALUE": 7.0}, (3,))
        assert batch.submitted_launches == int(allow_strided)
        jt.sync([raw], True)
        assert raw.location() == "device"
        expected = np.zeros((3, 16), dtype=np.float32)
        expected[:, :13] = 7
        np.testing.assert_array_equal(raw.numpy(), expected)


def test_eager_triton_batch_preserves_pending_readers_without_a_global_flush(monkeypatch):
    import jittor as jt
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("real CUDA and Triton are required")
    triton = pytest.importorskip("triton")
    global tl
    import triton.language as tl
    from jittor.compat.triton import backend, batch_launches

    @triton.jit
    def increment(value):
        i = tl.arange(0, 32)
        tl.store(value + i, tl.load(value + i) + 1)

    monkeypatch.setenv("JITTOR_TRITON_GUARD_ENABLE", "0")
    monkeypatch.setenv("JITTOR_TRITON_FAST_SYNC", "1")
    with jt.flag_scope(use_cuda=1, auto_graph_replay=0,
                       lazy_execution=1, auto_flush_ops=0), jt.no_grad():
        data = np.arange(32, dtype=np.float32)
        x = jt.array(data)
        x.sync(True, False)
        with batch_launches((increment,)) as warm:
            backend.run(increment, (x,), {}, (1,))
        assert warm.submitted_launches == 1
        x.sync(True, False)

        snapshot = x.clone()
        unrelated = jt.ones((113,), dtype="float32") * 19
        assert jt.core._first_finished([snapshot, unrelated]) == -1
        with batch_launches((increment,)) as batch:
            backend.run(increment, (x,), {}, (1,))
        assert batch.submitted_launches == 1
        assert jt.core._first_finished([unrelated]) == -1
        np.testing.assert_array_equal(snapshot.numpy(), data + 1)
        np.testing.assert_array_equal(x.numpy(), data + 2)
