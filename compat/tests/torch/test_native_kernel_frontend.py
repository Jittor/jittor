"""The functional kernel API must preserve the Torch frontend and its gradients."""
import numpy as np
import pytest
import jittor as jt
import torch

from jittor.kernels import TensorSpec, triton_call, tilelang_call, _Buffer
from _helpers import capability


def test_executor_buffers_bypass_the_eager_triton_bridge():
    from jittor.compat.triton import _args_have_jittor_var
    buffer = _Buffer((0, (4,), (1,), "float32", None), ("cuda", 0), 2)
    assert not _args_have_jittor_var((buffer,), {"output": buffer})
    assert _args_have_jittor_var((torch.zeros((4,), device="cpu"),), {})


def require_cuda():
    if not capability.check_accelerator("cuda", backend=jt).enabled:
        pytest.skip("CUDA required")


def check_double(forward):
    class Double(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return forward(x)
        @staticmethod
        def backward(ctx, dy):
            return dy * 2
    x = torch.arange(32, device="cuda", dtype=torch.float32).requires_grad_()
    y = Double.apply(x)
    assert isinstance(y, torch.Tensor)
    np.testing.assert_array_equal(y.detach().cpu().numpy(), np.arange(32) * 2)
    y.sum().backward()
    np.testing.assert_array_equal(x.grad.cpu().numpy(), np.full(32, 2))


def test_triton_native_kernel_with_torch_frontend():
    require_cuda()
    from jittor.compat.triton import backend
    triton = backend.real_triton()
    if triton is None:
        pytest.skip("Triton compiler is not installed")
    import triton.language as tl
    @triton.jit
    def double(X, Y):
        i = tl.arange(0, 32)
        tl.store(Y + i, tl.load(X + i) * 2)
    check_double(lambda x: triton_call(double, x, TensorSpec(x.shape, x.dtype), grid=(1,)))


def test_tilelang_native_kernel_with_torch_frontend():
    require_cuda()
    tilelang = pytest.importorskip("tilelang")
    import tilelang.language as T
    @T.prim_func
    def double(X: T.Tensor((32,), "float32"), Y: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(32):
                Y[i] = X[i] * 2
    kernel = tilelang.compile(double, out_idx=[], target="cuda", execution_backend="tvm_ffi")
    check_double(lambda x: tilelang_call(kernel, x, TensorSpec(x.shape, x.dtype)))


def test_prepared_triton_uses_current_scalars_and_grid():
    require_cuda()
    pytest.importorskip("triton")
    import triton
    import triton.language as tl
    from jittor.kernels.triton import _prepared_call

    @triton.jit
    def affine(x, output, scale, OFFSET: tl.constexpr):
        index = tl.program_id(0) * 4 + tl.arange(0, 4)
        tl.store(output + index, tl.load(x + index) * scale + OFFSET)

    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        reference = np.arange(16, dtype=np.float32)
        source = torch.tensor(reference, device="cuda")
        graph_count = jt.core.graph_launch_count()
        for scale, offset, blocks in ((0.5, 1, 2), (2.0, 1, 4), (0.5, 3, 4), (1.5, 1, 3)):
            count = blocks * 4
            result = _prepared_call(
                affine, source, TensorSpec((count,)), scale, OFFSET=offset, grid=(blocks,),
            )
            result.sync(False, False)
            assert result.location() == "device"
            np.testing.assert_array_equal(result.cpu().numpy(), reference[:count] * scale + offset)
        assert jt.core.graph_launch_count() == graph_count


def test_reusable_triton_launch_uses_new_storage_and_fresh_outputs():
    require_cuda()
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from jittor.kernels.triton import _prepare_triton_launch

    @triton.jit
    def biased_add(left, right, output, amount):
        idx = tl.arange(0, 16)
        tl.store(output + idx, tl.load(left + idx) + tl.load(right + idx) + amount)

    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback="error"):
        source = torch.arange(16, device="cuda", dtype=torch.float32)
        other = torch.full((16,), 4.0, device="cuda")
        graphs = jt.core.graph_launch_count()
        plan = _prepare_triton_launch(biased_add, source, other, TensorSpec((16,)), 1.5, grid=(1,))
        assert plan.input_names == ("left", "right")
        first = plan(source, other)
        second = plan(source + 3.0, other + 5.0)
        jt.sync([first, second], False, False)
        assert first.location() == second.location() == "device"
        np.testing.assert_array_equal(first.cpu().numpy(), np.arange(16) + 5.5)
        np.testing.assert_array_equal(second.cpu().numpy(), np.arange(16) + 13.5)
        with pytest.raises(ValueError, match="shape and dtype"):
            plan(source[:8], other)
        assert jt.core.graph_launch_count() == graphs


@pytest.mark.parametrize("bind_output", [False, True])
def test_reusable_triton_launch_orders_each_bound_state(bind_output):
    require_cuda()
    triton = pytest.importorskip("triton")
    import triton.language as tl
    from jittor.kernels import BoundOutput
    from jittor.kernels.triton import _prepare_triton_launch

    @triton.jit
    def advance(source, delta, state, output):
        i = tl.arange(0, 16)
        value = tl.load(source + i) + tl.load(delta + i)
        tl.store(state + i, value)
        tl.store(output + i, value)

    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0, backend_fallback="error"):
        states = [torch.full((16,), float(value), device="cuda") for value in (10, 20, 30)]
        delta = torch.arange(16, device="cuda", dtype=torch.float32)
        outputs = [torch.empty_like(states[0]) for _ in states] if bind_output else []
        jt.sync(states + outputs, False, False)
        exemplar, *destinations = states
        graphs = jt.core.graph_launch_count()
        output = BoundOutput(outputs[0]) if bind_output else TensorSpec((16,))
        plan = _prepare_triton_launch(
            advance, exemplar, delta, BoundOutput(exemplar), output,
            grid=(1,), align_16=True,
        )
        expected_names = ("source", "delta", "state", "output") if bind_output else ("source", "delta", "state")
        assert plan.input_names == expected_names
        previous = []
        for index, state in enumerate(destinations, 1):
            alias = jt.core._ready_step_gradient(state)
            previous.append(alias.copy())
            buffers = (state, delta, state, outputs[index]) if bind_output else (state, delta, state)
            bound, result = plan(*buffers)
            jt.sync([bound, result], False, False)
            assert bound.location() == result.location() == "device"
            if bind_output:
                assert result._mem_ptr_now == outputs[index]._mem_ptr_now
            np.testing.assert_array_equal(result.numpy(), state.numpy())
        for value, state, prior in zip((20, 30), destinations, previous, strict=True):
            np.testing.assert_array_equal(prior.numpy(), np.full(16, value))
            np.testing.assert_array_equal(state.numpy(), np.arange(16) + value)
        np.testing.assert_array_equal(exemplar.numpy(), np.full(16, 10))
        assert jt.core.graph_launch_count() == graphs


def test_prepared_triton_replays_indexed_state_updates():
    require_cuda()
    from jittor.compat.triton import backend
    from jittor.kernels.triton import _prepared_call

    triton = backend.real_triton()
    if triton is None:
        pytest.skip("Triton compiler is not installed")
    import triton.language as tl

    @triton.jit
    def indexed_increment(state, index, update, output, scale, bias, OFFSET: tl.constexpr):
        row = tl.load(index)
        value = tl.load(state + row) + tl.load(update) * scale + (bias - OFFSET)
        tl.store(output, value)

    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        state = torch.zeros(4, device="cuda")
        numbers = [
            (torch.tensor([i % 4], device="cuda", dtype=torch.int32),
             torch.tensor([i + 1.0], device="cuda", dtype=torch.float32))
            for i in range(12)
        ]
        jt.sync([state, *(value for pair in numbers for value in pair)], True, False)

        def step(index, update):
            row = _prepared_call(
                indexed_increment, state, index, update, TensorSpec((1,)),
                -0.5, 2**33 + 1, OFFSET=2**33, grid=(1,), num_warps=1,
            )
            state[index] = row
            return row

        reference = np.zeros(4, dtype=np.float32)
        captured = jt.capture_step(step)
        before = jt.core.graph_launch_count()
        storage = None
        snapshots = []
        try:
            for i, (index, update) in enumerate(numbers):
                value = captured(index, update)
                reference[i % 4] += 1 - (i + 1) * 0.5
                snapshots.append((value, reference[i % 4]))
                if i == 1:
                    storage = state._mem_ptr_now
                elif i > 1:
                    assert state._mem_ptr_now == storage
            torch.cuda.synchronize()
            assert jt.core.graph_launch_count() > before
        finally:
            captured.invalidate()
        assert state.location() == "device"
        np.testing.assert_array_equal(state.cpu().numpy(), reference)
        for actual, expected in snapshots:
            np.testing.assert_array_equal(actual.cpu().numpy(), [expected])


def test_prepared_triton_initializes_its_own_writable_output():
    require_cuda()
    from jittor.compat.triton import backend
    from jittor.kernels.triton import _InitializedOutput, _prepared_call

    triton = backend.real_triton()
    if triton is None:
        pytest.skip("Triton compiler is not installed")
    import triton.language as tl

    @triton.jit
    def increment(values, amount):
        i = tl.arange(0, 32)
        tl.store(values + i, tl.load(values + i) + tl.load(amount))

    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        source = torch.arange(32, device="cuda", dtype=torch.float32)
        amounts = [torch.tensor([i], device="cuda", dtype=torch.float32) for i in range(8)]
        jt.sync([source, *amounts], True, False)

        def step(amount):
            return _prepared_call(increment, _InitializedOutput(source), amount,
                                  grid=(1,), num_warps=1)

        captured = jt.capture_step(step)
        before = jt.core.graph_launch_count()
        try:
            outputs = [captured(amount) for amount in amounts]
            torch.cuda.synchronize()
            assert jt.core.graph_launch_count() > before
        finally:
            captured.invalidate()
        assert source.location() == "device"
        np.testing.assert_array_equal(source.cpu().numpy(), np.arange(32))
        for offset, value in enumerate(outputs):
            np.testing.assert_array_equal(value.cpu().numpy(), np.arange(32) + offset)


def test_step_replay_preserves_strided_indexed_state_storage():
    require_cuda()
    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        backing = torch.zeros((6, 3, 4), device="cuda", dtype=torch.float32)
        view = backing[:, :2, :]
        jt.sync([view], True, False)
        state = jt.core._ready_step_gradient(view)
        assert state.stride() == (12, 4, 1)
        address = state._mem_ptr_now
        arguments = [
            (torch.tensor([i % 6], device="cuda", dtype=torch.int32),
             torch.full((1, 2, 4), i, device="cuda", dtype=torch.float32))
            for i in range(12)
        ]
        jt.sync([value for pair in arguments for value in pair], True, False)
        axes = jt.NanoVector([0])

        def step(index, row):
            if jt.core._state_capture_active():
                row = row.contiguous()
                jt.sync([index, row], False, False)
                assert state._try_indexed_storage_write(row, [index], axes)
            return state[index].contiguous()

        replay = jt.capture_step(step)
        before = jt.core.graph_launch_count()
        outputs = []
        try:
            for index, row in arguments:
                outputs.append(replay(index, row))
            torch.cuda.synchronize()
            assert jt.core.graph_launch_count() > before
            assert len(replay._capture.state) == 1
            assert replay._capture.state[0][3], "indexed writes should remain in place"
            assert state._mem_ptr_now == address
        finally:
            replay.invalidate()
        assert state.location() == "device"
        for i, output in enumerate(outputs):
            np.testing.assert_array_equal(output.cpu().numpy(), np.full((1, 2, 4), i))
        expected = np.zeros((6, 3, 4), dtype=np.float32)
        for i in range(6):
            expected[i, :2] = i + 6
        np.testing.assert_array_equal(backing.cpu().numpy(), expected)


def test_prepared_kernel_can_read_strided_physical_storage_without_copy():
    require_cuda()
    with torch.inference_mode(), jt.flag_scope(use_cuda=1):
        backing = torch.arange(72, device="cuda", dtype=torch.float32).reshape(6, 3, 4)
        view = backing[:, :2, :]
        jt.sync([view], True, False)
        window = view._physical_storage_window()
        assert window.device == view.device
        assert window._mem_ptr_now == view._mem_ptr_now
        assert window.shape == (68,)
        assert window.is_contiguous()
        np.testing.assert_array_equal(window.cpu().numpy(), np.arange(68))


@pytest.mark.parametrize("physical_window", [False, True])
def test_prepared_triton_writes_bound_storage_after_pending_readers(physical_window):
    require_cuda()
    from jittor.compat.triton import backend
    from jittor.kernels import BoundOutput, triton_call

    triton = backend.real_triton()
    if triton is None:
        pytest.skip("Triton compiler is not installed")
    import triton.language as tl

    @triton.jit
    def advance(source, delta, updated, output):
        i = tl.arange(0, 8)
        value = tl.load(source + i) + tl.load(delta + i)
        tl.store(updated + i, value)
        tl.store(output + i, value)

    with torch.inference_mode(), jt.flag_scope(use_cuda=1, auto_graph_replay=0):
        backing = torch.full((16 if physical_window else 8,), 17.0, device="cuda")
        state = backing[:8] if physical_window else backing
        deltas = [torch.full((8,), float(i % 3 + 1), device="cuda") for i in range(12)]
        jt.sync([state, *deltas], True, False)
        if physical_window:
            state = state._physical_storage_window()
        alias = jt.core._ready_step_gradient(state)
        address = state._mem_ptr_now

        def step(delta):
            _, output = triton_call(
                advance, state, delta, BoundOutput(state), TensorSpec((8,), "float32"),
                grid=(1,), num_warps=1,
            )
            return output

        captured = jt.capture_step(step)
        launch_count = jt.core.graph_launch_count()
        snapshots = []
        expected = 17.0
        try:
            for i, delta in enumerate(deltas):
                prior = alias.copy()
                assert not prior.is_finished
                value = captured(delta)
                value.sync(False, False)
                snapshots.append((prior, expected, value, expected + i % 3 + 1))
                expected += i % 3 + 1
                assert state._mem_ptr_now == address
            assert jt.core.graph_launch_count() > launch_count
        finally:
            captured.invalidate()

        for prior, old, value, new in snapshots:
            np.testing.assert_array_equal(prior.numpy(), np.full(8, old))
            np.testing.assert_array_equal(value.numpy(), np.full(8, new))
        tail = np.full(8, 17.0) if physical_window else np.empty(0)
        np.testing.assert_array_equal(backing.numpy(), np.concatenate((np.full(8, expected), tail)))
