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
