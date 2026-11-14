"""The native callback owns graph dependencies, not just a naked pointer launch."""
import ctypes
import gc

import numpy as np
import pytest
import jittor as jt
from jittor.kernels import TensorSpec, call


def writable(buffer):
    count = int(np.prod(buffer.shape))
    array = np.ctypeslib.as_array((ctypes.c_float * count).from_address(buffer.data_ptr()))
    return array.reshape(buffer.shape)


def add_one(context):
    assert context.device == ("cpu", 0)
    x = np.from_dlpack(context.inputs[0])
    np.copyto(writable(context.outputs[0]), x + 1)


def test_lazy_pipeline_and_multiple_outputs():
    seen = []
    def launch(ctx):
        seen.append(True)
        x = np.from_dlpack(ctx.inputs[0])
        np.copyto(writable(ctx.outputs[0]), x + 1)
        np.copyto(writable(ctx.outputs[1]), x * 2)
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0):
        x = jt.array(np.arange(12, dtype=np.float32).reshape(3, 4))
        a, b = call(launch, x * 3, outputs=[TensorSpec(x.shape), TensorSpec(x.shape)], backend="cpu")
        assert not seen
        del x
        gc.collect()
        np.testing.assert_array_equal((a + b).numpy(), np.arange(12).reshape(3, 4) * 9 + 1)
        assert seen == [True]


@pytest.mark.parametrize("lazy", [0, 1])
def test_strided_input_and_repeated_calls(lazy):
    with jt.flag_scope(use_cuda=0, lazy_execution=lazy):
        for value in range(3):
            x = jt.array(np.arange(12, dtype=np.float32).reshape(3, 4)) + value
            x = x.transpose(1, 0)
            y = call(add_one, x, outputs=TensorSpec(x.shape), backend="cpu")
            np.testing.assert_array_equal(y.numpy(), (np.arange(12).reshape(3, 4) + value).T + 1)


def test_explicit_autograd_and_missing_backward():
    class Double(jt.Function):
        def execute(self, x):
            def launch(ctx):
                np.copyto(writable(ctx.outputs[0]), np.from_dlpack(ctx.inputs[0]) * 2)
            return call(launch, x, outputs=TensorSpec(x.shape), backend="cpu")
        def grad(self, dout):
            return dout * 2
    with jt.flag_scope(use_cuda=0):
        x = jt.array(np.arange(6, dtype=np.float32))
        y = Double.apply(x * 3)
        np.testing.assert_array_equal(jt.grad((y * y).sum(), x).numpy(), np.arange(6) * 72)
        z = call(add_one, x, outputs=TensorSpec(x.shape), backend="cpu")
        with pytest.raises(RuntimeError, match="no implicit derivative"):
            jt.grad(z.sum(), x)


def test_failures_are_not_hidden():
    def fail(ctx):
        raise ValueError("kernel launch sentinel")
    with jt.flag_scope(use_cuda=0):
        x = jt.ones((4,))
        with pytest.raises(Exception, match="kernel launch sentinel|Python error"):
            call(fail, x, outputs=TensorSpec(x.shape), backend="cpu").numpy()
        with pytest.raises(RuntimeError, match="placement.*backend"):
            call(add_one, x, outputs=TensorSpec(x.shape), backend="cuda").numpy()
    with pytest.raises(ValueError, match="nonnegative"):
        TensorSpec((-1,))
    with pytest.raises(ValueError, match="cpu and cuda"):
        call(add_one, x, outputs=TensorSpec(x.shape), backend="npu")


def test_tilelang_rejects_non_cuda_target_before_export():
    from types import SimpleNamespace
    from jittor.kernels.tilelang import executable_for
    kernel = SimpleNamespace(execution_backend="tvm_ffi", out_idx=[],
                             target=SimpleNamespace(kind=SimpleNamespace(name="llvm")))
    with pytest.raises(ValueError, match="CUDA compilation target"):
        executable_for(kernel)
