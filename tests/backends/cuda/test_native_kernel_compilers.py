"""Real CUDA kernels, native Jittor graph execution, no Torch shim required."""
import numpy as np
import pytest
import jittor as jt
from jittor.kernels import TensorSpec, triton_call, tilelang_call

pytestmark = pytest.mark.skipif(not jt.has_cuda, reason="CUDA required")


@pytest.fixture(autouse=True)
def cuda():
    with jt.flag_scope(use_cuda=1):
        yield


def add_kernel():
    triton = pytest.importorskip("triton")
    import triton.language as tl
    @triton.jit
    def add(X, Y, O, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        tl.store(O + i, tl.load(X + i, i < N, 0) + tl.load(Y + i, i < N, 0), i < N)
    return add


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("lazy", [0, 1])
def test_triton_pipeline_and_gradient(dtype, lazy):
    kernel = add_kernel()
    class Add(jt.Function):
        def execute(self, x, y):
            return triton_call(kernel, x, y, TensorSpec(x.shape, dtype),
                               N=x.numel(), BLOCK=128, grid=lambda m: ((m["N"] + 127) // 128,))
        def grad(self, dout):
            return dout, dout
    with jt.flag_scope(lazy_execution=lazy):
        for shift in (0, 2, -3):
            data = np.arange(257, dtype=np.float32) / 256 + shift
            x = jt.array(data).cast(dtype)
            y = jt.full((257,), 0.5, dtype=dtype)
            z = Add.apply(x * 2, y)
            assert z.location() in ("none", "device")
            np.testing.assert_allclose(z.float32().numpy(), data * 2 + 0.5, atol=0.04, rtol=0.01)
            dx, dy = jt.grad(z.float32().sum(), [x, y])
            np.testing.assert_array_equal(dx.float32().numpy(), np.full(257, 2))
            np.testing.assert_array_equal(dy.float32().numpy(), np.ones(257))
            assert z.location() == "device"


def test_triton_strides_and_multiple_outputs():
    triton = pytest.importorskip("triton")
    import triton.language as tl
    @triton.jit
    def split(X, A, B, M: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr):
        i = tl.program_id(0) * 128 + tl.arange(0, 128)
        x = tl.load(X + (i // N) * S0 + (i % N) * S1, i < M * N, 0)
        tl.store(A + i, x + 1, i < M * N)
        tl.store(B + i, x * 2, i < M * N)
    data = np.arange(15, dtype=np.float32).reshape(3, 5)
    x = jt.array(data).transpose(1, 0)
    a, b = triton_call(split, x, TensorSpec(x.shape), TensorSpec(x.shape),
                       M=5, N=3, S0=x._storage_strides()[0], S1=x._storage_strides()[1], grid=(1,))
    np.testing.assert_array_equal(a.numpy(), data.T + 1)
    np.testing.assert_array_equal(b.numpy(), data.T * 2)


def test_triton_matmul():
    triton = pytest.importorskip("triton")
    import triton.language as tl
    @triton.jit
    def matmul(A, B, C):
        m = tl.arange(0, 16)
        k = tl.arange(0, 32)
        n = tl.arange(0, 16)
        a = tl.load(A + m[:, None] * 32 + k[None, :])
        b = tl.load(B + k[:, None] * 16 + n[None, :])
        c = tl.dot(a, b)
        tl.store(C + m[:, None] * 16 + n[None, :], c)
    rng = np.random.default_rng(14)
    a = rng.normal(size=(16, 32)).astype(np.float16)
    b = rng.normal(size=(32, 16)).astype(np.float16)
    c = triton_call(matmul, jt.array(a), jt.array(b), TensorSpec((16, 16)), grid=(1,))
    np.testing.assert_allclose(c.numpy(), a.astype(np.float32) @ b.astype(np.float32), atol=1e-4, rtol=1e-4)


def make_tilelang_add(n):
    tilelang = pytest.importorskip("tilelang")
    import tilelang.language as T
    @T.prim_func
    def add(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, 128), threads=128) as block:
            for i in T.Parallel(128):
                index = block * 128 + i
                if index < n:
                    C[index] = A[index] + B[index]
    return tilelang.compile(add, out_idx=[], target="cuda", execution_backend="tvm_ffi")


@pytest.mark.parametrize("lazy", [0, 1])
def test_tilelang_pipeline_and_gradient(lazy):
    kernel = make_tilelang_add(257)
    class Add(jt.Function):
        def execute(self, x, y):
            return tilelang_call(kernel, x, y, TensorSpec(x.shape))
        def grad(self, dout):
            return dout, dout
    with jt.flag_scope(lazy_execution=lazy):
        for shift in (0, 1, 5):
            data = np.arange(257, dtype=np.float32) + shift
            x = jt.array(data)
            y = jt.full((257,), 0.5)
            z = Add.apply(x * 3, y)
            np.testing.assert_array_equal((z + 1).numpy(), data * 3 + 1.5)
            dx, dy = jt.grad(z.sum(), [x, y])
            np.testing.assert_array_equal(dx.numpy(), np.full(257, 3))
            np.testing.assert_array_equal(dy.numpy(), np.ones(257))
            assert z.location() == "device"


def test_tilelang_matmul():
    tilelang = pytest.importorskip("tilelang")
    import tilelang.language as T
    @T.prim_func
    def matmul(A: T.Tensor((16, 32), "float16"), B: T.Tensor((32, 16), "float16"),
               C: T.Tensor((16, 16), "float32")):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((16, 32), "float16")
            b = T.alloc_shared((32, 16), "float16")
            c = T.alloc_fragment((16, 16), "float32")
            T.copy(A, a)
            T.copy(B, b)
            T.clear(c)
            T.gemm(a, b, c)
            T.copy(c, C)
    kernel = tilelang.compile(matmul, out_idx=[], target="cuda", execution_backend="tvm_ffi")
    rng = np.random.default_rng(15)
    a = rng.normal(size=(16, 32)).astype(np.float16)
    b = rng.normal(size=(32, 16)).astype(np.float16)
    c = tilelang_call(kernel, jt.array(a), jt.array(b), TensorSpec((16, 16)))
    np.testing.assert_allclose(c.numpy(), a.astype(np.float32) @ b.astype(np.float32), atol=1e-4, rtol=1e-4)


def test_triton_reduction_and_kernel_backward():
    triton = pytest.importorskip("triton")
    import triton.language as tl
    @triton.jit
    def softmax(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        x = tl.load(X + i, i < N, -float("inf"))
        exp = tl.exp(x - tl.max(x, axis=0))
        tl.store(Y + i, exp / tl.sum(exp, axis=0), i < N)
    @triton.jit
    def backward(Y, DY, DX, N: tl.constexpr, BLOCK: tl.constexpr):
        i = tl.arange(0, BLOCK)
        y = tl.load(Y + i, i < N, 0)
        dy = tl.load(DY + i, i < N, 0)
        tl.store(DX + i, y * (dy - tl.sum(y * dy, axis=0)), i < N)
    class Softmax(jt.Function):
        def execute(self, x):
            self.y = triton_call(softmax, x, TensorSpec(x.shape), N=97, BLOCK=128, grid=(1,))
            return self.y
        def grad(self, dout):
            return triton_call(backward, self.y, dout, TensorSpec(dout.shape), N=97, BLOCK=128, grid=(1,))
    data = np.linspace(-3, 3, 97, dtype=np.float32)
    weights = np.linspace(-1, 2, 97, dtype=np.float32)
    expected = np.exp(data - data.max())
    expected /= expected.sum()
    x = jt.array(data)
    y = Softmax.apply(x)
    dx = jt.grad((y * jt.array(weights)).sum(), x)
    np.testing.assert_allclose(y.numpy(), expected, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(dx.numpy(), expected * (weights - (expected * weights).sum()), atol=1e-7, rtol=1e-5)


def test_tilelang_kernel_backward():
    kernel = make_tilelang_add(257)
    class Double(jt.Function):
        def execute(self, x):
            return tilelang_call(kernel, x, x, TensorSpec(x.shape))
        def grad(self, dout):
            return tilelang_call(kernel, dout, dout, TensorSpec(dout.shape))
    x = jt.array(np.arange(257, dtype=np.float32) / 256)
    y = Double.apply(x)
    np.testing.assert_allclose(jt.grad((y * y).sum(), x).numpy(), np.arange(257) / 32)


@pytest.mark.parametrize("automatic", [False, True])
def test_replay_does_not_omit_kernel(automatic):
    kernel = add_kernel()
    class Model(jt.nn.Module):
        def execute(self, x):
            return triton_call(kernel, x, x, TensorSpec(x.shape), N=257, BLOCK=128, grid=(3,)) + 1
    with jt.flag_scope(auto_graph_replay=int(automatic), async_execution=1, auto_flush_ops=23), jt.no_grad():
        model = Model()
        run = model if automatic else jt.graph_replay(model)
        for value in (1, 3, 5, 7):
            x = jt.full((257,), value, dtype="float32")
            np.testing.assert_array_equal(run(x).numpy(), np.full(257, 2 * value + 1))


@pytest.mark.parametrize("asynchronous", [0, 1])
def test_producers_and_consumers_share_the_execution_stream(asynchronous):
    kernel = add_kernel()
    with jt.flag_scope(async_execution=asynchronous, auto_flush_ops=8,
                       async_min_caller_us=0), jt.no_grad():
        x = jt.full((1025,), 0.125, dtype="float32")
        expected = np.full(1025, 0.125, dtype=np.float32)
        for _ in range(32):
            x = x * 0.75 + 0.25
            expected = expected * 0.75 + 0.25
        z = triton_call(kernel, x, x, TensorSpec(x.shape), N=1025, BLOCK=128, grid=(9,))
        np.testing.assert_allclose((z * 3).numpy(), expected * 6, atol=2e-6, rtol=2e-6)


def test_triton_does_not_require_torch():
    pytest.importorskip("triton")
    from _helpers.child_process import run_child_script
    result = run_child_script('''
import sys
import importlib.abc
class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("Torch intentionally unavailable", name=fullname)
sys.meta_path.insert(0, NoTorch())
import numpy as np
import jittor as jt
import triton
import triton.language as tl
from jittor.kernels import TensorSpec, triton_call
@triton.jit
def double(X, Y):
    i = tl.arange(0, 32)
    tl.store(Y + i, tl.load(X + i) * 2)
with jt.flag_scope(use_cuda=1):
    x = jt.array(np.arange(32, dtype=np.float32))
    y = triton_call(double, x, TensorSpec(x.shape), grid=(1,))
    np.testing.assert_array_equal(y.numpy(), np.arange(32) * 2)
assert "torch" not in sys.modules
print("native Triton without Torch passed")
''', timeout=120, text=True, merge_stderr=True, without_torch_mode=True)
    assert result.returncode == 0, result.stdout
    assert "native Triton without Torch passed" in result.stdout
