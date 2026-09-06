"""Exercise the real ACL providers with a recording SDK, without core/JIT import."""

import ast
from collections.abc import Sequence
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "backends/acl/kernels"


class _Tensor:
    def __init__(self, shape=(2, 4), dtype="float32"):
        self.shape = tuple(shape)
        self.ndim = len(self.shape)
        self.dtype = dtype

    def numel(self):
        result = 1
        for size in self.shape:
            result *= size
        return result

    def is_stop_grad(self):
        return True

    def reshape(self, shape):
        return _Tensor(shape, self.dtype)

    def assign(self, other):
        raise AssertionError("provider must not own setitem writeback")


@pytest.fixture
def providers(monkeypatch):
    calls = []

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    for name in (
        "jittor",
        "jittor._runtime",
        "jittor.backends",
        "jittor.backends.acl",
        "jittor.backends.acl.kernels",
        "jittor.backends.acl.kernels.ops",
    ):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    native = sys.modules["jittor"]
    native.Var = _Tensor
    native.core = SimpleNamespace(Var=_Tensor, dispatch_context=lambda tensors: ("acl_legacy", 0))
    native.flags = SimpleNamespace(no_grad=1)
    native.runtime = SimpleNamespace(use_cuda=1)
    native.ops = SimpleNamespace(arg_reduce=lambda *args: ("indices", "values"))
    dispatch = load("jittor._runtime.dispatch", ROOT / "python/jittor/_runtime/dispatch.py")

    def primitive(name):
        def construct(*args, **kwargs):
            calls.append((name, "construct", args, kwargs))

            def execute(*inputs, **options):
                calls.append((name, "execute", inputs, options))
                return _Tensor()

            return execute

        return construct

    for path in (
        KERNELS / "tensor.py",
        KERNELS / "normalization.py",
        KERNELS / "neural.py",
        KERNELS / "install.py",
    ):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.ImportFrom) or not (node.module or "").startswith("ops."):
                continue
            module_name = "jittor.backends.acl.kernels." + node.module
            module = sys.modules.get(module_name)
            if module is None:
                module = ModuleType(module_name)
                monkeypatch.setitem(sys.modules, module_name, module)
            for alias in node.names:
                if alias.name == "ACL_FLOAT_DTYPES":
                    setattr(module, alias.name, ("float16", "bfloat16", "float32"))
                else:
                    setattr(module, alias.name, primitive(alias.name))
    modules = {}
    for name in ("tensor", "normalization", "neural", "install"):
        module = load("jittor.backends.acl.kernels." + name, KERNELS / (name + ".py"))
        setattr(sys.modules["jittor.backends.acl.kernels"], name, module)
        modules[name] = module
    return SimpleNamespace(native=native, dispatch=dispatch, calls=calls, **modules)


def test_acl_install_publishes_real_owners_idempotently_without_facade_writes(providers):
    before = vars(providers.native).copy()
    providers.install.install()
    first = providers.dispatch._kernels.copy()
    providers.install.install()
    assert providers.dispatch._kernels == first
    assert vars(providers.native) == before
    assert providers.calls == []
    assert len(providers.install.KERNELS) == 44
    for operation, implementation in providers.install.KERNELS:
        assert providers.dispatch.registered_kernel(operation, "acl_legacy") is implementation
        assert implementation.__module__.startswith("jittor.backends.acl.kernels.") or (
            operation == "nn.scaled_dot_product_attention"
        )


def test_acl_rejection_does_not_reenter_native_dispatch(providers):
    x = _Tensor()
    assert providers.tensor.getitem_acl(x, 0, return_x=True) is None
    assert providers.tensor.setitem_acl(x, 0, x, reduce="add") is None
    assert providers.tensor.arg_reduce_acl(_Tensor(dtype="int64"), "max", 0) is None
    assert providers.tensor._roll_acl(_Tensor(dtype="float64"), 1) is None
    assert providers.tensor._split_acl(x, 0) is None
    assert providers.neural.resize_acl(x, (2, 2), mode="bilinear") is None
    assert providers.neural._silu_acl(_Tensor(dtype="float16")) is None
    assert providers.neural.softmax_acl(x, zero_all_neg_inf=True) is None
    assert providers.normalization.layer_norm_acl(x, (4,), 1, 0, 1e-5) is None
    assert providers.calls == []
    providers.install.install()
    assert providers.dispatch.select_kernel("nn.softmax", x, dim=-1) is providers.neural.softmax_acl
    assert providers.dispatch.select_kernel("nn.softmax", x, dim=-1, zero_all_neg_inf=True) is None


def test_acl_setitem_returns_result_and_arg_reduce_uses_unwrapped_kernel(providers):
    x = _Tensor()
    result = providers.tensor.setitem_acl(x, 0, _Tensor((4,)))
    assert result is not x
    assert providers.calls[-1][0:2] == ("SetItemACL", "execute")
    providers.calls.clear()
    providers.tensor.arg_reduce_acl(x, "max", 1)
    assert providers.calls[0][0:2] == ("ArgReduceACL", "construct")
    assert providers.calls[0][2] == (providers.native.ops.arg_reduce,)


def test_acl_registered_call_preserves_operator_error(providers, monkeypatch):
    providers.install.install()

    def failed(*args):
        raise RuntimeError("recorded ACL allocation failure")

    monkeypatch.setattr(providers.tensor, "TriuACL", failed)
    with pytest.raises(RuntimeError, match="recorded ACL allocation failure"):
        providers.dispatch.try_dispatch("tensor.triu", _Tensor(), 0)


def test_acl_matrix_transpose_flags_reach_the_shared_provider(providers):
    x = _Tensor((2, 4, 4))
    providers.neural.bmm_acl(x, x, trans_a=False, trans_b=True)
    assert providers.calls[0] == ("BmmACL", "construct", (True,), {})
    providers.calls.clear()
    providers.neural.matmul_acl(x, x, trans_b=True)
    assert providers.calls[0] == ("MatmulACL", "construct", (True,), {})


@pytest.mark.parametrize(
    "dtype,dilation,return_indices,op",
    [
        ("float32", None, False, "minimum"),
        ("float64", None, False, "maximum"),
        ("int32", None, False, "mean"),
        ("float32", 2, False, "maximum"),
        ("float32", (1, 2), False, "maximum"),
        ("float32", None, True, "mean"),
    ],
)
def test_acl_pool_declines_unsupported_semantics(providers, dtype, dilation, return_indices, op):
    x = _Tensor((1, 2, 5, 5), dtype)
    result = providers.neural.pool_acl(x, 3, 2, 1, dilation, return_indices, True, True, op)
    assert result is None
    assert providers.calls == []


@pytest.mark.parametrize(
    "size,kernel,stride,padding,ceil_mode,expected",
    [
        (3, 2, 2, 1, True, 2),
        (3, 1, 3, 0, True, 1),
        (5, 3, 2, 1, False, 3),
        (4, 3, 2, 1, True, 3),
    ],
)
@pytest.mark.parametrize("op", ["maximum", "mean"])
def test_acl_pool_uses_canonical_output_geometry(
    providers, monkeypatch, size, kernel, stride, padding, ceil_mode, expected, op
):
    geometry_source = (ROOT / "python/jittor/nn/functional/pooling.py").read_text(encoding="utf-8")
    geometry = next(
        node
        for node in ast.parse(geometry_source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_pool_output_size"
    )
    geometry_namespace = {}
    exec(
        compile(ast.get_source_segment(geometry_source, geometry), "<pool_geometry>", "exec"),
        geometry_namespace,
    )
    actual_geometry = geometry_namespace["_pool_output_size"]
    geometry_calls = []

    def record_geometry(*args):
        geometry_calls.append(args)
        return actual_geometry(*args)

    for name in ("jittor.nn", "jittor.nn.functional", "jittor.nn.functional.pooling"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    sys.modules["jittor.nn.functional.pooling"]._pool_output_size = record_geometry

    class Function:
        def __call__(self, *args):
            return self.execute(*args)

    providers.native.Function = Function
    launches = []

    def record_pool(name, inputs, output_dtypes, output_shapes, attr_code):
        launches.append((name, output_shapes, attr_code))
        return [_Tensor(shape, dtype) for shape, dtype in zip(output_shapes, output_dtypes)]

    pool_source = (KERNELS / "ops/pool_op.py").read_text(encoding="utf-8")
    pool_class = next(
        node
        for node in ast.parse(pool_source).body
        if isinstance(node, ast.ClassDef) and node.name == "PoolACL"
    )
    namespace = {"jt": providers.native, "pool_cmd": record_pool}
    exec(
        compile(ast.get_source_segment(pool_source, pool_class), "<actual_pool_acl>", "exec"),
        namespace,
    )
    monkeypatch.setattr(providers.neural, "PoolACL", namespace["PoolACL"])
    value = _Tensor((1, 2, size, size))
    result = providers.neural.pool_acl(
        value, kernel, stride, padding, (1, 1), False, ceil_mode, False, op
    )
    assert result.shape == (1, 2, expected, expected)
    assert geometry_calls == [(size, kernel, stride, padding, ceil_mode)] * 2
    assert launches[0][0] == ("Maxpool" if op == "maximum" else "Avgpool")
    assert "attr->countIncludePad = false" in launches[0][2]


@pytest.mark.parametrize("entry", ["provider", "public"])
@pytest.mark.parametrize(
    "shape,dims,expected,axes,inverse",
    [
        ((2, 3, 4), (), (4, 3, 2), (2, 1, 0), (2, 1, 0)),
        ((2, 3, 4), ((2, 0, 1),), (4, 2, 3), (2, 0, 1), (1, 2, 0)),
        ((2, 3, 4), ([2, 0, 1],), (4, 2, 3), (2, 0, 1), (1, 2, 0)),
        ((2, 3, 4), (2, 0, 1), (4, 2, 3), (2, 0, 1), (1, 2, 0)),
        ((2, 3, 4), (0, 2), (4, 3, 2), (2, 1, 0), (2, 1, 0)),
        ((2, 3, 4), (-1, -2), (2, 4, 3), (0, 2, 1), (0, 2, 1)),
        ((2, 3, 4), ((-1, 0, 1),), (4, 2, 3), (-1, 0, 1), (1, 2, 0)),
        ((2, 3), ((0, 1),), (2, 3), (0, 1), (0, 1)),
        ((2, 3), (0, 1), (3, 2), (1, 0), (1, 0)),
        ((2, 3), (-1, -2), (3, 2), (1, 0), (1, 0)),
    ],
)
def test_acl_transpose_preserves_argument_forms_with_real_shape_builder(
    providers, monkeypatch, entry, shape, dims, expected, axes, inverse
):
    launches = []

    def record_transpose(name, inputs, output_dtypes, output_shapes, attr_code, cuda_grad_src):
        launches.append((name, output_shapes, attr_code, cuda_grad_src))
        return [
            _Tensor(output_shape, dtype)
            for output_shape, dtype in zip(output_shapes, output_dtypes)
        ]

    source = (KERNELS / "ops/transpose_op.py").read_text(encoding="utf-8")
    implementation = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "TransPoseACL"
    )
    namespace = {"Sequence": Sequence, "transpose_cmd": record_transpose}
    exec(
        compile(ast.get_source_segment(source, implementation), "<actual_transpose_acl>", "exec"),
        namespace,
    )
    monkeypatch.setattr(providers.tensor, "TransPoseACL", namespace["TransPoseACL"])
    providers.install.install()
    call = providers.tensor.transpose_acl
    if entry == "public":
        source = (ROOT / "python/jittor/_runtime/core_api.py").read_text(encoding="utf-8")
        adapter = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "transpose"
        )

        def reject_fallback(*args):
            raise AssertionError("expected the registered ACL transpose")

        namespace = {
            "Sequence": Sequence,
            "NanoVector": tuple,
            "Var": _Tensor,
            "_try_dispatch": providers.dispatch.try_dispatch,
            "origin_transpose": reject_fallback,
        }
        exec(
            compile(ast.get_source_segment(source, adapter), "<actual_transpose_entry>", "exec"),
            namespace,
        )
        call = namespace["transpose"]
    result = call(_Tensor(shape), *dims)
    assert result.shape == expected
    assert len(launches) == 1
    name, output_shapes, forward_source, backward_source = launches[0]
    assert name == "Transpose"
    assert output_shapes == [list(expected)]
    assert "attr->axes = { " + ", ".join(map(str, axes)) + " };" in forward_source
    assert "attr->axes = { " + ", ".join(map(str, inverse)) + " };" in backward_source[0]


def test_acl_compiler_no_longer_contains_python_replacement_installer():
    source = (ROOT / "python/jittor/extern/acl/acl_compiler.py").read_text(encoding="utf-8")
    assert "def change_function" not in source
    assert "def warp" not in source
    assert "jt.nn." not in source
    for name in ("tensor", "neural", "normalization"):
        tree = ast.parse((KERNELS / (name + ".py")).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                assert not any(
                    isinstance(child, (ast.FunctionDef, ast.ClassDef)) for child in node.body
                ), node.name
