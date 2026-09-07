"""Native NN ownership reaches the real Python registry without ACL hardware."""

import ast
from contextlib import nullcontext
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


class Tensor:
    def __init__(self, shape, dtype="float32"):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = dtype

    def dim(self):
        return self.ndim

    def to(self, dtype):
        assert dtype == self.dtype
        return self


class Module:
    pass


def _definitions(relative, names, namespace):
    tree = ast.parse((ROOT / "python/jittor" / relative).read_text())
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in tree.body} == set(names)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    exec(compile(tree, relative, "exec"), namespace)
    return namespace


@pytest.fixture
def routing(monkeypatch):
    native = SimpleNamespace(
        Var=Tensor, Module=Module, nn=SimpleNamespace(),
        core=SimpleNamespace(Var=Tensor, dispatch_context=lambda tensors: ("acl", 0)),
        flag_scope=lambda **kwargs: nullcontext(),
        flags=SimpleNamespace(amp_reg=0),
        amp_flags=SimpleNamespace(keep_reduce=1, reduce16_no_fp32_acc=2),
    )
    monkeypatch.setitem(sys.modules, "jittor", native)
    path = ROOT / "python/jittor/_runtime/dispatch.py"
    spec = importlib.util.spec_from_file_location("_acl_native_routing_table", path)
    dispatch = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, dispatch)
    spec.loader.exec_module(dispatch)
    seen = []
    marker = object()

    def register(key):
        def implementation(*args, **kwargs):
            seen.append((key, args, kwargs))
            return marker
        dispatch.register_kernel(key, "acl", implementation)

    def ignored(*args):
        raise ValueError("non-default inplace rejected before dispatch")

    namespace = {
        "jt": native, "Module": Module,
        "try_dispatch": dispatch.try_dispatch, "select_kernel": dispatch.select_kernel,
        "_pair": lambda value: tuple(value) if isinstance(value, (tuple, list)) else (value, value),
        "_arg_policy": SimpleNamespace(ignored=ignored),
        "_INPLACE_CONSEQUENCE": "not performed in place",
    }
    return SimpleNamespace(namespace=namespace, jt=native, seen=seen,
                           marker=marker, register=register, dispatch=dispatch)


@pytest.mark.parametrize("name,key,kwargs", [
    ("relu", "nn.relu", {}),
    ("leaky_relu", "nn.leaky_relu", {"negative_slope": 0.2}),
    ("silu", "nn.silu", {}),
])
def test_activation_function_and_existing_class_share_registration(routing, name, key, kwargs):
    ns = _definitions("nn/functional/activation.py", [name], routing.namespace)
    routing.register(key)
    function = ns[name]
    setattr(routing.jt.nn, name, function)
    classes = {"relu": "ReLU", "leaky_relu": "LeakyReLU", "silu": "SiLU"}
    class_name = classes[name]
    _definitions("nn/modules/activation.py", ["_FunctionModule", class_name], ns)
    original_class = ns[class_name]
    instance = original_class(**kwargs)
    x = Tensor((2, 4))
    assert instance.execute(x) is routing.marker
    assert type(instance) is original_class
    assert routing.seen[-1][1][0] is x
    if name == "leaky_relu":
        assert routing.seen[-1][2]["scale"] == 0.2
    with pytest.raises(ValueError, match="inplace"):
        function(x, inplace=True)
    assert len(routing.seen) == 1


def test_resize_validates_geometry_and_mode_before_the_registered_kernel(routing):
    function = _definitions("nn/functional/interpolation.py", ["resize"], routing.namespace)["resize"]
    routing.register("nn.resize")
    x = Tensor((1, 3, 4, 5))
    assert function(x, (8, 10)) is routing.marker
    assert routing.seen[0][1] == (x, (8, 10), "nearest", False, False)
    for tensor, size, mode in ((Tensor((3, 4, 5)), (8, 10), "nearest"),
                               (x, (0, 10), "nearest"), (x, (8, 10), "bad")):
        with pytest.raises((ValueError, RuntimeError)):
            function(tensor, size, mode)
    assert len(routing.seen) == 1


def test_pool_keeps_the_native_module_and_routes_after_validation(routing):
    ns = _definitions("pool/core_2d.py", ["Pool"], routing.namespace)
    routing.register("nn.pool2d")
    original_class = ns["Pool"]
    pool = original_class((2, 3), stride=(2, 1), return_indices=True)
    x = Tensor((1, 2, 8, 10))
    assert pool.execute(x) is routing.marker
    assert type(pool) is original_class
    assert routing.seen[0][1] == (x, (2, 3), (2, 1), (0, 0), None, True,
                                  False, True, "maximum")
    with pytest.raises(RuntimeError):
        original_class(0)
    with pytest.raises(RuntimeError):
        pool.execute(Tensor((1, 2, 1, 1)))
    assert len(routing.seen) == 1


def test_average_pooling_uses_the_same_registry_without_duplicating_math(routing):
    ns = _definitions("nn/functional/pooling.py", ["_pool_output_size", "_avg_pool_nd", "avg_pool2d"],
                      routing.namespace)
    routing.register("nn.pool2d")
    x = Tensor((1, 2, 8, 10))
    assert ns["avg_pool2d"](x, 2, count_include_pad=False) is routing.marker
    assert routing.seen[0][1] == (x, (2, 2), (2, 2), (0, 0), None, False,
                                  False, False, "mean")
    with pytest.raises(RuntimeError, match="kernel_size"):
        ns["avg_pool2d"](x, 0)
    assert len(routing.seen) == 1


def test_conv_module_preserves_parameter_objects_and_public_geometry_checks(routing):
    ns = _definitions("nn/functional/convolution.py", ["_check_conv2d_output_size", "conv2d"],
                      routing.namespace)
    routing.jt.nn.conv2d = ns["conv2d"]
    _definitions("nn/modules/convolution.py", ["Conv"], ns)
    routing.register("conv2d")
    cls = ns["Conv"]
    instance = object.__new__(cls)
    instance.weight, instance.bias = Tensor((4, 2, 3, 3)), Tensor((4,))
    instance.stride, instance.padding, instance.dilation, instance.groups = (1, 1), (0, 0), (1, 1), 1
    parameters = instance.weight, instance.bias
    x = Tensor((1, 2, 8, 10))
    assert instance.execute(x) is routing.marker
    assert (instance.weight, instance.bias) == parameters
    assert type(instance) is cls
    assert routing.seen[-1][1][1:3] == parameters
    with pytest.raises(ValueError, match="4-D weight"):
        ns["conv2d"](x, Tensor((4, 2, 3)))
    with pytest.raises(ValueError, match="positive"):
        ns["conv2d"](x, parameters[0], stride=0)
    assert len(routing.seen) == 1


def test_layer_norm_keeps_affine_parameters_and_validates_before_dispatch(routing):
    routing.register("nn.layer_norm.training")
    ns = routing.namespace
    ns["_layer_norm_cuda"] = lambda *args: routing.dispatch.try_dispatch("nn.layer_norm.training", *args)
    ns["_layer_norm_no_grad_cuda"] = lambda *args: None
    _definitions("nn/functional/normalization.py", ["layer_norm"], ns)
    routing.jt.nn.layer_norm = ns["layer_norm"]
    _definitions("nn/modules/normalization.py", ["LayerNorm"], ns)
    cls = ns["LayerNorm"]
    layer = object.__new__(cls)
    layer.normalized_shape, layer.eps, layer.elementwise_affine = (4,), 1e-5, True
    layer.weight, layer.bias = Tensor((4,)), Tensor((4,))
    parameters = layer.weight, layer.bias
    x = Tensor((2, 4))
    assert layer.execute(x) is routing.marker
    assert type(layer) is cls
    assert (layer.weight, layer.bias) == parameters
    with pytest.raises(ValueError, match="normalized_shape"):
        ns["layer_norm"](x, (3,), *parameters)
    assert len(routing.seen) == 1


def test_matmul_bmm_and_transpose_reuse_existing_keys(routing):
    names = ["_check_matmul_shapes", "_transpose_base_last2", "_matmul_2d_cublas",
             "matmul", "matmul_transpose", "bmm", "bmm_transpose"]
    ns = _definitions("nn/functional/matrix.py", names, routing.namespace)
    routing.jt.nn.matmul = ns["matmul"]
    routing.register("matmul")
    routing.register("batched_matmul")
    for name, a, b, key in (
        ("matmul", Tensor((2, 3)), Tensor((3, 4)), "matmul"),
        ("matmul_transpose", Tensor((2, 3)), Tensor((4, 3)), "matmul"),
        ("bmm", Tensor((2, 3, 4)), Tensor((2, 4, 5)), "batched_matmul"),
        ("bmm_transpose", Tensor((2, 3, 4)), Tensor((2, 5, 4)), "batched_matmul"),
    ):
        assert ns[name](a, b) is routing.marker
        assert routing.seen[-1][0] == key
    with pytest.raises(AssertionError, match="dimension not match"):
        ns["matmul"](Tensor((2, 3)), Tensor((4, 5)))
    assert len(routing.seen) == 4


def test_dropout_keeps_the_native_probability_and_training_owner(routing):
    ns = _definitions("nn/functional/dropout.py", ["_check_probability", "dropout"], routing.namespace)
    routing.jt.nn.dropout = ns["dropout"]
    _definitions("nn/modules/dropout.py", ["Dropout"], ns)
    cls = ns["Dropout"]
    layer = cls(p=0.3, is_train=False)
    x = Tensor((2, 3))
    assert layer.execute(x) is x
    assert type(layer) is cls
    assert layer.p == 0.3 and layer.is_train is False
    with pytest.raises(AssertionError, match="probability"):
        cls(p=2)


def test_softmax_uses_existing_key_after_public_axis_validation(routing, monkeypatch):
    backend = SimpleNamespace(_softmax_v1=lambda x, **kwargs:
                              routing.dispatch.try_dispatch("nn.softmax", x, **kwargs))
    monkeypatch.setitem(sys.modules, "jittor.backends.cuda.kernels.nn",
                        SimpleNamespace(softmax_cuda=backend))
    ns = _definitions("nn/functional/softmax.py", ["_get_softmax_dim", "softmax"],
                      routing.namespace)
    routing.register("nn.softmax")
    x = Tensor((2, 3))
    assert ns["softmax"](x, log=True) is routing.marker
    assert routing.seen[0][2] == {"dim": 1, "log": True}
    with pytest.raises(IndexError, match="dimension"):
        ns["softmax"](x, dim=2)
    assert len(routing.seen) == 1


def test_sdpa_validates_dtype_and_dropout_before_registered_attention(routing):
    function = _definitions("nn/functional/attention.py", ["scaled_dot_product_attention"],
                            routing.namespace)["scaled_dot_product_attention"]
    routing.register("nn.scaled_dot_product_attention")
    query, key, value = Tensor((1, 2, 4, 8)), Tensor((1, 2, 4, 8)), Tensor((1, 2, 4, 8))
    assert function(query, key, value, is_causal=True) is routing.marker
    assert routing.seen[0][2] == {"attn_mask": None, "dropout_p": 0.0,
                                  "is_causal": True, "scale": None}
    with pytest.raises(RuntimeError, match="same dtype"):
        function(query, Tensor(key.shape, "float16"), value)
    with pytest.raises(ValueError, match="probability"):
        function(query, key, value, dropout_p=2)
    assert len(routing.seen) == 1


def test_rotary_embedding_has_a_stable_native_entry_and_rejects_missing_tables(routing):
    function = _definitions("nn/serving_ops.py", ["rotary_emb"], routing.namespace)["rotary_emb"]
    routing.register("nn.rotary_emb")
    xq, xk = Tensor((1, 2, 3, 64)), Tensor((1, 1, 3, 64))
    sin, cos = Tensor((1, 1, 3, 64)), Tensor((1, 1, 3, 64))
    assert function(xq, xk, freq_sin=sin, freq_cos=cos) is routing.marker
    assert routing.seen[0][1] == (xq, xk, None, sin, cos)
    with pytest.raises(ValueError, match="requires"):
        function(xq, xk)
    assert len(routing.seen) == 1


def test_rotary_generic_math_is_available_without_an_acl_registration(routing):
    import numpy as np

    routing.jt.concat = lambda values, dim: np.concatenate(values, axis=dim)
    function = _definitions("nn/serving_ops.py", ["rotary_emb"], routing.namespace)["rotary_emb"]
    query = np.array([[1, 2, 3, 4]], dtype="float32")
    key = np.array([[5, 6, 7, 8]], dtype="float32")
    real = np.zeros((1, 4), dtype="float32")
    imaginary = np.ones((1, 4), dtype="float32")
    actual_query, actual_key = function(query, key, freq_cos=real, freq_sin=imaginary)
    np.testing.assert_array_equal(actual_query, [[-3, -4, 1, 2]])
    np.testing.assert_array_equal(actual_key, [[-7, -8, 5, 6]])
