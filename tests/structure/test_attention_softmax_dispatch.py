"""Actual attention/dispatch execution with NumPy tensors, without a JIT build."""

import ast
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]


class _Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = self.value.dtype

    def cast(self, dtype):
        return _Tensor(self.value.astype(str(dtype)))

    def broadcast(self, shape):
        return _Tensor(np.broadcast_to(self.value, shape))

    def transpose(self, first, second):
        return _Tensor(self.value.swapaxes(first, second))

    def sum(self, dim, keepdims=False):
        return _Tensor(self.value.sum(dim, keepdims=keepdims))

    def __mul__(self, other):
        return _Tensor(self.value * _array(other))

    def __add__(self, other):
        return _Tensor(self.value + _array(other))

    def __and__(self, other):
        return _Tensor(self.value & _array(other))

    def __lt__(self, other):
        return _Tensor(self.value < _array(other))

    def __gt__(self, other):
        return _Tensor(self.value > _array(other))


def _array(value):
    return value.value if isinstance(value, _Tensor) else value


def _softmax(value, dim=-1, zero_all_neg_inf=False):
    with np.errstate(invalid="ignore"):
        shifted = value.value - value.value.max(dim, keepdims=True)
        exp = np.exp(shifted)
        result = exp / exp.sum(dim, keepdims=True)
    if zero_all_neg_inf:
        result = np.where(np.isneginf(value.value).all(dim, keepdims=True), 0, result)
    return _Tensor(result)


@pytest.mark.parametrize("mask_kind", ["bool", "float", "none"])
@pytest.mark.parametrize("is_causal", [False, True])
@pytest.mark.parametrize("zero_kernel", [False, True])
def test_attention_qualifies_the_softmax_variant_it_executes(
    monkeypatch, mask_kind, is_causal, zero_kernel
):
    def load(name, relative):
        spec = importlib.util.spec_from_file_location(name, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    for name in (
        "jittor",
        "jittor._runtime",
        "jittor.backends",
        "jittor.backends.cuda",
        "jittor.backends.cuda.kernels",
        "jittor.backends.cuda.kernels.nn",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    jt = sys.modules["jittor"]
    jt.core = SimpleNamespace(Var=_Tensor, dispatch_context=lambda tensors: ("acl", 0))
    jt.runtime = SimpleNamespace(use_cuda=1)
    jt.nn = SimpleNamespace(
        matmul=lambda left, right: _Tensor(np.matmul(left.value, right.value)),
        softmax=_softmax,
    )
    jt.array = _Tensor
    jt.ones = lambda shape, dtype: _Tensor(np.ones(shape, dtype=dtype))
    jt.zeros_like = lambda value: _Tensor(np.zeros_like(value.value))
    jt.triu = lambda value, diagonal: _Tensor(np.triu(value.value, diagonal))
    jt.ternary = lambda condition, yes, no: _Tensor(np.where(condition.value, yes.value, no.value))
    jt.logical_not = lambda value: _Tensor(np.logical_not(value.value))
    jt.isinf = lambda value: _Tensor(np.isinf(value.value))
    dispatch = load("jittor._runtime.dispatch", "python/jittor/_runtime/dispatch.py")
    libraries = ModuleType("jittor._runtime.backend_libraries")
    libraries.library_resource = lambda *args: ""
    monkeypatch.setitem(sys.modules, libraries.__name__, libraries)
    softmax_module = load(
        "jittor.backends.cuda.kernels.nn.softmax_cuda", "backends/cuda/kernels/nn/softmax_cuda.py"
    )
    sys.modules["jittor.backends.cuda.kernels.nn"].softmax_cuda = softmax_module
    attention = load("offline_attention", "python/jittor/nn/functional/attention.py")
    source = (ROOT / "backends/acl/kernels/neural.py").read_text(encoding="utf-8")
    node = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "softmax_supported"
    )
    namespace = {}
    exec(
        compile(ast.get_source_segment(source, node), "<actual_acl_softmax_support>", "exec"),
        namespace,
    )
    acl_supports = namespace["softmax_supported"]
    probes, executions = [], []

    def supports(value, log=False, zero_all_neg_inf=False, dim=-1):
        options = (log, zero_all_neg_inf, dim)
        probes.append(options)
        return zero_kernel or acl_supports(
            value, log=log, zero_all_neg_inf=zero_all_neg_inf, dim=dim
        )

    def implementation(value, log=False, zero_all_neg_inf=False, dim=-1):
        executions.append((log, zero_all_neg_inf, dim))
        return _softmax(value, dim, zero_all_neg_inf)

    dispatch.register_kernel("nn.softmax", "acl", implementation, supports=supports)
    query = _Tensor(np.zeros((1, 1, 2, 2), dtype=np.float32))
    value = _Tensor(np.array([[[[2, 4], [6, 8]]]], dtype=np.float32))
    keep = np.array([[True, False], [False, False]])
    if mask_kind == "bool":
        mask = _Tensor(keep)
    elif mask_kind == "float":
        mask = _Tensor(np.where(keep, 0, -np.inf).astype(np.float32))
    else:
        mask = None
    result = attention.scaled_dot_product_attention(
        query, query, value, attn_mask=mask, is_causal=is_causal
    )
    expected = (
        [[[[2, 4], [0, 0]]]]
        if mask is not None
        else [[[[2, 4], [4, 6]]]]
        if is_causal
        else [[[[4, 6], [4, 6]]]]
    )
    np.testing.assert_array_equal(result.value, np.asarray(expected, dtype=np.float32))
    expected_options = (False, mask is not None, -1)
    assert probes == [expected_options]
    assert executions == ([expected_options] if zero_kernel or mask is None else [])
