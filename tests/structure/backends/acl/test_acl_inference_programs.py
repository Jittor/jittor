"""Record real ACL graph builders without loading Jittor, compiling, or using CANN."""

from contextlib import nullcontext
import struct
import sys
from types import SimpleNamespace

import pytest

from _helpers.acl_recording_pipeline import build_pipeline


@pytest.fixture
def pipeline(monkeypatch):
    return build_pipeline(monkeypatch)


@pytest.fixture
def inference_pipeline(pipeline, monkeypatch):
    load, Tensor, calls = pipeline
    jt = sys.modules["jittor"]
    function_calls, empty_calls = [], []
    mapped_calls = []

    def mapped_concat(inputs, dim):
        output = object()
        mapped_calls.append((inputs, dim, output))
        return output

    class Function:
        def __call__(self, *args):
            function_calls.append(args)
            return self.execute(*args)

    def empty(shape, dtype):
        output = Tensor(shape, dtype)
        empty_calls.append(output)
        return output

    record_code = jt.code

    def code(*args, **kwargs):
        outputs = record_code(*args, **kwargs)
        calls[-1]["result_shapes"] = [tensor.shape for tensor in outputs]
        calls[-1]["result_dtypes"] = [tensor.dtype for tensor in outputs]
        calls[-1]["results"] = outputs
        return outputs

    monkeypatch.setattr(jt, "Function", Function)
    monkeypatch.setattr(jt, "empty", empty)
    monkeypatch.setattr(jt, "code", code)
    monkeypatch.setattr(jt, "core", SimpleNamespace(
        ops=SimpleNamespace(mapped_concat=mapped_concat)), raising=False)
    monkeypatch.setattr(jt.flags, "no_grad", 1)
    return SimpleNamespace(load=load, Tensor=Tensor, calls=calls, jt=jt,
                           function_calls=function_calls, empty_calls=empty_calls,
                           mapped_calls=mapped_calls)


@pytest.mark.parametrize("dtype,weight_dtype", [
    ("float16", "float16"), ("float32", "float32"),
    ("bfloat16", "bfloat16"), ("float16", "float32"),
])
def test_rmsnorm_function_keeps_original_context_and_output_buffers(inference_pipeline, dtype, weight_dtype):
    p = inference_pipeline
    norm_class = p.load("norms_op").RmsNormACL
    assert "__call__" not in norm_class.__dict__
    assert not hasattr(p.load("norms_op"), "_rmsnorm_inference_program")
    norm = norm_class()
    x, weight = p.Tensor((2, 3, 128), dtype), p.Tensor((128,), weight_dtype)
    output = norm(x, weight, 1e-6)
    assert p.function_calls == [(x, weight, 1e-6)]
    assert len(p.empty_calls) == 2
    assert len(p.calls) == 1
    assert p.calls[0]["inputs"][0] is x and p.calls[0]["inputs"][1] is weight
    assert p.calls[0]["result_shapes"] == [(2, 3, 128), (2, 3, 1)]
    assert p.calls[0]["result_dtypes"] == [dtype, "float32"]
    assert output is p.calls[0]["outputs"][0]
    assert norm.rstd is p.calls[0]["outputs"][1]
    data = p.load("_attributes").attribute_data
    assert p.calls[0]["data"] == data("RmsNorm", {"eps": 1e-6})


def test_rmsnorm_legacy_attributes_retain_signed_zero(inference_pipeline):
    p = inference_pipeline
    norm = p.load("norms_op").RmsNormACL()
    for eps in (0.0, -0.0):
        norm(p.Tensor((2, 8)), p.Tensor((8,)), eps)
        actual = p.calls[-1]["data"]["acl_attr.field.0.value"]
        assert struct.pack("d", actual) == struct.pack("d", eps)


def test_rmsnorm_training_still_uses_function_and_saved_rstd(inference_pipeline, monkeypatch):
    p = inference_pipeline
    module = p.load("norms_op")
    monkeypatch.setattr(module, "soc_name", lambda: "Ascend910B")
    norm = module.RmsNormACL()
    x, weight, grad_output = p.Tensor((2, 3, 8)), p.Tensor((8,)), p.Tensor((2, 3, 8))
    monkeypatch.setattr(p.jt.flags, "no_grad", 0)
    result = norm(x, weight, 1e-6)
    assert p.function_calls == [(x, weight, 1e-6)]
    assert len(p.empty_calls) == 2
    assert result is p.calls[-1]["outputs"][0]
    saved_rstd = p.calls[-1]["outputs"][1]
    norm.grad(grad_output)
    assert p.calls[-1]["inputs"] == [grad_output, x, saved_rstd, weight]
    assert p.calls[-1]["result_shapes"] == [(2, 3, 8), (8,)]
    assert "RmsNormGradOpRunner" in p.calls[-1]["cuda_src"]


def test_310p_training_retains_vendor_aux_but_passes_device_statistics_to_backward(inference_pipeline, monkeypatch):
    p = inference_pipeline
    module = p.load("norms_op")
    monkeypatch.setattr(module, "soc_name", lambda: "Ascend310P3")
    monkeypatch.setattr(p.jt.flags, "no_grad", 0)
    monkeypatch.setattr(p.jt, "no_grad", nullcontext, raising=False)
    monkeypatch.setattr(p.jt, "flag_scope", lambda **kwargs: nullcontext(), raising=False)
    monkeypatch.setattr(p.Tensor, "float32", lambda x: p.Tensor(x.shape, "float32"), raising=False)
    monkeypatch.setattr(p.Tensor, "__mul__", lambda x, y: p.Tensor(x.shape, "float32"), raising=False)
    monkeypatch.setattr(p.Tensor, "mean", lambda x, dim, keepdims: p.Tensor(x.shape[:-1] + (1,), "float32"), raising=False)
    monkeypatch.setattr(p.Tensor, "__add__", lambda x, eps: p.Tensor(x.shape, "float32"), raising=False)
    statistics = p.Tensor((2, 3, 1), "float32")
    monkeypatch.setattr(p.jt, "rsqrt", lambda x: statistics, raising=False)
    x, weight, grad_output = p.Tensor((2, 3, 8)), p.Tensor((8,)), p.Tensor((2, 3, 8))
    norm = module.RmsNormACL()
    norm(x, weight, 1e-6)
    vendor = p.calls[-1]["outputs"][1]
    assert norm._vendor_rstd is vendor
    assert norm.rstd is statistics and norm.rstd is not vendor
    norm.grad(grad_output)
    assert p.calls[-1]["inputs"] == [grad_output, x, statistics, weight]


def test_concat_inference_passes_current_inputs_and_axis_to_native_once(inference_pipeline):
    p = inference_pipeline
    concat = p.load("concat_op").ConcatACL()
    cases = [
        ([p.Tensor((2, 3), "float16"), p.Tensor((2, 5), "float16")], -1),
        ([p.Tensor((2, 7), "float32"), p.Tensor((2, 8), "float32")], 1),
        ([p.Tensor((2, 4)), p.Tensor((3, 4))], 0),
        ([p.Tensor((2, 4)) for _ in range(3)], 1),
    ]
    for count, (inputs, dim) in enumerate(cases, 1):
        result = concat(inputs, dim)
        assert len(p.mapped_calls) == count
        native_inputs, native_dim, native_result = p.mapped_calls[-1]
        assert native_inputs is inputs and native_dim == dim
        assert result is native_result
    assert p.calls == p.empty_calls == []
    assert not hasattr(concat, "input")


def test_concat_training_still_encodes_current_split_sizes(inference_pipeline, monkeypatch):
    p = inference_pipeline
    concat = p.load("concat_op").ConcatACL()
    concat([p.Tensor((2, 3)), p.Tensor((2, 5))], 1)
    monkeypatch.setattr(p.jt.flags, "no_grad", 0)
    for left, right in ((3, 5), (7, 8)):
        concat([p.Tensor((2, left)), p.Tensor((2, right))], 1)
        call = p.calls[-1]
        assert call["data"]["multi_grad"] == 1
        assert "SplitWithSizeOpRunner" in call["cuda_grad_src"][0]
        expected = p.load("_attributes").attribute_program(
            "SplitWithSize", {"splitSize": [left, right], "dim": 1}, variable="op",
        ).data
        assert all(call["data"].get(key) == value for key, value in expected.items())
        assert call["result_shapes"] == [(2, left + right)]


def test_concat_inference_does_not_read_python_tensor_metadata(inference_pipeline):
    p = inference_pipeline
    class NativeTensor:
        def __getattr__(self, name):
            raise AssertionError("Python read tensor metadata: " + name)

    inputs = (NativeTensor(), NativeTensor())
    result = p.load("concat_op").ConcatACL()(inputs, -1)
    assert p.mapped_calls == [(inputs, -1, result)]
    assert p.calls == []


def test_concat_inference_retains_validation_errors_and_backend_failures(inference_pipeline, monkeypatch):
    p = inference_pipeline
    concat = p.load("concat_op").ConcatACL()
    error = RuntimeError("native concat construction failed")

    def fail(*args):
        raise error

    monkeypatch.setattr(p.jt.core.ops, "mapped_concat", fail)
    with pytest.raises(ValueError, match="at least one"):
        concat([], 0)
    with pytest.raises(ValueError, match="dim out of range"):
        concat([p.Tensor((2, 3))], 2)
    with pytest.raises(ValueError, match="same dtype"):
        concat([p.Tensor((2, 3), "float16"), p.Tensor((2, 3), "float32")], 0)
    with pytest.raises(ValueError, match="same shape"):
        concat([p.Tensor((2, 3)), p.Tensor((2, 4))], 0)
    with pytest.raises(RuntimeError) as caught:
        concat([p.Tensor((2, 3)), p.Tensor((2, 3))], 0)
    assert caught.value is error
