"""FP16 mapped RMSNorm and transitions to the original FP32 training path."""

import gc

import numpy as np
import pytest
import jittor as jt

from _helpers import capability as _test_capability
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor._runtime.graph_replay import GraphReplay
from jittor.backends.acl.kernels import normalization
from jittor.backends.acl.kernels.ops import norms_op
from jittor.backends.acl.kernels.ops._code import acl_code


@pytest.fixture(autouse=True)
def acl_inference():
    if not _test_capability.check_accelerator("acl", backend=jt).enabled:
        pytest.skip("real ACL device required for mapped RMSNorm")
    before = jt.core.backend_fallback_count()
    with jt.flag_scope(use_acl=1, use_cuda=1, auto_graph_replay=0), jt.no_grad(), forbid_backend_fallbacks():
        yield
    assert jt.core.backend_fallback_count() == before


def _inputs(shape, dtype, weight_dtype):
    rng = np.random.RandomState(4910)
    values = rng.uniform(-1.5, 1.5, shape).astype(dtype)
    weights = rng.uniform(0.5, 1.5, (shape[-1],)).astype(weight_dtype)
    x, weight = jt.array(values).stop_grad(), jt.array(weights).stop_grad()
    jt.sync([x, weight], True, False)
    return x, weight, values, weights


def _reference(values, weights, epsilon):
    # The existing RmsNormAttr stores epsilon as float before CANN receives it.
    wide = values.astype(np.float64)
    rstd = 1 / np.sqrt(np.mean(wide * wide, axis=-1, keepdims=True) + float(np.float32(epsilon)))
    return (wide * rstd * weights).astype(values.dtype), rstd.astype(np.float32)


@pytest.mark.parametrize("shape", [(16,), (2, 17), (2, 3, 64)])
@pytest.mark.parametrize("epsilon", [1e-6, 0.125])
def test_mapped_rms_norm_matches_reference_and_original_function(shape, epsilon):
    x, weight, values, weights = _inputs(shape, "float16", "float16")
    output = jt.core.ops.mapped_rms_norm(x, weight, epsilon)
    context = norms_op.RmsNormACL()
    legacy = context.execute(x, weight, epsilon)
    jt.sync([output, legacy], True, False)
    assert isinstance(output, jt.Var)
    assert tuple(output.shape) == shape
    assert str(output.dtype) == "float16"
    assert output.location() == legacy.location() == "device"
    actual = output.numpy()
    np.testing.assert_array_equal(actual, legacy.numpy())
    np.testing.assert_allclose(actual, _reference(values, weights, epsilon)[0], rtol=3e-3, atol=3e-3)


def test_inference_scratch_survives_dropping_the_intermediate_holder():
    x, weight, values, weights = _inputs((2, 3, 32), "float16", "float16")
    # Both inputs stop gradients, and the result holder is also discarded
    # before execution. Only the consuming graph keeps the op and scratch.
    assert x.is_stop_grad() and weight.is_stop_grad()
    result = (jt.core.ops.mapped_rms_norm(x, weight, 1e-6) + 0.25).tanh()
    gc.collect()
    result.sync(True, False)
    assert result.location() == "device"
    expected = _reference(values, weights, 1e-6)[0]
    expected = np.tanh((expected + 0.25).astype(expected.dtype))
    np.testing.assert_allclose(result.numpy(), expected, rtol=3e-3, atol=2e-3)


def test_inference_scratch_remains_valid_across_graph_replays():
    _, weight, _, weights = _inputs((2, 32), "float16", "float16")

    class Norm(jt.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", weight)

        def execute(self, x):
            return jt.core.ops.mapped_rms_norm(x, self.weight, 1e-6)

    replay = GraphReplay(Norm())
    rng = np.random.RandomState(4911)
    try:
        for _ in range(8):
            values = rng.uniform(-1.5, 1.5, (2, 32)).astype(np.float16)
            output = replay(jt.array(values).stop_grad())
            output.sync(True, False)
            assert output.location() == "device"
            np.testing.assert_allclose(output.numpy(), _reference(values, weights, 1e-6)[0],
                                       rtol=3e-3, atol=2e-3)
        assert replay.refused is None
        assert replay.stats["replayed"] >= 6
    finally:
        replay.invalidate()


def test_inference_shape_changes_rebuild_scratch_without_stale_values():
    _, weight, _, weights = _inputs((1, 32), "float16", "float16")

    class Norm(jt.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", weight)

        def execute(self, x):
            return jt.core.ops.mapped_rms_norm(x, self.weight, 1e-6)

    replay = GraphReplay(Norm())
    rng = np.random.RandomState(4914)
    try:
        for shape in ((1, 32), (2, 3, 32), (3, 5, 32), (1, 32)):
            values = rng.uniform(-1.5, 1.5, shape).astype(np.float16)
            output = replay(jt.array(values))
            np.testing.assert_allclose(output.numpy(), _reference(values, weights, 1e-6)[0],
                                       rtol=3e-3, atol=2e-3)
        assert replay.stats["captured"] == 4
    finally:
        replay.invalidate()


@pytest.mark.parametrize("execute", [False, True])
def test_dropping_a_mapped_rms_norm_graph_releases_private_scratch(execute):
    x, weight, _, _ = _inputs((2, 32), "float16", "float16")

    def build_and_drop():
        output = jt.core.ops.mapped_rms_norm(x, weight, 1e-6)
        if execute:
            output.sync(True, False)

    build_and_drop()
    gc.collect()
    before = (jt.introspection.counters.live_vars, jt.introspection.counters.live_ops)
    for _ in range(12):
        build_and_drop()
    gc.collect()
    after = (jt.introspection.counters.live_vars, jt.introspection.counters.live_ops)
    assert after == before


def test_mapped_rms_norm_materializes_strided_inputs():
    x, weight, values, weights = _inputs((2, 3, 32), "float16", "float16")
    with jt.flag_scope(transpose_storage_view=1):
        strided = jt.core.ops.transpose(x, (1, 0, 2))
    assert not strided._storage_is_contiguous()
    output = jt.core.ops.mapped_rms_norm(strided, weight, 1e-6)
    output.sync(True, False)
    assert output.location() == "device"
    expected = _reference(values.transpose(1, 0, 2), weights, 1e-6)[0]
    np.testing.assert_allclose(output.numpy(), expected, rtol=3e-3, atol=3e-3)


def test_positive_epsilon_is_validated_before_legacy_float_rounding():
    x, weight, _, _ = _inputs((2, 16), "float16", "float16")
    epsilon = 1e-50
    assert epsilon > 0 and np.float32(epsilon) == 0
    output = normalization._rms_norm_cuda_acl(x, weight, epsilon)
    assert output is not None
    context = norms_op.RmsNormACL()
    legacy = context.execute(x, weight, epsilon)
    jt.sync([output, legacy], True, False)
    np.testing.assert_array_equal(output.numpy(), legacy.numpy())


@pytest.mark.parametrize("epsilon", [0.0, -1.0, float("inf"), float("nan"), object()])
def test_normalization_hook_preserves_none_for_unsupported_epsilon(epsilon):
    x, weight, _, _ = _inputs((2, 16), "float16", "float16")
    assert normalization._rms_norm_cuda_acl(x, weight, epsilon) is None


@pytest.mark.parametrize("shape,weight_shape,dtype,weight_dtype", [
    ((0, 16), (16,), "float16", "float16"),
    ((2, 16), (15,), "float16", "float16"),
    ((2, 16), (1, 16), "float16", "float16"),
    ((0, 16), (16,), "float32", "float32"),
    ((2, 16), (15,), "float32", "float32"),
    ((2, 16), (1, 16), "float32", "float32"),
    ((2, 16), (16,), "float32", "float16"),
    ((2, 16), (16,), "float64", "float64"),
])
def test_normalization_hook_preserves_none_for_unsupported_metadata(shape, weight_shape, dtype, weight_dtype):
    x = jt.ones(shape, dtype=dtype)
    weight = jt.ones(weight_shape, dtype=weight_dtype)
    assert normalization._rms_norm_cuda_acl(x, weight, 1e-6) is None


def test_normalization_hook_does_not_hide_valid_input_runtime_errors(monkeypatch):
    x, weight, _, _ = _inputs((2, 16), "float16", "float16")
    failure = RuntimeError("unexpected RMSNorm construction failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(jt.core.ops, "mapped_rms_norm", fail)
    with pytest.raises(RuntimeError) as caught:
        normalization._rms_norm_cuda_acl(x, weight, 1e-6)
    assert caught.value is failure


def test_mixed_dtype_native_rejects_but_dispatcher_preserves_legacy_route(monkeypatch):
    x, weight, _, _ = _inputs((2, 16), "float16", "float32")
    with pytest.raises(RuntimeError, match="float16 input and weight"):
        jt.core.ops.mapped_rms_norm(x, weight, 1e-6)
    # This checks routing, not mixed-dtype CANN support: the original provider
    # may reject that combination on this SoC. No cast or new claim is made.
    seen, result = [], object()

    class Legacy:
        def __call__(self, activation, scale, epsilon):
            seen.append((activation, scale, epsilon))
            return result

    monkeypatch.setattr(normalization, "RmsNormACL", Legacy)
    assert normalization._rms_norm_cuda_acl(x, weight, 1e-6) is result
    assert len(seen) == 1
    assert seen[0][0] is x and seen[0][1] is weight and seen[0][2] == 1e-6


@pytest.mark.parametrize("invalid", ["epsilon", "shape", "dtype", "training", "cpu"])
def test_native_entry_rejects_inputs_outside_its_inference_contract(invalid):
    x, weight, _, _ = _inputs((2, 16), "float16", "float16")
    if invalid == "epsilon":
        with pytest.raises(RuntimeError, match="epsilon"):
            jt.core.ops.mapped_rms_norm(x, weight, 0.0)
    elif invalid == "shape":
        with pytest.raises(RuntimeError, match="weight"):
            jt.core.ops.mapped_rms_norm(x, jt.ones((15,), dtype="float16"), 1e-6)
    elif invalid == "dtype":
        with pytest.raises(RuntimeError, match="float16 input and weight"):
            jt.core.ops.mapped_rms_norm(x, weight.float32(), 1e-6)
    elif invalid == "training":
        with jt.enable_grad(), pytest.raises(RuntimeError, match="no_grad"):
            jt.core.ops.mapped_rms_norm(x, weight, 1e-6)
    else:
        with jt.flag_scope(use_acl=0, use_cuda=0):
            with pytest.raises(RuntimeError, match="ACL device backend"):
                jt.core.ops.mapped_rms_norm(x, weight, 1e-6)


def test_float32_native_rejects_while_original_provider_remains_correct(monkeypatch):
    x, weight, values, weights = _inputs((2, 3, 1024), "float32", "float32")
    with pytest.raises(RuntimeError, match="float16 input and weight"):
        jt.core.ops.mapped_rms_norm(x, weight, 1e-6)

    def unexpected(*args):
        raise AssertionError("FP32 must retain the original Function path")

    monkeypatch.setattr(jt.core.ops, "mapped_rms_norm", unexpected)
    output = normalization._rms_norm_cuda_acl(x, weight, 1e-6)
    assert output is not None
    np.testing.assert_allclose(output.numpy(), _reference(values, weights, 1e-6)[0],
                               rtol=2e-5, atol=2e-5)


def _assert_fp32_training(shape, seed, amp_forward=False):
    rng = np.random.RandomState(seed)
    values = rng.randn(*shape).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, shape[-1]).astype(np.float32)
    cotangent = rng.randn(*shape).astype(np.float32)
    rstd = 1 / np.sqrt(np.mean(values * values, axis=-1, keepdims=True) + 1e-6)
    weighted = cotangent * weights
    expected = (
        values * rstd * weights,
        weighted * rstd - values * rstd ** 3 * np.mean(weighted * values, axis=-1, keepdims=True),
        np.sum(cotangent * values * rstd, axis=tuple(range(len(shape) - 1))),
    )
    with jt.enable_grad():
        x, weight = jt.array(values), jt.array(weights)
        with jt.flag_scope(amp_reg=2 if amp_forward else 0):
            output = normalization._rms_norm_cuda_acl(x, weight, 1e-6)
        assert output is not None
        gradients = jt.grad((output * jt.array(cotangent)).sum(), [x, weight])
        # One batch, as in the regression: syncing the forward separately
        # changes the storage reuse that the transition has to survive.
        actual = jt.fetch_sync([output] + list(gradients))
        for index, (got, wanted) in enumerate(zip(actual, expected)):
            tolerance = 2e-5 if index == 0 else 3e-5
            np.testing.assert_allclose(got, wanted, rtol=tolerance, atol=tolerance)
        assert normalization._rms_norm_cuda_acl(x.float16(), weight, 1e-6) is None


def test_training_hook_keeps_its_differentiable_path():
    _assert_fp32_training((2, 16), 4912)


def test_310p_training_statistics_stay_float32_under_amp_prefer16(monkeypatch):
    if not norms_op.soc_name().startswith("Ascend310P"):
        pytest.skip("requires an Ascend310P accelerator for this vendor workaround")
    saved = []

    class InspectRms(norms_op.RmsNormACL):
        def execute(self, *args):
            result = super().execute(*args)
            saved.append(self.rstd)
            assert self._vendor_rstd is not self.rstd
            return result

    monkeypatch.setattr(normalization, "RmsNormACL", InspectRms)
    with jt.flag_scope(amp_reg=0):
        _assert_fp32_training((2, 3, 1024), 4915, amp_forward=True)
    assert len(saved) == 1
    assert str(saved[0].dtype) == "float32"
    assert saved[0].is_stop_grad()


def test_training_after_float32_code_warmup_does_not_use_vendor_statistics():
    # This exact sequence exposed partly zero vendor rstd although y was
    # correct. Training must use independent device statistics on 310P.
    shape = (2, 3, 1024)
    rng = np.random.RandomState(2026)
    for round_index in range(2):
        with jt.enable_grad():
            x = jt.array(rng.randn(*shape).astype(np.float32))
            weight = jt.array(rng.uniform(0.5, 1.5, shape[-1]).astype(np.float32))
        warmup = acl_code(
            "RmsNorm", inputs=[x, weight], output_dtypes=["float32", "float32"],
            output_shapes=[shape, shape[:-1] + (1,)], attributes={"eps": 1e-6},
        )[0]
        warmup.numpy()
        del warmup, x, weight
        _assert_fp32_training(shape, 20260830 + round_index)


@pytest.mark.parametrize("shape", [(1, 1, 64), (2, 3, 1024), (3, 5, 128)])
@pytest.mark.parametrize("warmup", ["fp16_y", "fp16_replay", "fp32_function"])
def test_inference_to_float32_training_preserves_all_gradients(shape, warmup):
    rng = np.random.RandomState(4913)
    values = rng.randn(*shape).astype(np.float32)
    weights = rng.uniform(0.5, 1.5, shape[-1]).astype(np.float32)

    def infer(round_index):
        dtype = "float32" if warmup == "fp32_function" else "float16"
        warm_values, warm_weights = values.astype(dtype), weights.astype(dtype)
        # Match callers that create trainable inputs before opening no_grad.
        with jt.enable_grad():
            x, weight = jt.array(warm_values), jt.array(warm_weights)
        held = None
        if warmup == "fp16_replay":
            class Norm(jt.Module):
                def __init__(self):
                    super().__init__()
                    self.register_buffer("weight", weight)

                def execute(self, activation):
                    return jt.core.ops.mapped_rms_norm(activation, self.weight, 1e-6)

            held = GraphReplay(Norm())
            for _ in range(3):
                output = held(x)
                output.sync(True, False)
            assert held.refused is None
            assert held.stats["replayed"] >= 3
        else:
            output = normalization._rms_norm_cuda_acl(x, weight, 1e-6)
        assert output is not None
        tolerance = 2e-5 if dtype == "float32" else 3e-3
        np.testing.assert_allclose(output.numpy(), _reference(warm_values, warm_weights, 1e-6)[0],
                                   rtol=tolerance, atol=tolerance)
        return held

    for round_index in range(2):
        retained = []
        if round_index:
            # Perturb both small auxiliary and full activation allocations;
            # leave one buffer alive across inference and backward.
            sizes = (1, shape[-1] + 1, int(np.prod(shape)) + 7)
            churn = [jt.array(np.full((size,), 0.25, dtype=dtype))
                     for size in sizes for dtype in ("float16", "float32")]
            jt.sync(churn, True, False)
            retained = churn[-1:]
            del churn
            gc.collect()
        capture = infer(round_index)
        try:
            _assert_fp32_training(shape, 20260830 + round_index)
        finally:
            if capture is not None:
                capture.invalidate()
        del capture, retained
        gc.collect()
