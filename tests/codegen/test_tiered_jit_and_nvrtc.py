"""Tests for Tier-0 precompiled meta-ops, tiered async JIT promotion, and in-process NVRTC."""

import json
import os

import numpy as np
import pytest

import jittor as jt
from _helpers import capability as _test_capability
from _helpers.child_process import run_child_script


def _has_cuda() -> bool:
    return _test_capability.check_accelerator("cuda", backend=jt).enabled


def _run_child(code: str, extra_env: dict, clear_nvrtc_cache: bool = False):
    if clear_nvrtc_cache:
        jit_dir = os.path.join(jt.compiler.cache_path, "jit")
        if os.path.isdir(jit_dir):
            for name in os.listdir(jit_dir):
                if name.endswith(".nvrtc.cubin"):
                    try:
                        os.remove(os.path.join(jit_dir, name))
                    except OSError:
                        pass
    proc = run_child_script(
        code,
        env=extra_env,
        without_torch_mode=True,
        timeout=180,
    )
    stdout = proc.stdout.decode("utf-8", "replace") if isinstance(proc.stdout, bytes) else proc.stdout
    stderr = proc.stderr.decode("utf-8", "replace") if isinstance(proc.stderr, bytes) else proc.stderr
    assert proc.returncode == 0, (
        f"Child failed rc={proc.returncode}\nSTDOUT:\n{stdout[-2000:]}\n"
        f"STDERR:\n{stderr[-2000:]}"
    )
    for line in reversed(stdout.splitlines()):
        if line.startswith("RESULT_JSON:"):
            return json.loads(line[len("RESULT_JSON:"):])
    return stdout


def _parity_suite(use_cuda: int):
    """Run comprehensive forward & backward ops under Tier-0 and compare against Tier-1."""
    rng = np.random.RandomState(2026)
    x_np = rng.randn(32, 48).astype(np.float32)
    y_np = rng.randn(32, 48).astype(np.float32)
    b1_np = rng.randn(1, 48).astype(np.float32)
    b2_np = rng.randn(32, 1).astype(np.float32)
    m_np = rng.randn(32, 48) > 0
    m2_np = rng.randn(32, 48) > 0
    xi_np = rng.randint(-20, 20, size=(32, 48)).astype(np.int32)
    yi_np = rng.randint(1, 7, size=(32, 48)).astype(np.int32)
    yi_signed_np = np.where(rng.rand(32, 48) > 0.5, yi_np, -yi_np).astype(np.int32)
    xi64_np = rng.randint(-1000, 1000, size=(16, 24)).astype(np.int64)
    yi64_np = rng.randint(1, 13, size=(16, 24)).astype(np.int64)
    xi8_np = rng.randint(-50, 50, size=(16, 24)).astype(np.int8)
    xu8_np = rng.randint(1, 50, size=(16, 24)).astype(np.uint8)
    x64_np = rng.randn(16, 24).astype(np.float64)
    y64_np = rng.randn(16, 24).astype(np.float64)
    x3d_np = rng.randn(4, 8, 12).astype(np.float32)

    def run_all(tier: str):
        with jt.flag_scope(use_cuda=use_cuda, jit_tier=tier, use_nvrtc=0, reuse_dying_inputs=2):
            x = jt.array(x_np)
            y = jt.array(y_np)
            b1 = jt.array(b1_np)
            b2 = jt.array(b2_np)
            m = jt.array(m_np)
            m2 = jt.array(m2_np)
            xi = jt.array(xi_np)
            yi = jt.array(yi_signed_np)
            xi64 = jt.array(xi64_np)
            yi64 = jt.array(yi64_np)
            xi8 = jt.array(xi8_np)
            xu8 = jt.array(xu8_np)
            x64 = jt.array(x64_np)
            y64 = jt.array(y64_np)
            x3d = jt.array(x3d_np)
            jt.sync_all(True)

            out = {}
            # Unary math ops on float32
            out["unary_chain"] = (
                jt.abs(x) + (-y) + jt.exp(jt.minimum(jt.maximum(x, -2.0), 2.0))
                + jt.log(jt.abs(x) + 1.0) + jt.sqrt(jt.abs(y) + 0.5)
                + jt.sin(x) + jt.cos(y) + jt.tan(x * 0.2) + jt.tanh(y)
                + jt.sigmoid(x) + jt.erf(y)
                + jt.ceil(x * 2.0) + jt.floor(y * 2.0) + jt.round(x * 3.0)
            ).numpy()
            # Binary + broadcast + strided transpose
            out["bcast_strided"] = (
                (x + b1) * (y - b2) / (jt.abs(x) + 1.0)
                + jt.maximum(x, y) - jt.minimum(x, b1)
                + (x.transpose(1, 0).transpose(1, 0) * 0.5)
            ).numpy()
            # Negative base with integer exponent pow
            out["neg_pow"] = (jt.pow(x, 3.0) + jt.pow(y, 2.0)).numpy()
            # NaN propagation in maximum / minimum
            nan_a = jt.array(np.array([1.0, np.nan, -2.0, 5.0], dtype=np.float32))
            nan_b = jt.array(np.array([2.0, 3.0, np.nan, 4.0], dtype=np.float32))
            out["nan_max"] = jt.maximum(nan_a, nan_b).numpy()
            out["nan_min"] = jt.minimum(nan_a, nan_b).numpy()
            # Float floor_divide / mod
            out["f_divmod"] = (jt.floor_divide(x, jt.abs(y) + 0.5) + jt.mod(x, jt.abs(y) + 0.5)).numpy()
            # Int32 arithmetic, floor_divide, mod, bitwise
            out["i32_ops"] = ((xi + yi) * 3 - jt.floor_divide(xi, yi) + jt.mod(xi, yi)).numpy()
            out["i32_bit"] = ((xi & yi) | (xi ^ yi) + jt.bitwise_not(xi)).numpy()
            # Int64, int8, uint8, bool logic & bool arithmetic
            out["i64_ops"] = (xi64 * 2 + jt.floor_divide(xi64, yi64) + jt.mod(xi64, yi64)).numpy()
            out["i8_u8"] = (xi8.int32() + xu8.int32()).numpy()
            out["bool_logic"] = (
                (x > y).logical_and(m).logical_or(jt.logical_not(m)).logical_xor(x < b1)
            ).numpy()
            out["bool_add"] = (m + m2).numpy()
            out["bool_mul"] = (m * m2).numpy()
            out["bool_sub"] = (m - m2).numpy()
            # Ternary (where)
            out["ternary"] = jt.where(m, x * 2.0 + b1, y - b2).numpy()
            # Casts across fp16, bf16, fp64
            out["fp16"] = (x.float16() * y.float16() + b1.float16()).float32().numpy()
            out["bf16"] = (x.bfloat16() * y.bfloat16() - b2.bfloat16()).float32().numpy()
            out["fp64"] = (jt.sin(x64) + jt.exp(y64 * 0.25) * x64).numpy()
            # High-precision float64 -> round_int / floor_int / ceil_int (beyond fp32 24-bit mantissa)
            big_f64 = jt.array(np.array([100000003.4, 100000003.6, -100000003.6], dtype=np.float64))
            out["f64_round_int"] = jt.round_int(big_f64).numpy()
            out["f64_floor_int"] = jt.floor_int(big_f64).numpy()
            out["f64_ceil_int"] = jt.ceil_int(big_f64).numpy()
            # Signed integer edge values (INT32_MIN abs/neg without UB)
            imin32 = jt.array(np.array([-2147483648, -1, 0, 1, 2147483647], dtype=np.int32))
            out["imin32_abs"] = jt.abs(imin32).numpy()
            out["imin32_neg"] = (-imin32).numpy()
            # Reductions: contiguous, strided, multi-axis, full, 0-D scalar, logical
            out["red_sum_1"] = (x * y + b1).sum(dim=1).numpy()
            out["red_mean_0"] = (x - b2).mean(dim=0).numpy()
            out["red_max_1"] = x.max(dim=1).numpy()
            out["red_min_0"] = y.min(dim=0).numpy()
            out["red_prod"] = (jt.abs(x3d[:, :4, :4]) * 0.1 + 1.0).prod(dim=2).numpy()
            out["red_multi"] = jt.tanh(x3d).sum(dims=(0, 2)).numpy()
            out["red_strided"] = x.transpose(1, 0).sum(dim=1).numpy()
            out["red_full"] = (x + b1).sum().numpy()
            out["red_all"] = jt.all_(x > -10.0, dim=1).numpy()
            out["red_any"] = jt.any_(m, dim=0).numpy()
            # Scalar 0-D reduction
            out["red_scalar"] = jt.array(np.float32(3.5)).sum().numpy()
            # Multi-output fused op + reuse_dying_inputs aliasing safety
            tmp = x + y
            o1 = tmp * 2.0 + 1.0
            o2 = tmp - 3.0
            jt.sync([o1, o2], True)
            out["multi_o1"] = o1.numpy()
            out["multi_o2"] = o2.numpy()

            # Non-fused standalone Op::jit_run Tier-0 path
            with jt.flag_scope(no_fuse=1):
                nf = (jt.sin(x) + jt.cos(y)).sum(dim=1)
                nf.sync(True)
                out["no_fuse"] = nf.numpy()

            # Backward gradient parity
            w = jt.array(x_np.copy())
            v = jt.array(y_np.copy())
            loss = (jt.tanh(w * v + b1) * jt.sigmoid(w - b2)).mean()
            gw, gv = jt.grad(loss, [w, v])
            jt.sync([gw, gv], True)
            out["grad_w"] = gw.numpy()
            out["grad_v"] = gv.numpy()

            # Graceful fallback for unsupported op (complex64)
            zc = jt.array((x_np[:4, :4] + 1j * y_np[:4, :4]).astype(np.complex64))
            out["complex_fallback"] = (zc * zc + 1.0).numpy()
            return out

    before = jt.core._jit_tier_stats()
    got = run_all("0")
    after = jt.core._jit_tier_stats()
    ref = run_all("1")
    assert after[2] - before[2] >= 25, (
        f"Expected Tier-0 to execute parity graphs (not hit cached Tier-1), "
        f"stats: {before} -> {after}"
    )

    for k in ref:
        r, g = ref[k], got[k]
        if r.dtype == np.bool_ or np.issubdtype(r.dtype, np.integer):
            if r.dtype == np.bool_:
                # Verify raw bool bytes are strictly 0 or 1
                assert set(np.unique(g.view(np.uint8)).tolist()).issubset({0, 1}), f"Invalid bool byte in {k}"
            np.testing.assert_array_equal(g, r, err_msg=f"Mismatch in {k} (cuda={use_cuda})")
        elif k in ("fp16", "bf16"):
            np.testing.assert_allclose(g, r, rtol=1e-2, atol=1e-2, equal_nan=True,
                                       err_msg=f"Mismatch in {k} (cuda={use_cuda})")
        else:
            np.testing.assert_allclose(g, r, rtol=1e-4, atol=1e-5, equal_nan=True,
                                       err_msg=f"Mismatch in {k} (cuda={use_cuda})")


def test_tier0_cpu_parity_and_fallback():
    _parity_suite(use_cuda=0)


@_test_capability.accelerator_required("cuda")
def test_tier0_cuda_parity_and_fallback():
    _parity_suite(use_cuda=1)


def test_tier0_cpu_int_div_by_zero_raises():
    with jt.flag_scope(use_cuda=0, jit_tier="0"):
        a = jt.array(np.array([10, 20], dtype=np.int32))
        b = jt.array(np.array([2, 0], dtype=np.int32))
        jt.sync_all(True)
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            c = jt.floor_divide(a, b)
            c.sync(True)


def test_jit_tier_flag_validation():
    prev = jt.flags.jit_tier
    try:
        with pytest.raises(Exception):
            jt.flags.jit_tier = "bogus_tier"
        assert jt.flags.jit_tier == prev
    finally:
        jt.flags.jit_tier = prev


@pytest.mark.parametrize("use_cuda", [0, 1])
def test_tiered_auto_async_promotion_and_clean_exit(use_cuda):
    if use_cuda and not _has_cuda():
        pytest.skip("CUDA is required")
    script = f"""
import json, numpy as np, jittor as jt
with jt.flag_scope(use_cuda={use_cuda}):
    a = jt.array(np.random.RandomState(7).randn(64, 64).astype(np.float32))
    b = jt.array(np.random.RandomState(8).randn(64, 64).astype(np.float32))
    jt.sync_all(True)
    ref = np.tanh(a.numpy() * 1.3 + b.numpy() * 0.7 - 0.2)

    s0 = jt.core._jit_tier_stats()
    y1 = jt.tanh(a * 1.3 + b * 0.7 - 0.2); y1.sync(True)
    s1 = jt.core._jit_tier_stats()

    y2 = jt.tanh(a * 1.3 + b * 0.7 - 0.2); y2.sync(True)
    jt.core._jit_tier_wait_bg()
    s2 = jt.core._jit_tier_stats()

    y3 = jt.tanh(a * 1.3 + b * 0.7 - 0.2); y3.sync(True)
    s3 = jt.core._jit_tier_stats()

    np.testing.assert_allclose(y1.numpy(), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(y2.numpy(), ref, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(y3.numpy(), ref, rtol=1e-5, atol=1e-6)
    print("RESULT_JSON:" + json.dumps({{"s0": s0, "s1": s1, "s2": s2, "s3": s3}}))
"""
    res = _run_child(
        script,
        {"JT_JIT_TIER": "auto", "JT_JIT_TIER1_THRESHOLD": "2", "JT_USE_NVRTC": "1"},
        clear_nvrtc_cache=True,
    )
    assert res["s1"][2] > res["s0"][2], f"1st call must use Tier-0: {res}"
    assert res["s1"][3] == res["s0"][3], f"1st call must not enqueue bg compile (threshold=2): {res}"
    assert res["s2"][4] > res["s0"][4], f"2nd call must promote in background: {res}"
    assert res["s3"][2] == res["s2"][2], f"3rd call must hit promoted Tier-1/NVRTC: {res}"


@_test_capability.accelerator_required("cuda")
@pytest.mark.parametrize(
    ("layout", "gate_dtype"),
    [
        ("channel", "float32"),
        ("channel", "bfloat16"),
        ("row", "float32"),
        ("offset", "float32"),
        ("row_pitch", "float32"),
        ("inner_stride", "float32"),
        ("odd_width", "float32"),
    ],
)
def test_nvrtc_packed_broadcasts_and_strided_fallback(layout, gate_dtype):
    rng = np.random.default_rng(91)
    width = 65 if layout == "odd_width" else 64
    shape = (2, 129, width)

    def values(dims):
        # These values round exactly to BF16, so the reference needs only the
        # final narrowing of the expression.
        return rng.integers(-64, 65, size=dims).astype(np.float32) / 32

    left, right = values(shape), values(shape)
    gate_shape = (2, 1, width)
    key = (..., slice(None))
    if layout == "row":
        gate_shape = (2, 129, 1)
    elif layout == "offset":
        gate_shape = (2, 1, width + 8)
        key = (..., slice(1, width + 1))
    elif layout == "row_pitch":
        gate_shape = (2, 1, width + 1)
        key = (..., slice(width))
    elif layout == "inner_stride":
        gate_shape = (2, 1, width * 2)
        key = (..., slice(None, None, 2))

    gates = values(gate_shape)
    with jt.flag_scope(use_cuda=1, use_nvrtc=1, jit_tier="1", cuda_kernel_math="strict",
                       backend_fallback="error"), jt.no_grad():
        a = jt.array(left).bfloat16()
        b = jt.array(right).bfloat16()
        g = jt.array(gates).cast(gate_dtype)[key]
        jt.sync([a, b, g], True)
        launches = jt.core._nvrtc_stats()[3]
        out = (a.float32() + b.float32() * g.float32()).bfloat16()
        out.sync(True)
        assert out.location() == "device"
        assert jt.core._nvrtc_stats()[3] > launches
        expected = jt.array(left + right * gates[key]).bfloat16().float32()
        np.testing.assert_array_equal(out.float32().numpy(), expected.numpy())


@_test_capability.accelerator_required("cuda")
def test_nvrtc_strided_offsets_across_runtime_shapes():
    rng = np.random.default_rng(73)
    with jt.flag_scope(use_cuda=1, use_nvrtc=1, jit_tier="1", backend_fallback="error"), jt.no_grad():
        for batch, heads, tokens, width in [(2, 3, 97, 64), (3, 5, 131, 48), (2, 3, 97, 63)]:
            values = rng.integers(-20, 21, size=(batch, heads, tokens, width)).astype(np.float32)
            offsets = rng.integers(-8, 9, size=(batch, 1, tokens, 1)).astype(np.float32)
            weights = rng.integers(-4, 5, size=(1, heads, 1, width)).astype(np.float32)
            x, offset, weight = [jt.array(v) for v in (values, offsets, weights)]
            jt.sync([x, offset, weight], True)
            launches = jt.core._nvrtc_stats()[3]
            result = x * weight + offset
            result.sync(True)
            assert result.location() == "device" and jt.core._nvrtc_stats()[3] > launches
            np.testing.assert_array_equal(result.numpy(), values * weights + offsets)


@_test_capability.accelerator_required("cuda")
def test_nvrtc_compile_disk_cache_math_and_graph_capture():
    # Child 1: cold NVRTC compile, comprehensive parity (vec4, tails, strided,
    # multi-output, reuse_dying_inputs, scalar-broadcast shape reuse [1]->[64],
    # bool arithmetic raw bytes, strict vs fast math, fp16/bf16/f64/i64/i8/u8/bool,
    # and CUDA graph capture).
    script_cold = """
import json, numpy as np, jittor as jt
with jt.flag_scope(use_cuda=1, reuse_dying_inputs=2):
    rng = np.random.RandomState(123)

    stats0 = jt.core._nvrtc_stats()
    assert stats0[0] == 1, f"NVRTC should be available: {stats0}"

    # 1. Vectorized & non-multiple-of-4 tail lengths
    for n in (1, 3, 4, 5, 17, 1024, 1027):
        x_np = rng.randn(n).astype(np.float32)
        y_np = rng.randn(n).astype(np.float32)
        x = jt.array(x_np); y = jt.array(y_np); jt.sync_all(True)
        z = jt.tanh(x * 1.25 + y * 0.75 - 0.1) + jt.sigmoid(x - y)
        np.testing.assert_allclose(
            z.numpy(),
            np.tanh(x_np * 1.25 + y_np * 0.75 - 0.1) + 1.0 / (1.0 + np.exp(-(x_np - y_np))),
            rtol=1e-5, atol=1e-6,
        )

    # 1b. Structural is_scalar cache reuse: compile with main_shape=[1], then reuse with [64]
    s1 = jt.array(np.array([3.0], dtype=np.float32))
    v1 = jt.array(np.array([2.0], dtype=np.float32))
    v64 = jt.array(np.arange(64, dtype=np.float32))
    jt.sync_all(True)
    r_small = v1 + s1.broadcast([1])
    np.testing.assert_allclose(r_small.numpy(), np.array([5.0], dtype=np.float32))
    r_large = v64 + s1.broadcast([64])
    np.testing.assert_allclose(r_large.numpy(), np.arange(64, dtype=np.float32) + 3.0)

    # 2. Strided/broadcast, multi-output, fp16/bf16/f64/int32/int64/bool, NaN max/min, neg pow
    x2_np = rng.randn(64, 128).astype(np.float32)
    y2_np = rng.randn(64, 128).astype(np.float32)
    b1_np = rng.randn(1, 128).astype(np.float32)
    b2_np = rng.randn(64, 1).astype(np.float32)
    m_np = rng.randn(64, 128) > 0
    m2_np = rng.randn(64, 128) > 0
    x2, y2 = jt.array(x2_np), jt.array(y2_np)
    b1, b2 = jt.array(b1_np), jt.array(b2_np)
    m, m2 = jt.array(m_np), jt.array(m2_np)
    jt.sync_all(True)

    s_out = jt.where(m, x2 * 2.0 + b1, y2 - b2) + jt.pow(x2, 3.0)
    ref_s = np.where(m_np, x2_np * 2.0 + b1_np, y2_np - b2_np) + np.power(x2_np, 3.0)
    np.testing.assert_allclose(s_out.numpy(), ref_s, rtol=1e-4, atol=1e-5)

    base_np = np.array([-3.125, -0.0, 0.0, 0.25, 9.75], dtype=np.float32)
    base = jt.array(base_np)
    base.sync(True)
    cubic = jt.pow(base, 3.0)
    squared = jt.pow(base, 2.0)
    gradient = jt.grad(cubic.sum(), base)
    oracle = base_np.astype(np.float64)
    for actual, expected in ((squared, oracle**2), (cubic, oracle**3),
                             (gradient, 3 * oracle**2)):
        actual.sync(True)
        assert actual.location() == 'device'
        values = actual.numpy()
        np.testing.assert_array_equal(values, expected.astype(np.float32))
        np.testing.assert_array_equal(np.signbit(values), np.signbit(expected))

    # Bool arithmetic must store strictly 0/1 bytes (never 2 for True + True)
    ba = (m + m2).numpy()
    bm = (m * m2).numpy()
    bs = (m - m2).numpy()
    assert set(np.unique(ba.view(np.uint8)).tolist()).issubset({0, 1})
    np.testing.assert_array_equal(ba, np.logical_or(m_np, m2_np))
    np.testing.assert_array_equal(bm, np.logical_and(m_np, m2_np))
    np.testing.assert_array_equal(bs, np.logical_xor(m_np, m2_np))

    # Multi-output + dying input reuse
    tmp = (x2 + 1.0) * (y2 - 0.5)
    mo1 = tmp * 2.0 + 0.25
    mo2 = tmp * -0.5 + 1.5
    jt.sync([mo1, mo2], True)
    ref_tmp = (x2_np + 1.0) * (y2_np - 0.5)
    np.testing.assert_allclose(mo1.numpy(), ref_tmp * 2.0 + 0.25, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mo2.numpy(), ref_tmp * -0.5 + 1.5, rtol=1e-5, atol=1e-6)

    # fp16 & bf16 & f64 & int64 & NaN max/min (exercises vec8 uint4 path + scalar tail)
    h_out = (x2.float16() + b1.float16()).float32()
    np.testing.assert_allclose(h_out.numpy(), (x2_np.astype(np.float16) + b1_np.astype(np.float16)).astype(np.float32), rtol=1e-2, atol=1e-2)
    bf_out = (x2.bfloat16() * y2.bfloat16()).float32()
    np.testing.assert_allclose(bf_out.numpy(), x2_np * y2_np, rtol=2e-2, atol=2e-2)
    for n_vec8 in (16, 49, 13, 7):
        a16_np = rng.standard_normal(n_vec8).astype(np.float16)
        b16_np = rng.standard_normal(n_vec8).astype(np.float16)
        a16, b16 = jt.array(a16_np), jt.array(b16_np)
        jt.sync([a16, b16], True)
        c16 = a16 + b16
        c16.sync(True)
        assert c16.location() == 'device'
        np.testing.assert_array_equal(c16.numpy(), (a16_np + b16_np).astype(np.float16))
        if n_vec8 >= 13:
            # Unaligned slice (offset by 1 half = 2 bytes) falls back cleanly to scalar loop
            ua, ub = a16[1:], b16[1:]
            uc = ua + ub
            uc.sync(True)
            np.testing.assert_array_equal(uc.numpy(), (a16_np[1:] + b16_np[1:]).astype(np.float16))
    # Vectorized CUDA fast_gelu (16-byte uint4 path + tail + unaligned) for fp16/bf16/fp32
    with jt.no_grad():
        for dt_name, np_dt in (("float16", np.float16), ("float32", np.float32)):
            for n_g in (256, 313, 7):
                g_in_np = rng.standard_normal((1, n_g)).astype(np_dt)
                w_np = np.eye(n_g, dtype=np_dt)
                g_x = jt.array(g_in_np)
                g_w = jt.array(w_np)
                jt.sync([g_x, g_w], True)
                lin = jt.nn.matmul_transpose(g_x, g_w)
                g_out = jt.nn.gelu(lin)
                g_out.sync(True)
                assert g_out.location() == 'device'
                xf = g_in_np.astype(np.float32)
                import math as _m
                ref_g = (0.5 * xf * (1.0 + np.vectorize(_m.erf)(xf * 0.7071067811865476))).astype(np_dt)
                np.testing.assert_allclose(g_out.numpy().astype(np.float32), ref_g.astype(np.float32), rtol=1e-2 if dt_name == "float16" else 1e-5, atol=1e-2 if dt_name == "float16" else 1e-5)
    nan_a = jt.array(np.array([1.0, np.nan, -2.0], dtype=np.float32))
    nan_b = jt.array(np.array([2.0, 3.0, np.nan], dtype=np.float32))
    jt.sync_all(True)
    assert np.isnan(jt.maximum(nan_a, nan_b).numpy()[1])
    assert np.isnan(jt.maximum(nan_a, nan_b).numpy()[2])
    assert np.isnan(jt.minimum(nan_a, nan_b).numpy()[1])
    assert np.isnan(jt.minimum(nan_a, nan_b).numpy()[2])

    # 3. Strict vs Fast CUDA math policy via NVRTC
    a = jt.array(np.full(64, 1 + 2**-13, dtype=np.float32))
    b = jt.array(np.full(64, 1 - 2**-13, dtype=np.float32))
    c = jt.array(np.full(64, -1, dtype=np.float32))
    jt.sync_all(True)
    with jt.runtime.scope(cuda_kernel_math="strict"):
        ys = (a * b + c)
        ys.sync(True)
        np.testing.assert_array_equal(ys.numpy(), np.zeros(64, dtype=np.float32))
    with jt.runtime.scope(cuda_kernel_math="fast"):
        yf = (a * b + c)
        yf.sync(True)
        np.testing.assert_array_equal(yf.numpy(), np.full(64, -(2**-26), dtype=np.float32))

    # 4. CUDA Graph Capture compatibility with NVRTC kernel
    gx = jt.array(np.arange(256, dtype=np.float32))
    gy = jt.array(np.ones(256, dtype=np.float32))
    jt.sync_all(True)
    with jt.flag_scope(keep_graph=1):
        out_g = jt.tanh(gx * 0.01 + gy * 2.0)
        out_g.sync(True)  # allocate & preload module
        assert jt.core.graph_capture_begin()
        out_g.sync(False)
        gh = jt.core.graph_capture_end()
        assert gh != 0, f"Graph capture failed: {jt.core.graph_capture_refusal()}"
        jt.core.graph_launch(gh)
        jt.core.graph_wait()
        np.testing.assert_allclose(out_g.numpy(), np.tanh(np.arange(256, dtype=np.float32) * 0.01 + 2.0), rtol=1e-5, atol=1e-6)
        jt.core.graph_release(gh)

    stats1 = jt.core._nvrtc_stats()
    print("RESULT_JSON:" + json.dumps({"stats0": stats0, "stats1": stats1}))
"""
    res1 = _run_child(script_cold, {"JT_JIT_TIER": "1", "JT_USE_NVRTC": "1"}, clear_nvrtc_cache=True)
    assert res1["stats1"][1] > 0, f"Expected cold NVRTC compiles: {res1}"
    assert res1["stats1"][3] > 0, f"Expected NVRTC launches: {res1}"
    assert res1["stats1"][4] == 0, f"Unexpected NVRTC fallbacks: {res1}"

    # Child 2: warm disk cache reload (without clearing jit/)
    script_warm = """
import json, numpy as np, jittor as jt
with jt.flag_scope(use_cuda=1):
    x = jt.array(np.arange(1024, dtype=np.float32))
    y = jt.array(np.ones(1024, dtype=np.float32))
    jt.sync_all(True)
    z = jt.tanh(x * 1.25 + y * 0.75 - 0.1) + jt.sigmoid(x - y)
    z.sync(True)
    print("RESULT_JSON:" + json.dumps({"stats": jt.core._nvrtc_stats()}))
"""
    res2 = _run_child(script_warm, {"JT_JIT_TIER": "1", "JT_USE_NVRTC": "1"}, clear_nvrtc_cache=False)
    assert res2["stats"][2] >= 1, f"Expected NVRTC disk cache hit on warm run: {res2}"
    assert res2["stats"][3] >= 1, f"Expected NVRTC launch on warm run: {res2}"
