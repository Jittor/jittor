"""Ascend 950PR native ACL FP8 / MXFP8 hardware verification tests."""

import numpy as np
import pytest

import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.backends.acl.kernels._soc import supports_fp8


def _require_acl_fp8():
    if not (bool(getattr(jt.compiler, "has_acl", False)) and supports_fp8()):
        pytest.skip("requires Ascend950-class NPU with ACL backend")


def _cos_sim(a, b):
    a_f = np.asarray(a, dtype=np.float64).ravel()
    b_f = np.asarray(b, dtype=np.float64).ravel()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12))


class TestAclFP8Hardware:
    @pytest.fixture(autouse=True)
    def _check_hardware(self):
        _require_acl_fp8()
    def test_aclnn_cast_and_transpose_zero_fallback(self):
        with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            vals = np.array(
                [[0.0, 1.0, -2.5, 448.0], [0.001953125, -0.5, 12.0, -16.0]],
                dtype=np.float32,
            )
            for src_dt in ("float32", "float16", "bfloat16"):
                x = jt.array(vals, dtype=src_dt)
                for fp8_dt in ("float8_e4m3fn", "float8_e5m2"):
                    q = x.cast(fp8_dt)
                    q.sync()
                    assert str(q.dtype) == fp8_dt

                    qt = q.transpose((1, 0)).contiguous()
                    qt.sync()
                    assert tuple(qt.shape) == (4, 2)
                    assert str(qt.dtype) == fp8_dt

                    back = qt.cast(src_dt)
                    back.sync()
                    np.testing.assert_allclose(
                        back.float32().numpy(),
                        vals.T,
                        rtol=0.15 if fp8_dt == "float8_e5m2" else 0.08,
                        atol=1e-2,
                    )
            assert jt.core.backend_fallback_count() == before

    def test_dynamic_block_quant_and_block128_matmul_zero_fallback(self):
        rng = np.random.default_rng(123)
        M, K, N = 16, 256, 256
        x_np = rng.standard_normal((M, K)).astype(np.float32) * 0.5
        w_np = rng.standard_normal((N, K)).astype(np.float32) * 0.5
        b_np = rng.standard_normal((N,)).astype(np.float32) * 0.1
        ref = x_np @ w_np.T + b_np

        with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = jt.array(x_np, dtype="bfloat16")
            w = jt.array(w_np, dtype="bfloat16")
            bias = jt.array(b_np, dtype="bfloat16")

            qw, sw = jt.nn.dynamic_block_quant(w, row_block_size=128, col_block_size=128, dtype="float8_e4m3fn")
            qx, sx = jt.nn.dynamic_block_quant(x, row_block_size=1, col_block_size=128, dtype="float8_e4m3fn")
            qw.sync()
            sw.sync()
            qx.sync()
            sx.sync()
            assert str(qw.dtype) == "float8_e4m3fn"
            assert str(sw.dtype) == "float32"
            assert tuple(sw.shape) == (N // 128, K // 128)
            assert tuple(sx.shape) == (M, K // 128)

            y = jt.nn.fp8_linear(x, qw, sw, bias=bias, mode="block128", out_dtype="bfloat16")
            y.sync()
            assert jt.core.backend_fallback_count() == before
            assert str(y.dtype) == "bfloat16"
            assert tuple(y.shape) == (M, N)
            assert _cos_sim(y.float32().numpy(), ref) > 0.998

            # 3-D batched activation [B, S, K]
            x3d = x.reshape((2, M // 2, K))
            y3d = jt.nn.fp8_linear(x3d, qw, sw, bias=bias, mode="block128", out_dtype="bfloat16")
            y3d.sync()
            assert jt.core.backend_fallback_count() == before
            assert tuple(y3d.shape) == (2, M // 2, N)
            np.testing.assert_allclose(y3d.float32().numpy().reshape((M, N)), y.float32().numpy(), atol=1e-2)

    def test_dynamic_mx_quant_and_mxfp8_matmul_zero_fallback(self):
        rng = np.random.default_rng(456)
        M, K, N = 32, 256, 128
        x_np = rng.standard_normal((M, K)).astype(np.float32) * 0.5
        w_np = rng.standard_normal((N, K)).astype(np.float32) * 0.5
        ref = x_np @ w_np.T

        with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            x = jt.array(x_np, dtype="bfloat16")
            w = jt.array(w_np, dtype="bfloat16")

            qw, sw = jt.nn.dynamic_mx_quant(w, axis=-1, block_size=32, dtype="float8_e4m3fn")
            qw.sync()
            sw.sync()
            assert str(qw.dtype) == "float8_e4m3fn"
            assert str(sw.dtype) == "float8_e8m0fnu"
            assert tuple(sw.shape) == (N, K // 64, 2)

            y = jt.nn.fp8_linear(x, qw, sw, mode="mxfp8", out_dtype="bfloat16")
            y.sync()
            assert jt.core.backend_fallback_count() == before
            assert str(y.dtype) == "bfloat16"
            assert _cos_sim(y.float32().numpy(), ref) > 0.997

    def test_per_tensor_and_per_token_quant_matmul_zero_fallback(self):
        rng = np.random.default_rng(789)
        M, K, N = 16, 128, 64
        x_np = rng.standard_normal((M, K)).astype(np.float32) * 0.5
        w_np = rng.standard_normal((N, K)).astype(np.float32) * 0.5

        with jt.flag_scope(use_cuda=1), forbid_backend_fallbacks():
            before = jt.core.backend_fallback_count()
            # Per-token activation scale [M, 1], per-channel weight scale [1, N]
            sx_np = (np.max(np.abs(x_np), axis=1, keepdims=True) / 448.0).astype(np.float32)
            sw_np = (np.max(np.abs(w_np), axis=1, keepdims=True) / 448.0).astype(np.float32)

            qx = jt.array((x_np / sx_np), dtype="float32").cast("float8_e4m3fn")
            qw = jt.array((w_np / sw_np), dtype="float32").cast("float8_e4m3fn")
            sx = jt.array(sx_np, dtype="float32")
            sw = jt.array(sw_np.T, dtype="float32")  # [1, N]

            y = jt.nn.quant_matmul(
                qx, qw, sx, sw,
                out_dtype="bfloat16",
                trans_b=True,
                group_size=0,
            )
            y.sync()
            assert jt.core.backend_fallback_count() == before
            ref = x_np @ w_np.T
            assert _cos_sim(y.float32().numpy(), ref) > 0.998

            # Zero-copy strided transpose view passed directly as [K, N]
            y_view = jt.nn.quant_matmul(
                qx, qw.transpose((1, 0)), sx, sw,
                out_dtype="float32",
                trans_b=False,
                group_size=0,
            )
            y_view.sync()
            assert jt.core.backend_fallback_count() == before
            assert _cos_sim(y_view.numpy(), ref) > 0.998
