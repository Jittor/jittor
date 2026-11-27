"""Native FP8 (float8_e4m3fn, float8_e5m2, float8_e8m0fnu) dtype & CPU op tests."""

import math
import numpy as np
import pytest

import jittor as jt


class TestFP8NativeDtypes:
    def test_nanostring_metadata_and_finfo(self):
        for name, is_signed, fmax in (
            ("float8_e4m3fn", True, 448.0),
            ("float8_e5m2", True, 57344.0),
            ("float8_e8m0fnu", False, 2.0**127),
        ):
            ns = jt.NanoString(name)
            assert ns.dsize() == 1
            assert ns.is_float()
            assert ns.is_floating_point()
            assert not ns.is_int()
            assert not ns.is_bool()
            assert not ns.is_complex()

            x = jt.empty((4,), dtype=name)
            assert str(x.dtype) == name
            assert x.is_floating_point()
            assert x.is_signed() is is_signed
            assert not x.is_complex()

            info = jt.finfo(name)
            assert info.bits == 8
            assert info.max == fmax

    def test_e4m3fn_bit_exact_encoding_and_decoding_cpu(self):
        with jt.flag_scope(use_cuda=0):
            # Exact PyTorch 2.13 OCP E4M3FN bit vectors:
            # 0.0 -> 0x00, -0.0 -> 0x80, 1.0 -> 0x38, -2.5 -> 0xC2,
            # 448.0 -> 0x7E, 500.0 (saturate) -> 0x7E, -500.0 -> 0xFE,
            # +inf (saturate) -> 0x7E, -inf -> 0xFE, min subnormal 2^-9 -> 0x01,
            # underflow 2^-10 -> 0x00 (ties to even), 3*2^-10 -> 0x02, nan -> 0x7F.
            vals = np.array(
                [
                    0.0, -0.0, 1.0, -2.5, 448.0, 500.0, -500.0,
                    np.inf, -np.inf, 2.0**-9, 2.0**-10, 3.0 * 2.0**-10, np.nan,
                ],
                dtype=np.float32,
            )
            q = jt.array(vals, dtype="float32").cast("float8_e4m3fn")
            bits = q.reinterpret_view(q.shape, "uint8").numpy()
            expected_bits = np.array(
                [0x00, 0x80, 0x38, 0xC2, 0x7E, 0x7E, 0xFE, 0x7E, 0xFE, 0x01, 0x00, 0x02, 0x7F],
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(bits, expected_bits)

            decoded = q.numpy()
            assert decoded.dtype == np.float32
            assert decoded[0] == 0.0 and not math.copysign(1.0, decoded[0]) < 0
            assert decoded[1] == 0.0 and math.copysign(1.0, decoded[1]) < 0
            assert decoded[2] == 1.0
            assert decoded[3] == -2.5
            assert decoded[4] == 448.0
            assert decoded[5] == 448.0
            assert decoded[6] == -448.0
            assert decoded[7] == 448.0
            assert decoded[8] == -448.0
            assert decoded[9] == 2.0**-9
            assert decoded[10] == 0.0
            assert decoded[11] == 2.0 * 2.0**-9
            assert np.isnan(decoded[12])

            # Scalar item() decoding
            assert q[3].item() == pytest.approx(-2.5)
            # Direct cast back to float32 / bfloat16 / float16
            np.testing.assert_allclose(q[:9].float32().numpy(), decoded[:9], atol=0, rtol=0)
            np.testing.assert_allclose(q[:5].bfloat16().numpy(), decoded[:5], atol=1e-2)
            np.testing.assert_allclose(q[:5].float16().numpy(), decoded[:5], atol=1e-2)

    def test_e5m2_bit_exact_encoding_and_decoding_cpu(self):
        with jt.flag_scope(use_cuda=0):
            vals = np.array(
                [0.0, -0.0, 1.0, -2.5, 448.0, 500.0, 57344.0, 65504.0, -65504.0, np.inf, -np.inf, 2.0**-16, np.nan],
                dtype=np.float32,
            )
            q = jt.float8_e5m2(jt.array(vals, dtype="float32"))
            bits = q.reinterpret_view(q.shape, "uint8").numpy()
            expected_bits = np.array(
                [0x00, 0x80, 0x3C, 0xC1, 0x5F, 0x60, 0x7B, 0x7C, 0xFC, 0x7C, 0xFC, 0x01, 0x7F],
                dtype=np.uint8,
            )
            np.testing.assert_array_equal(bits, expected_bits)

            decoded = q.numpy()
            assert decoded[2] == 1.0
            assert decoded[3] == -2.5
            assert decoded[4] == 448.0
            assert decoded[5] == 512.0
            assert decoded[6] == 57344.0
            assert np.isposinf(decoded[7]) and np.isneginf(decoded[8])
            assert np.isposinf(decoded[9]) and np.isneginf(decoded[10])
            assert decoded[11] == 2.0**-16
            assert np.isnan(decoded[12])

    def test_e8m0fnu_bit_exact_powers_of_two_cpu(self):
        with jt.flag_scope(use_cuda=0):
            raw = np.array([0, 1, 126, 127, 128, 135, 254, 255], dtype=np.uint8)
            s = jt.array(raw, dtype="uint8").reinterpret_view(raw.shape, "float8_e8m0fnu")
            decoded = s.numpy()
            assert decoded[0] == np.float32(2.0**-127)
            assert decoded[1] == np.float32(2.0**-126)
            assert decoded[2] == np.float32(0.5)
            assert decoded[3] == np.float32(1.0)
            assert decoded[4] == np.float32(2.0)
            assert decoded[5] == np.float32(256.0)
            assert decoded[6] == np.float32(2.0**127)
            assert np.isnan(decoded[7])
            assert s[3].item() == pytest.approx(1.0)

    def test_dtype_promotion_and_structural_ops_cpu(self):
        with jt.flag_scope(use_cuda=0):
            a = jt.array([1.0, 2.0], dtype="float8_e4m3fn")
            b = jt.array([3.0, 4.0], dtype="float8_e4m3fn")
            c = jt.concat([a, b], dim=0)
            assert str(c.dtype) == "float8_e4m3fn"
            np.testing.assert_array_equal(c.numpy(), np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))

            w = jt.where(jt.array([True, False]), a, b)
            assert str(w.dtype) == "float8_e4m3fn"
            np.testing.assert_array_equal(w.numpy(), np.array([1.0, 4.0], dtype=np.float32))

            # Mixed promotion never accidentally lands on float16
            mixed_bf16 = a + jt.array([1.0, 1.0], dtype="bfloat16")
            assert str(mixed_bf16.dtype) == "bfloat16"
            mixed_e5 = a + jt.array([1.0, 1.0], dtype="float8_e5m2")
            assert str(mixed_e5.dtype) == "float32"

    def test_cpu_quant_and_quant_matmul_reference(self):
        with jt.flag_scope(use_cuda=0):
            rng = np.random.default_rng(42)
            x_np = rng.standard_normal((4, 128)).astype(np.float32) * 0.5
            w_np = rng.standard_normal((128, 128)).astype(np.float32) * 0.5
            ref = x_np @ w_np.T

            x = jt.array(x_np, dtype="bfloat16")
            w = jt.array(w_np, dtype="bfloat16")

            # Block-128x128 FP8
            qw, sw = jt.nn.dynamic_block_quant(w, row_block_size=128, col_block_size=128, dtype="float8_e4m3fn")
            assert str(qw.dtype) == "float8_e4m3fn"
            assert tuple(sw.shape) == (1, 1)
            y_blk = jt.nn.fp8_linear(x, qw, sw, mode="block128").float32().numpy()
            cos_blk = float(np.dot(y_blk.ravel(), ref.ravel()) / (np.linalg.norm(y_blk) * np.linalg.norm(ref)))
            assert cos_blk > 0.998

            # MXFP8 1x32
            qw_mx, sw_mx = jt.nn.dynamic_mx_quant(w, axis=-1, block_size=32, dtype="float8_e4m3fn")
            assert str(qw_mx.dtype) == "float8_e4m3fn"
            assert str(sw_mx.dtype) == "float8_e8m0fnu"
            assert tuple(sw_mx.shape) == (128, 2, 2)
            y_mx = jt.nn.fp8_linear(x, qw_mx, sw_mx, mode="mxfp8").float32().numpy()
            cos_mx = float(np.dot(y_mx.ravel(), ref.ravel()) / (np.linalg.norm(y_mx) * np.linalg.norm(ref)))
            assert cos_mx > 0.997
