# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""ACL unit tests for Wave 5 / Wave 5D native mapped operators and fast paths."""

import numpy as np
import pytest

import jittor as jt
from _helpers import capability as _test_capability


pytestmark = pytest.mark.skipif(
    not _test_capability.check_accelerator("acl", backend=jt).enabled,
    reason="ACL backend is unavailable",
)


def _supports_bf16():
    from jittor.backends.acl.kernels._soc import supports_bfloat16
    return bool(supports_bfloat16())


def test_acl_mod_and_fmod_zero_fallback():
    with jt.flag_scope(use_cuda=1):
        a = jt.array(np.array([7.5, -7.5, 5.0, -5.0], dtype="float32"))
        b = jt.array(np.array([3.0, 3.0, -3.0, -3.0], dtype="float32"))
        np.testing.assert_allclose((a % b).numpy(), np.remainder(a.numpy(), b.numpy()), atol=1e-5)
        np.testing.assert_allclose(jt.fmod(a, b).numpy(), np.fmod(a.numpy(), b.numpy()), atol=1e-5)

        ai = jt.array(np.array([10, 17, 25, 0], dtype="int64"))
        bi = jt.array(np.array([4, 5, 7, 3], dtype="int64"))
        np.testing.assert_array_equal((ai % bi).numpy(), np.remainder(ai.numpy(), bi.numpy()))


def test_acl_contiguous_permute_and_strided_copy():
    rng = np.random.default_rng(42)
    with jt.flag_scope(use_cuda=1):
        arr = rng.standard_normal((4, 8, 12, 16)).astype("float32")
        x = jt.array(arr)
        # 4D NCHW -> NHWC pure permutation
        y_perm = x.permute(0, 2, 3, 1).contiguous()
        np.testing.assert_allclose(y_perm.numpy(), np.transpose(arr, (0, 2, 3, 1)), atol=1e-6)
        # Singleton dim permutation
        s_arr = rng.standard_normal((1, 8, 1, 16)).astype("float32")
        y_sing = jt.array(s_arr).permute(0, 3, 1, 2).contiguous()
        np.testing.assert_allclose(y_sing.numpy(), np.transpose(s_arr, (0, 3, 1, 2)), atol=1e-6)
        # Non-permutation strided slice (ContiguousCopyOpRunner)
        y_slice = x[:, :, :, :8].contiguous()
        np.testing.assert_allclose(y_slice.numpy(), arr[:, :, :, :8], atol=1e-6)


def test_acl_gelu_bf16_and_relu_mapped_unary():
    if not _supports_bf16():
        pytest.skip("bfloat16 unsupported on this SoC")
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((8, 64)).astype("float32")
    with jt.flag_scope(use_cuda=1), jt.no_grad():
        x_f = jt.array(arr)
        x_b = x_f.bfloat16()
        for approx in ("none", "tanh"):
            y_b = jt.nn.gelu(x_b, approximate=approx).float32().numpy()
            y_ref = jt.nn.gelu(x_f, approximate=approx).numpy()
            np.testing.assert_allclose(y_b, y_ref, rtol=2e-2, atol=2e-2)
        y_relu = jt.nn.relu(x_b).float32().numpy()
        np.testing.assert_allclose(y_relu, np.maximum(x_b.float32().numpy(), 0.0), atol=1e-5)


def test_acl_mapped_histc():
    with jt.flag_scope(use_cuda=1):
        vals = jt.array(np.array([0.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0], dtype="float32"))
        h = jt.histc(vals, bins=4, min=0, max=3)
        np.testing.assert_allclose(h.numpy(), np.array([1.0, 2.0, 1.0, 3.0], dtype="float32"), atol=1e-5)


def test_acl_mapped_grouped_matmul_contiguous_and_transposed():
    if not _supports_bf16():
        pytest.skip("bfloat16 unsupported on this SoC")
    rng = np.random.default_rng(13)
    E, M_e, K, N = 4, 3, 32, 16
    M = E * M_e
    x_np = rng.standard_normal((M, K)).astype("float32")
    w_nk_np = rng.standard_normal((E, N, K)).astype("float32")
    offs_np = np.array([3, 6, 9, 12], dtype="int64")

    ref = np.zeros((M, N), dtype="float32")
    start = 0
    for e, end in enumerate(offs_np):
        ref[start:end] = x_np[start:end] @ w_nk_np[e].T
        start = end

    with jt.flag_scope(use_cuda=1), jt.no_grad():
        x = jt.array(x_np).bfloat16()
        w_nk = jt.array(w_nk_np).bfloat16()
        offs = jt.array(offs_np, dtype="int64")
        # Zero-copy transposed 3D weight view (E, K, N) from (E, N, K)
        y_view = jt.core.ops.mapped_grouped_matmul(x, w_nk.transpose(1, 2), offs)
        np.testing.assert_allclose(y_view.float32().numpy(), ref, rtol=3e-2, atol=1e-1)
        # Contiguous 3D weight (E, K, N)
        y_contig = jt.core.ops.mapped_grouped_matmul(x, w_nk.transpose(1, 2).contiguous(), offs)
        np.testing.assert_allclose(y_contig.float32().numpy(), ref, rtol=3e-2, atol=1e-1)


def test_acl_mapped_addmm_fwd_and_bwd():
    rng = np.random.default_rng(19)
    x_np = rng.standard_normal((6, 32)).astype("float32")
    w_np = rng.standard_normal((16, 32)).astype("float32")
    b_np = rng.standard_normal((16,)).astype("float32")
    dy_np = rng.standard_normal((6, 16)).astype("float32")
    with jt.flag_scope(use_cuda=1):
        x = jt.array(x_np)
        w = jt.array(w_np)
        b = jt.array(b_np)
        y = jt.nn.linear(x, w, b)
        np.testing.assert_allclose(y.numpy(), x_np @ w_np.T + b_np, rtol=1e-4, atol=1e-4)
        gx, gw, gb = jt.grad((y * jt.array(dy_np)).sum(), [x, w, b])
        np.testing.assert_allclose(gx.numpy(), dy_np @ w_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gw.numpy(), dy_np.T @ x_np, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(gb.numpy(), dy_np.sum(axis=0), rtol=1e-4, atol=1e-4)


def test_acl_mapped_flash_attention_causal_and_realshift():
    if not _supports_bf16():
        pytest.skip("bfloat16 unsupported on this SoC")
    rng = np.random.default_rng(23)
    B, H, S, D = 1, 4, 16, 64
    q_np = rng.standard_normal((B, H, S, D)).astype("float32") * 0.2
    k_np = rng.standard_normal((B, H, S, D)).astype("float32") * 0.2
    v_np = rng.standard_normal((B, H, S, D)).astype("float32") * 0.2
    mask_np = np.zeros((1, 1, S, S), dtype="float32")
    mask_np[:, :, :, S // 2 :] = -1e4

    scores = np.matmul(q_np, np.swapaxes(k_np, -2, -1)) * (D ** -0.5) + mask_np
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)
    ref = np.matmul(probs, v_np)

    with jt.flag_scope(use_cuda=1), jt.no_grad():
        q = jt.array(q_np).bfloat16()
        k = jt.array(k_np).bfloat16()
        v = jt.array(v_np).bfloat16()
        m = jt.array(mask_np).bfloat16()
        y = jt.nn.scaled_dot_product_attention(q, k, v, attn_mask=m)
        np.testing.assert_allclose(y.float32().numpy(), ref, rtol=3e-2, atol=3e-2)
        y_causal = jt.nn.scaled_dot_product_attention(q, k, v, is_causal=True)
        assert tuple(y_causal.shape) == (B, H, S, D)


def test_acl_mapped_rms_norm_bfloat16_opt_in_and_contract():
    if not _supports_bf16():
        pytest.skip("bfloat16 unsupported on this SoC")
    rng = np.random.default_rng(29)
    x_np = rng.standard_normal((2, 8, 64)).astype("float32")
    w_np = rng.standard_normal((64,)).astype("float32")
    eps = 1e-6
    ref = (x_np / np.sqrt((x_np * x_np).mean(axis=-1, keepdims=True) + eps)) * w_np
    with jt.flag_scope(use_cuda=1), jt.no_grad():
        x = jt.array(x_np).bfloat16()
        w = jt.array(w_np).bfloat16()
        y = jt.core.ops.mapped_rms_norm(x, w, eps, True)
        np.testing.assert_allclose(y.float32().numpy(), ref, rtol=3e-2, atol=3e-2)
        with pytest.raises(RuntimeError):
            jt.core.ops.mapped_rms_norm(x, w, eps)


def test_acl_batch_norm_running_stats_ssa_and_bf16_eval():
    if not _supports_bf16():
        pytest.skip("bfloat16 unsupported on this SoC")
    rng = np.random.default_rng(31)
    x_np = rng.standard_normal((8, 16, 7, 7)).astype("float32")
    dy_np = rng.standard_normal((8, 16, 7, 7)).astype("float32")
    with jt.flag_scope(use_cuda=1):
        for dt in ("float32", "bfloat16"):
            bn = jt.nn.BatchNorm2d(16)
            if dt == "bfloat16":
                bn.weight.update(bn.weight.bfloat16())
                bn.bias.update(bn.bias.bfloat16())
                bn.running_mean.update(bn.running_mean.bfloat16())
                bn.running_var.update(bn.running_var.bfloat16())
            bn.train()
            x = jt.array(x_np).cast(dt)
            y = bn(x)
            # Read running stats before syncing y to verify pure SSA dependency
            rm = bn.running_mean.float32().numpy()
            rv = bn.running_var.float32().numpy()
            ref_rm = 0.1 * x_np.mean(axis=(0, 2, 3))
            ref_rv = 0.9 + 0.1 * x_np.var(axis=(0, 2, 3), ddof=1)
            tol = 1e-4 if dt == "float32" else 3e-2
            np.testing.assert_allclose(rm, ref_rm, rtol=tol, atol=tol)
            np.testing.assert_allclose(rv, ref_rv, rtol=tol, atol=tol)
            gx, gw, gb = jt.grad((y * jt.array(dy_np).cast(dt)).sum(), [x, bn.weight, bn.bias])
            assert tuple(gx.shape) == x_np.shape
            assert tuple(gw.shape) == (16,) and tuple(gb.shape) == (16,)
            bn.eval()
            with jt.no_grad():
                ye = bn(x.detach())
                assert tuple(ye.shape) == x_np.shape


def test_acl_setitem_add_reduction_and_getitem_leading_index_and_split():
    with jt.flag_scope(use_cuda=1):
        # SetitemOp ns_add reduction via multi-slice backward (ViT qkv[0..2])
        qkv = jt.randn((3, 4, 8, 16), dtype="float32")
        loss = (qkv[0] * 1.0 + qkv[1] * 2.0 + qkv[2] * 3.0).sum()
        gqkv = jt.grad(loss, [qkv])[0].numpy()
        np.testing.assert_allclose(gqkv[0], 1.0, atol=1e-5)
        np.testing.assert_allclose(gqkv[1], 2.0, atol=1e-5)
        np.testing.assert_allclose(gqkv[2], 3.0, atol=1e-5)

        # ViT CLS-token slice backward x[:, 0]
        tok = jt.randn((4, 10, 16), dtype="float32")
        gtok = jt.grad(tok[:, 0].sum(), [tok])[0].numpy()
        np.testing.assert_allclose(gtok[:, 0], 1.0, atol=1e-5)
        np.testing.assert_allclose(gtok[:, 1:], 0.0, atol=1e-5)

        # Leading-axis 1D int64 negative index gather + backward
        tbl = jt.arange(20).float32().reshape((5, 4))
        idx = jt.array(np.array([-1, 0, -2, 2], dtype="int64"))
        out = tbl[idx]
        np.testing.assert_allclose(
            out.numpy(),
            np.array([[16, 17, 18, 19], [0, 1, 2, 3], [12, 13, 14, 15], [8, 9, 10, 11]], dtype="float32"),
        )
        gtbl = jt.grad(out.sum(), [tbl])[0].numpy()
        np.testing.assert_allclose(gtbl[:, 0], np.array([1, 0, 1, 1, 1], dtype="float32"))

        # Single-chunk zero-copy split
        parts = jt.split(tbl, [5], dim=0)
        assert len(parts) == 1 and parts[0].var_ptr == tbl.var_ptr
