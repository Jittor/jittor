"""Strict ACL backend tests for Wave 6 native mapped ops and optimizations."""
import math
import numpy as np
import pytest
import jittor as jt
from jittor import nn


pytestmark = pytest.mark.skipif(
    not getattr(jt.compiler, "has_acl", False),
    reason="ACL backend required",
)


def _assert_close(actual, expected, rtol, atol, label=""):
    a = np.asarray(actual, dtype=np.float32)
    e = np.asarray(expected, dtype=np.float32)
    diff = np.abs(a - e)
    tol = atol + rtol * np.abs(e)
    assert np.all(diff <= tol), (
        f"{label}: max_abs={diff.max():.6f}, max_rel={(diff / np.maximum(np.abs(e), 1e-8)).max():.6f}"
    )


def _ref_sdpa_fwd_bwd_np(q, k, v, g, is_causal=False):
    # Exact FP64 NumPy reference for SDPA forward + backward (including GQA).
    b, hq, s, d = q.shape
    hkv = k.shape[1]
    rep = hq // hkv
    q64 = q.astype(np.float64)
    k64 = np.repeat(k.astype(np.float64), rep, axis=1)
    v64 = np.repeat(v.astype(np.float64), rep, axis=1)
    g64 = g.astype(np.float64)
    scale = 1.0 / math.sqrt(d)
    scores = np.matmul(q64, np.swapaxes(k64, -1, -2)) * scale
    if is_causal:
        mask = np.triu(np.ones((s, s), dtype=bool), k=1)
        scores = np.where(mask, -1e9, scores)
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    out = np.matmul(probs, v64)

    dv_exp = np.matmul(np.swapaxes(probs, -1, -2), g64)
    dp = np.matmul(g64, np.swapaxes(v64, -1, -2))
    dot = (dp * probs).sum(axis=-1, keepdims=True)
    ds = probs * (dp - dot) * scale
    if is_causal:
        ds = np.where(mask, 0.0, ds)
    dq = np.matmul(ds, k64)
    dk_exp = np.matmul(np.swapaxes(ds, -1, -2), q64)
    dk = dk_exp.reshape(b, hkv, rep, s, d).sum(axis=2)
    dv = dv_exp.reshape(b, hkv, rep, s, d).sum(axis=2)
    return out.astype(np.float32), dq.astype(np.float32), dk.astype(np.float32), dv.astype(np.float32)


@pytest.mark.parametrize("dtype", ["bfloat16", "float16"])
@pytest.mark.parametrize("hq,hkv,seq,causal", [
    (12, 12, 197, False),
    (12, 4, 64, True),
])
def test_flash_attention_bf16_fp16_gqa_training(dtype, hq, hkv, seq, causal):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(123)
    q_np = (rng.standard_normal((2, hq, seq, 64)) * 0.2).astype(np.float32)
    k_np = (rng.standard_normal((2, hkv, seq, 64)) * 0.2).astype(np.float32)
    v_np = (rng.standard_normal((2, hkv, seq, 64)) * 0.2).astype(np.float32)
    g_np = (rng.standard_normal((2, hq, seq, 64)) * 0.2).astype(np.float32)
    out_ref, dq_ref, dk_ref, dv_ref = _ref_sdpa_fwd_bwd_np(q_np, k_np, v_np, g_np, is_causal=causal)

    with jt.flag_scope(use_cuda=1, missing_grad_error=1):
        # Target BF16 / FP16 FlashAttentionACL path
        q = jt.array(q_np).cast(dtype)
        k = jt.array(k_np).cast(dtype)
        v = jt.array(v_np).cast(dtype)
        g = jt.array(g_np).cast(dtype)
        out = nn.scaled_dot_product_attention(
            q, k, v, is_causal=causal, enable_gqa=(hq != hkv)
        )
        dq, dk, dv = jt.grad((out * g).sum(), [q, k, v])
        jt.sync_all()

        assert float(np.abs(dq_ref).max()) > 1e-3
        assert float(np.abs(dk_ref).max()) > 1e-3
        assert float(np.abs(dv_ref).max()) > 1e-3
        atol = 3e-2 if dtype == "bfloat16" else 1e-2
        rtol = 3e-2 if dtype == "bfloat16" else 1e-2
        _assert_close(out.float32().numpy(), out_ref, rtol, atol, f"sdpa_out_{dtype}")
        _assert_close(dq.float32().numpy(), dq_ref, rtol, atol, f"sdpa_dq_{dtype}")
        _assert_close(dk.float32().numpy(), dk_ref, rtol, atol, f"sdpa_dk_{dtype}")
        _assert_close(dv.float32().numpy(), dv_ref, rtol, atol, f"sdpa_dv_{dtype}")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
@pytest.mark.parametrize("ceil_mode", [False, True])
def test_maxpool_infer_bit_exact_with_indices_variant(dtype, ceil_mode):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(7)
    x_np = rng.standard_normal((2, 16, 15, 15)).astype(np.float32)
    with jt.flag_scope(use_cuda=1):
        x = jt.array(x_np).cast(dtype)
        with jt.no_grad():
            y_infer = nn.max_pool2d(x, 3, stride=2, padding=1, ceil_mode=ceil_mode)
            y_idx, _ = nn.max_pool2d(
                x, 3, stride=2, padding=1,
                return_indices=True, ceil_mode=ceil_mode,
            )
            jt.sync_all()
        np.testing.assert_array_equal(y_infer.float32().numpy(), y_idx.float32().numpy())


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("has_bias,groups,stride,padding,dilation", [
    (True, 1, 1, 1, 1),
    (False, 1, 2, 1, 1),
    (True, 4, 1, 1, 1),
    (False, 1, 1, 2, 2),
])
def test_mapped_conv2d_forward_and_backward(dtype, has_bias, groups, stride, padding, dilation):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(42)
    x_np = (rng.standard_normal((2, 8, 9, 9)) * 0.2).astype(np.float32)
    w_np = (rng.standard_normal((12, 8 // groups, 3, 3)) * 0.2).astype(np.float32)
    b_np = (rng.standard_normal((12,)) * 0.2).astype(np.float32) if has_bias else None
    g_np = None

    with jt.flag_scope(use_cuda=1):
        # Reference via ConvACL (CodeOp)
        from jittor.backends.acl.kernels.ops.conv_op import ConvACL
        xr = jt.array(x_np).cast(dtype)
        wr = jt.array(w_np).cast(dtype)
        br = jt.array(b_np).cast(dtype) if has_bias else None
        yr = ConvACL()(xr, wr, br, stride, padding, dilation, groups)
        g_np = (rng.standard_normal(tuple(yr.shape)) * 0.2).astype(np.float32)
        gr = jt.array(g_np).cast(dtype)
        ref_targets = [xr, wr, br] if has_bias else [xr, wr]
        ref_grads = jt.grad((yr * gr).sum(), ref_targets)
        jt.sync_all()

        # Target MappedConv2dOp via nn.conv2d
        x = jt.array(x_np).cast(dtype)
        w = jt.array(w_np).cast(dtype)
        b = jt.array(b_np).cast(dtype) if has_bias else None
        y = nn.conv2d(x, w, b, stride=stride, padding=padding, dilation=dilation, groups=groups)
        g = jt.array(g_np).cast(dtype)
        targets = [x, w, b] if has_bias else [x, w]
        grads = jt.grad((y * g).sum(), targets)
        jt.sync_all()

        atol = 2e-2 if dtype == "bfloat16" else 1e-4
        rtol = 2e-2 if dtype == "bfloat16" else 1e-4
        _assert_close(y.float32().numpy(), yr.float32().numpy(), rtol, atol, "conv_y")
        for idx, (ga, ge) in enumerate(zip(grads, ref_grads)):
            _assert_close(ga.float32().numpy(), ge.float32().numpy(), rtol, atol, f"conv_g{idx}")

        # Verify stem-layer conv where input has stop_grad (skips dx cleanly)
        x_sg = jt.array(x_np).cast(dtype).stop_grad()
        w2 = jt.array(w_np).cast(dtype)
        y2 = nn.conv2d(x_sg, w2, None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        dw2 = jt.grad((y2 * g).sum(), w2)
        jt.sync_all()
        _assert_close(dw2.float32().numpy(), ref_grads[1].float32().numpy(), rtol, atol, "conv_stem_dw")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_mapped_batch_norm_eval_and_train(dtype):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(19)
    x_np = (rng.standard_normal((4, 16, 7, 7)) * 0.5 + 0.2).astype(np.float32)
    w_np = (rng.standard_normal((16,)) * 0.2 + 1.0).astype(np.float32)
    b_np = (rng.standard_normal((16,)) * 0.2).astype(np.float32)
    rm_np = (rng.standard_normal((16,)) * 0.1).astype(np.float32)
    rv_np = (np.abs(rng.standard_normal((16,))) + 0.5).astype(np.float32)
    g_np = (rng.standard_normal((4, 16, 7, 7)) * 0.2).astype(np.float32)

    with jt.flag_scope(use_cuda=1):
        from jittor.backends.acl.kernels.ops.norms_op import BatchNormACL
        # Eval mode (no_grad + with_grad)
        xr = jt.array(x_np).cast(dtype)
        wr = jt.array(w_np).cast(dtype)
        br = jt.array(b_np).cast(dtype)
        rmr = jt.array(rm_np).cast(dtype).stop_grad()
        rvr = jt.array(rv_np).cast(dtype).stop_grad()
        gr = jt.array(g_np).cast(dtype)
        yr_eval = BatchNormACL(1e-5, 0.1, False)(xr, wr, br, rmr, rvr)
        eval_ref_grads = jt.grad((yr_eval * gr).sum(), [xr, wr, br])
        jt.sync_all()

        x = jt.array(x_np).cast(dtype)
        w = jt.array(w_np).cast(dtype)
        b = jt.array(b_np).cast(dtype)
        rm = jt.array(rm_np).cast(dtype).stop_grad()
        rv = jt.array(rv_np).cast(dtype).stop_grad()
        g = jt.array(g_np).cast(dtype)
        with jt.no_grad():
            y_ng = nn.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5)
            jt.sync_all()
        y_ev = nn.batch_norm(x, rm, rv, w, b, training=False, eps=1e-5)
        eval_grads = jt.grad((y_ev * g).sum(), [x, w, b])
        jt.sync_all()

        atol = 2e-2 if dtype == "bfloat16" else 1e-4
        rtol = 2e-2 if dtype == "bfloat16" else 1e-4
        _assert_close(y_ng.float32().numpy(), yr_eval.float32().numpy(), rtol, atol, "bn_eval_ng")
        _assert_close(y_ev.float32().numpy(), yr_eval.float32().numpy(), rtol, atol, "bn_eval")
        for idx, (ga, ge) in enumerate(zip(eval_grads, eval_ref_grads)):
            _assert_close(ga.float32().numpy(), ge.float32().numpy(), rtol, atol, f"bn_eval_g{idx}")

        # Train mode via nn.BatchNorm2d (tests running stats + num_batches_tracked)
        bn = nn.BatchNorm2d(16, eps=1e-5, momentum=0.1)
        bn.weight.update(jt.array(w_np).cast(dtype))
        bn.bias.update(jt.array(b_np).cast(dtype))
        bn.running_mean.update(jt.array(rm_np).cast(dtype))
        bn.running_var.update(jt.array(rv_np).cast(dtype))
        bn.train()

        yr_tr, rm_ref, rv_ref = BatchNormACL(1e-5, 0.1, True).call_with_running_stats(
            xr, wr, br, rmr, rvr
        )
        tr_ref_grads = jt.grad((yr_tr * gr).sum(), [xr, wr, br])
        jt.sync_all()

        x_tr = jt.array(x_np).cast(dtype)
        y_tr = bn(x_tr)
        tr_grads = jt.grad((y_tr * g).sum(), [x_tr, bn.weight, bn.bias])
        jt.sync_all()
        _assert_close(y_tr.float32().numpy(), yr_tr.float32().numpy(), rtol, atol, "bn_train_y")
        _assert_close(bn.running_mean.float32().numpy(), rm_ref.float32().numpy(), rtol, atol, "bn_rm")
        _assert_close(bn.running_var.float32().numpy(), rv_ref.float32().numpy(), rtol, atol, "bn_rv")
        assert int(bn.num_batches_tracked.item()) == 1
        for idx, (ga, ge) in enumerate(zip(tr_grads, tr_ref_grads)):
            _assert_close(ga.float32().numpy(), ge.float32().numpy(), rtol, atol, f"bn_tr_g{idx}")

        # Run 2 more steps to verify cached device scalar increment for num_batches_tracked
        for expected_step in (2, 3):
            _ = bn(jt.array(x_np).cast(dtype))
            jt.sync_all()
            assert int(bn.num_batches_tracked.item()) == expected_step


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("norm_shape", [(64,), (4, 8)])
def test_mapped_layer_norm_infer_and_train(dtype, norm_shape):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(31)
    x_shape = (3, 5) + norm_shape
    x_np = (rng.standard_normal(x_shape) * 0.5).astype(np.float32)
    w_np = (rng.standard_normal(norm_shape) * 0.2 + 1.0).astype(np.float32)
    b_np = (rng.standard_normal(norm_shape) * 0.2).astype(np.float32)
    g_np = (rng.standard_normal(x_shape) * 0.2).astype(np.float32)

    with jt.flag_scope(use_cuda=1):
        from jittor.backends.acl.kernels.ops.norms_op import LayerNormACL
        xr = jt.array(x_np).cast(dtype)
        wr = jt.array(w_np).cast(dtype)
        br = jt.array(b_np).cast(dtype)
        gr = jt.array(g_np).cast(dtype)
        yr = LayerNormACL(norm_shape, 1e-5, True)(xr, wr, br)
        ref_grads = jt.grad((yr * gr).sum(), [xr, wr, br])
        jt.sync_all()

        x = jt.array(x_np).cast(dtype)
        w = jt.array(w_np).cast(dtype)
        b = jt.array(b_np).cast(dtype)
        g = jt.array(g_np).cast(dtype)
        with jt.no_grad():
            y_ng = nn.layer_norm(x, norm_shape, w, b, eps=1e-5)
            jt.sync_all()
        y = nn.layer_norm(x, norm_shape, w, b, eps=1e-5)
        grads = jt.grad((y * g).sum(), [x, w, b])
        jt.sync_all()

        atol = 2e-2 if dtype == "bfloat16" else 1e-4
        rtol = 2e-2 if dtype == "bfloat16" else 1e-4
        _assert_close(y_ng.float32().numpy(), yr.float32().numpy(), rtol, atol, "ln_ng")
        _assert_close(y.float32().numpy(), yr.float32().numpy(), rtol, atol, "ln_y")
        for idx, (ga, ge) in enumerate(zip(grads, ref_grads)):
            _assert_close(ga.float32().numpy(), ge.float32().numpy(), rtol, atol, f"ln_g{idx}")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("act", ["relu", "gelu_none", "gelu_tanh"])
def test_mapped_unary_relu_gelu_training(dtype, act):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(55)
    x_np = (rng.standard_normal((4, 32)) * 1.5).astype(np.float32)
    g_np = (rng.standard_normal((4, 32)) * 0.5).astype(np.float32)

    with jt.flag_scope(use_cuda=1):
        from jittor.backends.acl.kernels.ops.relu_op import ReLUACL
        from jittor.backends.acl.kernels.ops.gelu_op import GeluACL
        xr = jt.array(x_np).cast(dtype)
        gr = jt.array(g_np).cast(dtype)
        if act == "relu":
            yr = ReLUACL()(xr)
        elif act == "gelu_none":
            yr = GeluACL()(xr, "none")
        else:
            yr = GeluACL()(xr, "tanh")
        dxr = jt.grad((yr * gr).sum(), xr)
        jt.sync_all()

        x = jt.array(x_np).cast(dtype)
        g = jt.array(g_np).cast(dtype)
        if act == "relu":
            y = nn.relu(x)
        elif act == "gelu_none":
            y = nn.gelu(x, approximate="none")
        else:
            y = nn.gelu(x, approximate="tanh")
        dx = jt.grad((y * g).sum(), x)
        jt.sync_all()

        atol = 2e-2 if dtype == "bfloat16" else 1e-5
        rtol = 2e-2 if dtype == "bfloat16" else 1e-5
        _assert_close(y.float32().numpy(), yr.float32().numpy(), rtol, atol, f"{act}_y")
        _assert_close(dx.float32().numpy(), dxr.float32().numpy(), rtol, atol, f"{act}_dx")


@pytest.mark.parametrize("dtype", ["float32", "bfloat16", "float16"])
def test_mapped_concat_strided_views_and_broadcast(dtype):
    if dtype == "bfloat16" and not jt.backends.acl.kernels._soc.supports_bfloat16():
        pytest.skip("SoC lacks bfloat16")
    rng = np.random.default_rng(99)
    x_np = rng.standard_normal((2, 8, 16, 64)).astype(np.float32)
    with jt.flag_scope(use_cuda=1), jt.no_grad():
        x = jt.array(x_np).cast(dtype)
        # RoPE rotate_half pattern: (-x[..., 32:], x[..., :32])
        x1 = x[..., :32]
        x2 = x[..., 32:]
        assert not x1._storage_is_contiguous()
        y_strided = jt.concat([-x2, x1], dim=-1)
        y_contig = jt.concat([-x2.contiguous(), x1.contiguous()], dim=-1)
        jt.sync_all()
        np.testing.assert_array_equal(y_strided.float32().numpy(), y_contig.float32().numpy())

        # Strided step-2 slice views
        xs = x[:, :, ::2, 8:40]
        assert not xs._storage_is_contiguous()
        yt_strided = jt.concat([xs, xs], dim=-1)
        yt_contig = jt.concat([xs.contiguous(), xs.contiguous()], dim=-1)
        jt.sync_all()
        np.testing.assert_array_equal(yt_strided.float32().numpy(), yt_contig.float32().numpy())

        # Zero-stride broadcast input must still produce exact result
        b = jt.array([1.0, 2.0]).cast(dtype).reshape(1, 2).broadcast((4, 2))
        c = jt.array(rng.standard_normal((4, 3)).astype(np.float32)).cast(dtype)
        yb = jt.concat([b, c], dim=-1)
        yb_ref = jt.concat([b.contiguous(), c], dim=-1)
        jt.sync_all()
        np.testing.assert_array_equal(yb.float32().numpy(), yb_ref.float32().numpy())


def test_tolist_cache_correctness_and_invalidation():
    with jt.flag_scope(use_cuda=1), jt.no_grad():
        x = jt.array([10, 20, 30], dtype="int64")
        first = x.tolist()
        assert first == [10, 20, 30]
        # Mutating returned Python list must not corrupt cached entry
        first[0] = -1
        assert x.tolist() == [10, 20, 30]

        # Rebinding via += / update / assign / setitem must invalidate
        x += 1
        assert x.tolist() == [11, 21, 31]

        x[1] = 99
        assert x.tolist() == [11, 99, 31]

        x.update(jt.array([1, 2, 3], dtype="int64"))
        assert x.tolist() == [1, 2, 3]

        x.assign(jt.array([4, 5, 6], dtype="int64"))
        assert x.tolist() == [4, 5, 6]

        # View invalidation when base is mutated
        base = jt.array([7, 8, 9, 10], dtype="int64")
        view = base[:2]
        assert view.tolist() == [7, 8]
        base[0] = 77
        assert view.tolist() == [77, 8]

        # Raw .data escape disables caching cleanly
        raw = jt.array([5, 6], dtype="int32")
        assert raw.tolist() == [5, 6]
        arr = raw.data
        arr[0] = 42
        assert raw.tolist() == [42, 6]


def test_acl_rsqrt_softplus_tril_gradients():
    from jittor.backends.acl.kernels.ops.silu_op import RsqrtACL, SoftplusACL
    from jittor.backends.acl.kernels.ops.triu_op import TrilACL

    rng = np.random.default_rng(77)
    x_pos = (np.abs(rng.standard_normal((4, 8))) + 0.5).astype(np.float32)
    x_mat = rng.standard_normal((6, 6)).astype(np.float32)
    g_pos = rng.standard_normal((4, 8)).astype(np.float32)
    g_mat = rng.standard_normal((6, 6)).astype(np.float32)

    with jt.flag_scope(use_cuda=1, missing_grad_error=1):
        # RsqrtACL
        x1 = jt.array(x_pos)
        g1 = jt.array(g_pos)
        y1 = RsqrtACL()(x1)
        dx1 = jt.grad((y1 * g1).sum(), x1)
        jt.sync_all()
        expected_dx1 = -0.5 * g_pos * (x_pos ** -1.5)
        _assert_close(dx1.numpy(), expected_dx1, 1e-4, 1e-4, "rsqrt_dx")

        # SoftplusACL
        x2 = jt.array(x_pos)
        y2 = SoftplusACL()(x2, 1.0, 20.0)
        dx2 = jt.grad((y2 * g1).sum(), x2)
        jt.sync_all()
        expected_dx2 = g_pos / (1.0 + np.exp(-x_pos))
        _assert_close(dx2.numpy(), expected_dx2, 1e-4, 1e-4, "softplus_dx")

        # TrilACL
        x3 = jt.array(x_mat)
        g3 = jt.array(g_mat)
        y3 = TrilACL()(x3, 0)
        dx3 = jt.grad((y3 * g3).sum(), x3)
        jt.sync_all()
        expected_dx3 = np.tril(g_mat, 0)
        _assert_close(dx3.numpy(), expected_dx3, 1e-5, 1e-5, "tril_dx")

