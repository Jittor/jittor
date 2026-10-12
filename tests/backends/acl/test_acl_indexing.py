"""
getitem / setitem correctness suite for the ACL (Ascend) backend.

Each case computes a reference on CPU (use_acl=0 path / numpy) and compares the
ACL result for BOTH forward and backward. Run on a free NPU:

    python tests/backends/acl/test_acl_indexing.py

Exit code 0 = all pass. Failures are printed with the max abs error.
"""

from _helpers import capability as _test_capability
import numpy as np
import jittor as jt
import pytest


def _soc_name():
    try:
        from jittor.backends.acl.kernels._soc import soc_name
        return soc_name()
    except Exception:
        return ""


def _acl_supports_bfloat16():
    try:
        from jittor.backends.acl.kernels._soc import supports_bfloat16
        return supports_bfloat16()
    except Exception:
        return True


def _acl_supports_fp32_cube():
    try:
        from jittor.backends.acl.kernels._soc import supports_fp32_cube
        return supports_fp32_cube()
    except Exception:
        return True


# Checks whose wrong result is a declared platform defect on this SoC, not a
# capability gap the test can guard at call time. On Ascend310P the in-place
# setitem graph optimization materializes the getitem backward's gradient value
# inside the zeros base allocation, and the base fill can leave unwritten
# bytes (see agent/manuals/known-issues.md).
_XFAIL_BY_SOC = {"Ascend310P": {"getbwd/int_index"}}

PASS, FAIL = 0, 0
def check(name, got, ref, tol=1e-4):
    global PASS, FAIL
    got = np.asarray(got); ref = np.asarray(ref)
    if got.shape != ref.shape:
        print(f"FAIL {name}: shape {got.shape} != ref {ref.shape}"); FAIL += 1; return
    err = np.abs(got.astype(np.float64) - ref.astype(np.float64)).max() if got.size else 0.0
    if err <= tol:
        print(f"ok   {name}  (err={err:.2e})"); PASS += 1
    else:
        for prefix, names in _XFAIL_BY_SOC.items():
            if name in names and _soc_name().startswith(prefix):
                print(f"xfail {name}: max err {err:.2e} (declared {prefix} defect)")
                return
        print(f"FAIL {name}: max err {err:.2e}"); FAIL += 1

def fwd_bwd(np_x, idx_fn, seed=0):
    """Return (acl_out, acl_gradx) and (ref_out, ref_gradx) using numpy autograd
    for the gather/scatter. idx_fn(x) returns x[...]. ref via numpy."""
    pass

# ---------- forward getitem cases (compare ACL vs numpy) ----------
def g(name, np_x, fn_jt, fn_np):
    x = jt.array(np_x)
    out = fn_jt(x)
    out.sync()
    check("get/"+name, out.numpy(), fn_np(np_x))

X2 = np.arange(24, dtype=np.float32).reshape(4, 6)
X3 = np.arange(120, dtype=np.float32).reshape(4, 5, 6)

# ---------- bool-mask getitem (full-shape and leading-shape) ----------
def gmask(name, np_x, np_mask):
    x = jt.array(np_x); m = jt.array(np_mask)
    out = x[m]; out.sync()
    check("get/"+name, out.numpy(), np_x[np_mask])

# ---------- getitem backward ----------
def gback(name, np_x, fn_jt, fn_np_setgrad):
    x = jt.array(np_x)
    out = fn_jt(x)
    loss = (out * out).sum()
    gx = jt.grad(loss, x); gx.sync()
    # ref grad: d/dx sum(x[idx]^2) = 2*x at idx positions, 0 elsewhere.
    # fn_np_setgrad(ref, val) must SCATTER val into ref at the same index (use
    # np.add.at for fancy indices, which return copies not views).
    ref = np.zeros_like(np_x, dtype=np.float64)
    fn_np_setgrad(ref, 2 * np_x.astype(np.float64))
    check("getbwd/"+name, gx.numpy(), ref)

# ---------- setitem ----------
def s(name, np_x, fn_jt, fn_np):
    x = jt.array(np_x)
    fn_jt(x)
    x.sync()
    ref = np_x.copy(); fn_np(ref)
    check("set/"+name, x.numpy(), ref)

def set_arr_jt(x): x[jt.array([0,2])] = jt.array(np.full((2,6),3.0,dtype=np.float32))
def set_arr_np(a): a[np.array([0,2])] = 3.0

# bool-mask setitem
def smask(name, np_x, np_mask, val):
    x = jt.array(np_x); m = jt.array(np_mask)
    x[m] = val
    x.sync()
    ref = np_x.copy(); ref[np_mask] = val
    check("set/"+name, x.numpy(), ref)


def smask_values(name, np_x, np_mask):
    values = np.arange(np.count_nonzero(np_mask), dtype=np_x.dtype)
    x = jt.array(np_x); m = jt.array(np_mask)
    x[m] = jt.array(values)
    x.sync()
    ref = np_x.copy(); ref[np_mask] = values
    check("set/"+name, x.numpy(), ref)


def test_acl_indexing():
    global PASS, FAIL
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    PASS = FAIL = 0
    with jt.flag_scope(use_acl=1):
        g("basic_slice", X2, lambda x: x[1:3], lambda a: a[1:3])
        g("full_slice", X2, lambda x: x[:], lambda a: a[:])
        g("step_slice", X2, lambda x: x[::2], lambda a: a[::2])
        g("col_slice", X2, lambda x: x[:, 1:4], lambda a: a[:, 1:4])
        g("int_index", X2, lambda x: x[2], lambda a: a[2])
        g("neg_int_index", X2, lambda x: x[-1], lambda a: a[-1])
        g("neg_slice", X2, lambda x: x[-3:-1], lambda a: a[-3:-1])
        g("int_then_slice", X3, lambda x: x[1, 1:3], lambda a: a[1, 1:3])
        g("slice_then_int", X3, lambda x: x[1:3, 2], lambda a: a[1:3, 2])
        g("two_int", X3, lambda x: x[1, 2], lambda a: a[1, 2])
        g("ellipsis", X3, lambda x: x[..., 1], lambda a: a[..., 1])
        g("step2_mid", X3, lambda x: x[:, ::2, :], lambda a: a[:, ::2, :])
        g(
            "int_array",
            X2,
            lambda x: x[jt.array([0, 2, 3])],
            lambda a: a[np.array([0, 2, 3])],
        )
        g(
            "int_array_2d",
            X2,
            lambda x: x[jt.array([0, 2]), jt.array([1, 3])],
            lambda a: a[np.array([0, 2]), np.array([1, 3])],
        )
        g(
            "slice_then_int_array",
            X2,
            lambda x: x[:, jt.array([1, 4]).int64()],
            lambda a: a[:, np.array([1, 4])],
        )
        gmask("mask_full", X2, X2 > 10)
        gmask("mask_row", X2, np.array([True, False, True, False]))
        gback(
            "slice",
            X2,
            lambda x: x[1:3],
            lambda ref, value: ref.__setitem__(np.s_[1:3], value[1:3]),
        )
        gback(
            "int_array",
            X2,
            lambda x: x[jt.array([0, 2, 3])],
            lambda ref, value: np.add.at(
                ref, np.array([0, 2, 3]), value[np.array([0, 2, 3])]
            ),
        )
        gback(
            "col_slice",
            X2,
            lambda x: x[:, 1:4],
            lambda ref, value: ref.__setitem__(np.s_[:, 1:4], value[:, 1:4]),
        )
        gback(
            "slice_then_int_array",
            X2,
            lambda x: x[:, jt.array([1, 4])],
            lambda ref, value: np.add.at(
                ref,
                (slice(None), np.array([1, 4])),
                value[:, np.array([1, 4])],
            ),
        )
        gback("int_index", X2, lambda x: x[2], lambda ref, value: ref.__setitem__(2, value[2]))
        s(
            "slice_scalar",
            X2.copy(),
            lambda x: x.__setitem__(slice(1, 3), 9.0),
            lambda a: a.__setitem__(slice(1, 3), 9.0),
        )
        s("int_scalar", X2.copy(), lambda x: x.__setitem__(2, 7.0), lambda a: a.__setitem__(2, 7.0))
        s(
            "col_slice_val",
            X2.copy(),
            lambda x: x.__setitem__((slice(None), slice(1, 4)), 5.0),
            lambda a: a.__setitem__((slice(None), slice(1, 4)), 5.0),
        )
        s("int_array_val", X2.copy(), set_arr_jt, set_arr_np)
        smask("mask_scalar", X2.copy(), X2 > 10, 0.0)
        smask("mask_scalar_empty", X2.copy(), np.zeros_like(X2, dtype=bool), 0.0)
        smask_values("mask_values", X2.copy(), X2 > 10)
    print(f"\n==== {PASS} passed, {FAIL} failed ====")
    assert FAIL == 0, f"{FAIL} ACL indexing checks failed"


def test_acl_getitem_preserves_async_dependencies():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    source = np.arange(24, dtype=np.float32).reshape(4, 6)
    with jt.flag_scope(use_acl=1):
        x = jt.array(source)
        sliced = x[:, 1:5]
        gathered = sliced[jt.array([3, 1])]
        actual = (gathered[:, 1:] * 2.0 + gathered[:, :-1]).numpy()
    expected_slice = source[:, 1:5]
    expected_gather = expected_slice[np.array([3, 1])]
    expected = expected_gather[:, 1:] * 2.0 + expected_gather[:, :-1]
    np.testing.assert_array_equal(actual, expected)


def test_acl_full_slice_uses_identity_forward_and_backward():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    source = np.arange(2 * 4 * 6, dtype=np.float32).reshape(2, 4, 6)
    weights = np.linspace(-1.0, 1.0, source.size, dtype=np.float32).reshape(
        source.shape
    )

    with jt.flag_scope(use_acl=1, use_cuda=1), jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
        x = jt.array(source)
        full_slice = x[:, :, :]
        gradient = jt.grad((full_slice * jt.array(weights)).sum(), x)
        full_slice.sync()
        gradient.sync()
        locations = full_slice.location(), gradient.location()
        actual, actual_gradient = jt.fetch_sync([full_slice, gradient])

    assert full_slice is not x
    assert locations == ("device", "device")
    np.testing.assert_array_equal(actual, source)
    np.testing.assert_array_equal(actual_gradient, weights)
    messages = [entry["msg"].lower() for entry in logs]
    assert not any("slicev2" in message for message in messages)
    assert not any("stridedsliceassignv2_grad" in message for message in messages)


def test_acl_flat_unit_stride_slices_are_storage_views():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")

    # ZeRO unflattens one dense model buffer into many parameter views.  The
    # slices must not materialize copies, otherwise total live memory grows by
    # the sum of every suffix-sized slice and real models OOM during init.
    elements = 16 * 1024 * 1024
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
        flat = jt.ones((elements,), dtype="float32")
        flat.sync()
        before = jt.core.device_memory_used(0)
        pieces = [flat[i * elements // 16:(i + 1) * elements // 16]
                  for i in range(16)]
        jt.sync(pieces)
        after = jt.core.device_memory_used(0)
        actual = pieces[-1][:4].numpy()

    assert after == before
    np.testing.assert_array_equal(actual, np.ones(4, dtype=np.float32))
    messages = [entry["msg"].lower() for entry in logs]
    assert not any("slicev2" in message for message in messages)


def test_disabling_device_execution_disables_acl_dispatch():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    source = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    mask = source.sum(axis=-1) > 20

    with jt.flag_scope(use_acl=1, use_cuda=0):
        assert jt.introspection.policy.runtime.use_cuda == 0
        assert jt.introspection.policy.runtime.use_cuda == 0
        output = jt.array(source)[jt.array(mask)]
        actual = output.numpy()

    np.testing.assert_array_equal(actual, source[mask])


def test_acl_contiguous_last_axis_slice_gradients_use_concat():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor.backends.acl.kernels.ops.getitem_op import _slice_zero_cache

    source_shape = (2, 3, 8)
    slices = (slice(0, 3), slice(2, 6), slice(5, 8), slice(0, 3))
    results = []
    expected = []

    dtypes = ("float16", "bfloat16", "float32")
    if not _acl_supports_bfloat16():
        # Ascend310P ships no DT_BFLOAT16 kernels; the declared-skip bucket
        # records this instead of an unexplained failure.
        print("xfail dtypes: bfloat16 unsupported on this SoC")
        dtypes = ("float16", "float32")

    with jt.flag_scope(use_acl=1, use_cuda=1), jt.log_capture_scope(
            log_v=0, log_vprefix="acl_op_exec.cc=100") as logs:
        for dtype in dtypes:
            for offset, last_slice in enumerate(slices):
                x = jt.zeros(source_shape, dtype=dtype)
                sliced_shape = x[..., last_slice].shape
                weights_np = (
                    np.arange(np.prod(sliced_shape), dtype=np.float32)
                    .reshape(sliced_shape) + offset + 1
                )
                weights = jt.array(weights_np)
                if dtype != "float32":
                    weights = weights.cast(dtype)
                gradient = jt.grad((x[..., last_slice] * weights).sum(), x)
                gradient.sync()
                location = gradient.location()
                (actual,) = jt.fetch_sync([gradient.float()])
                reference = np.zeros(source_shape, dtype=np.float32)
                reference[..., last_slice] = weights_np
                results.append((location, actual))
                expected.append(reference)

    for (location, actual), reference in zip(results, expected):
        assert location == "device"
        np.testing.assert_array_equal(actual, reference)
    cached = {(key[1], key[2]) for key in _slice_zero_cache}
    assert ((2, 3, 2), "float32") in cached
    assert ((2, 3, 5), "float32") in cached
    assert ((2, 3, 2), "float16") in cached
    assert ((2, 3, 5), "float16") in cached
    if _acl_supports_bfloat16():
        assert ((2, 3, 2), "bfloat16") in cached
        assert ((2, 3, 5), "bfloat16") in cached


def test_acl_slice_gradients_remain_lazy_and_zero_initialized():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    source = np.arange(48, dtype=np.float32).reshape(2, 4, 6)
    gradients = []
    expected = []
    with jt.flag_scope(use_acl=1):
        for offset in range(4):
            x = jt.array(source + offset)
            weights = jt.array(
                np.full((2, 2, 3), offset + 1, dtype=np.float32)
            )
            loss = (x[:, 1:3, 1::2] * weights).sum()
            gradients.append(jt.grad(loss, x))
            reference = np.zeros_like(source)
            reference[:, 1:3, 1::2] = offset + 1
            expected.append(reference)
        jt.sync(gradients)
    for actual, reference in zip(gradients, expected):
        np.testing.assert_array_equal(actual.numpy(), reference)


def test_acl_rfft_keeps_lazy_dft_constants_alive():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    if not _acl_supports_fp32_cube():
        # rfft lowers to an FP32 matmul against DFT constants. Ascend310P's
        # cube unit has no FP32 KEEP_DTYPE path, and the HF32 opt-in loses
        # more precision than this test's 1e-6 tolerance can accept.
        pytest.skip("rfft needs an FP32 cube path; unsupported on Ascend310P")
    samples = (
        ((8,), 910, -1, None),
        ((3, 6), 911, -1, None),
        ((7,), 912, -1, None),
        ((8,), 913, -1, "ortho"),
    )
    with jt.flag_scope(use_acl=1, use_cuda=1):
        for shape, seed, dim, norm in samples:
            source = np.random.RandomState(seed).uniform(
                -9, 9, size=shape
            ).astype("float32")
            x = jt.array(source)
            actual = jt.fft.rfft(x, dim=dim, norm=norm)
            expected = np.fft.rfft(x.numpy(), axis=dim, norm=norm)
            np.testing.assert_allclose(
                actual.numpy(), expected, rtol=1.3e-6, atol=1e-5
            )


if __name__ == "__main__":
    test_acl_indexing()
    raise SystemExit(1 if FAIL else 0)


def test_acl_bool_slice_assignment_matches_numpy():
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    source = np.array([[True, False, True, True],
                       [False, True, False, True]], dtype=bool)
    expected = source.copy()
    expected[..., :2] = False
    with jt.flag_scope(use_acl=1):
        actual = jt.array(source)
        actual[..., :2] = False
        actual.sync()
        np.testing.assert_array_equal(actual.numpy(), expected)
        assert actual.dtype == "bool"


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_acl_mixed_tensor_index_preserves_base_and_broadcast_gradient(dtype):
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor._runtime.fallback import forbid_backend_fallbacks

    original = np.arange(72, dtype=np.float32).reshape(1, 2, 9, 4) / 8
    values = np.array([[[[-2., -1., 1., 2.]]]], dtype=np.float32)
    positions = np.array([0, 2, 5], dtype=np.int32)
    weights = np.arange(72, dtype=np.float32).reshape(original.shape) / 64
    expected = original.copy()
    expected[:, :, positions, :] = values
    base_grad = weights.copy()
    base_grad[:, :, positions, :] = 0
    value_grad = weights[:, :, positions, :].sum(axis=(1, 2), keepdims=True)

    with jt.flag_scope(use_acl=1, use_cuda=1), forbid_backend_fallbacks():
        x, v = jt.array(original).cast(dtype), jt.array(values).cast(dtype)
        index = (slice(None), slice(None), jt.array(positions), slice(None))
        updated = jt.setitem(x, index, v)
        gx, gv = jt.grad((updated * jt.array(weights).cast(dtype)).sum(), [x, v])
        jt.sync([updated, gx, gv])
        assert all(t.location() == "device" for t in (updated, gx, gv))
        actual, unchanged, dx, dv = jt.fetch_sync([updated, x, gx, gv])
    np.testing.assert_array_equal(actual, expected.astype(dtype))
    np.testing.assert_array_equal(unchanged, original.astype(dtype))
    np.testing.assert_allclose(dx, base_grad, rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(dv, value_grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("index", [
    (slice(None), slice(None), slice(16, 48, 2)),
    (None, slice(None), 1, Ellipsis, slice(16, 48)),
    (slice(1, None), Ellipsis, slice(None, None, 2)),
    (slice(None), slice(None), None, slice(None)),
], ids=["last_axis_step", "newaxis_integer_ellipsis", "leading_offset", "full_slice_newaxis"])
def test_acl_inference_basic_index_shares_storage_and_follows_base(dtype, index):
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor._runtime.fallback import forbid_backend_fallbacks

    source = (np.arange(2 * 4 * 64).reshape(2, 4, 64) / 8).astype(dtype)
    expected = source[index]
    byte_offset = (expected.__array_interface__["data"][0]
                   - source.__array_interface__["data"][0])
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.no_grad(), forbid_backend_fallbacks():
        x = jt.array(source)
        x.sync()
        selected = x[index]
        selected.sync()
        assert selected is not x
        assert selected.location() == "device"
        assert selected._storage_address == x._storage_address + byte_offset
        assert selected._storage_offset() == byte_offset // source.itemsize
        assert list(selected._storage_strides()) == [
            stride // source.itemsize for stride in expected.strides
        ]
        np.testing.assert_array_equal(selected.numpy(), expected)
        # Updating the materialized base must be visible through the already
        # materialized selection, including full slices with an inserted axis.
        x.assign(x + 1)
        x.sync()
        np.testing.assert_array_equal(selected.numpy(), expected + 1)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_acl_inference_basic_index_consumes_noncontiguous_input_and_output(dtype):
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor._runtime.fallback import forbid_backend_fallbacks

    source = (np.arange(2 * 4 * 64).reshape(2, 4, 64) / 8).astype(dtype)
    expected = source[..., ::2][..., 8:24:2]
    with jt.flag_scope(use_acl=1, use_cuda=1), jt.no_grad(), forbid_backend_fallbacks():
        x = jt.array(source)
        strided = x[..., ::2]
        selected = strided[..., 8:24:2]
        jt.sync([x, strided, selected])
        assert selected._storage_address == x._storage_address + 16 * source.itemsize
        assert list(selected._storage_strides()) == [256, 64, 4]
        assert not selected._storage_is_contiguous()
        elementwise = -(selected + selected)
        # ACL Concat is an opaque CodeOp consumer and must materialize the
        # strided input at its existing dense-storage boundary.
        concatenated = jt.concat([selected, selected], dim=-1)
        jt.sync([elementwise, concatenated])
        assert elementwise.location() == concatenated.location() == "device"
        actual, actual_concat = jt.fetch_sync([elementwise, concatenated])
    np.testing.assert_array_equal(actual, -(expected + expected))
    np.testing.assert_array_equal(actual_concat, np.concatenate([expected, expected], axis=-1))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("strided", [False, True], ids=["contiguous", "strided"])
def test_acl_inference_index_copy_preserves_inputs(dtype, index_dtype, dim, strided, monkeypatch):
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor._runtime.fallback import forbid_backend_fallbacks
    from jittor.backends.acl.kernels.ops import setitem_op

    original = np.arange(60, dtype=np.float32).reshape(3, 4, 5).astype(dtype)
    positions = np.asarray([0, -1], dtype=index_dtype)
    value_shape = list(original.shape)
    value_shape[dim] = len(positions)
    values = (-np.arange(np.prod(value_shape), dtype=np.float32) - 1).reshape(value_shape).astype(dtype)
    numpy_index = [slice(None)] * original.ndim
    numpy_index[dim] = positions
    expected = original.copy()
    expected[tuple(numpy_index)] = values

    routed = []
    original_call = setitem_op.IndexCopyACL.__call__

    def record(self, base, axis, index, source):
        routed.append((axis, str(index.dtype)))
        return original_call(self, base, axis, index, source)

    monkeypatch.setattr(setitem_op.IndexCopyACL, "__call__", record)
    fallback_before = jt.core.backend_fallback_count()
    with jt.runtime.scope(use_cuda=1, transpose_storage_view=1), jt.no_grad(), forbid_backend_fallbacks():
        if strided:
            base = jt.array(original.transpose(1, 0, 2).copy()).transpose(1, 0, 2)
            source = jt.array(values.transpose(1, 0, 2).copy()).transpose(1, 0, 2)
            assert not base._storage_is_contiguous()
            assert not source._storage_is_contiguous()
        else:
            base, source = jt.array(original), jt.array(values)
        indices = jt.array(positions, dtype=index_dtype)
        assert str(indices.dtype) == index_dtype
        index = [slice(None)] * original.ndim
        index[dim] = indices
        updated = jt.setitem(base, tuple(index), source)
        jt.sync([updated, base, source])
        assert updated is not base
        assert updated._storage_address != base._storage_address
        assert tuple(updated.shape) == original.shape
        assert str(updated.dtype) == dtype
        assert all(value.location() == "device" for value in (updated, base, source))
        actual, unchanged_base, unchanged_source = jt.fetch_sync([updated, base, source])
    assert routed == [(dim, index_dtype)]
    assert jt.core.backend_fallback_count() == fallback_before
    # A separate output address alone is insufficient: CANN's dim=0
    # out-of-place API was observed to also overwrite the original base.
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(unchanged_base, original)
    np.testing.assert_array_equal(unchanged_source, values)


@pytest.mark.parametrize("case", [
    "empty_index", "broadcast_source", "partial_slice", "multiple_indices",
    "scalar_source", "training",
])
def test_acl_index_copy_declines_other_assignment_forms(case, monkeypatch):
    if not _test_capability.check_accelerator('acl', backend=jt).enabled:
        pytest.skip("ACL backend is unavailable")
    from jittor.backends.acl.kernels.ops import setitem_op

    def unexpected(*args, **kwargs):
        pytest.fail("assignment outside the inference subset reached IndexCopy")

    monkeypatch.setattr(setitem_op.IndexCopyACL, "__call__", unexpected)
    with jt.runtime.scope(use_cuda=1), jt.no_grad():
        base = jt.empty((3, 4, 5), dtype="float32")
        index = jt.array([0, 2], dtype="int32")
        value = jt.empty((3, 2, 5), dtype="float32")
        slices = (slice(None), index, slice(None))
        if case == "empty_index":
            slices = (slice(None), jt.empty((0,), dtype="int32"), slice(None))
            value = jt.empty((3, 0, 5), dtype="float32")
        elif case == "broadcast_source":
            value = jt.empty((1, 2, 1), dtype="float32")
        elif case == "partial_slice":
            slices = (slice(0, 2), index, slice(None))
            value = jt.empty((2, 2, 5), dtype="float32")
        elif case == "multiple_indices":
            slices = (index, index, slice(None))
            value = jt.empty((2, 2, 5), dtype="float32")
        elif case == "scalar_source":
            value = 3.0
        if case == "training":
            with jt.enable_grad():
                assert setitem_op._single_axis_index_copy(base, slices, value) is None
        else:
            assert setitem_op._single_axis_index_copy(base, slices, value) is None
