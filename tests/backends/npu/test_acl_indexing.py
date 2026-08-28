"""
getitem / setitem correctness suite for the ACL (Ascend) backend.

Each case computes a reference on CPU (use_acl=0 path / numpy) and compares the
ACL result for BOTH forward and backward. Run on a free NPU:

    ASCEND_RT_VISIBLE_DEVICES=6 python test_indexing.py

Exit code 0 = all pass. Failures are printed with the max abs error.
"""
import numpy as np
import jittor as jt
import pytest

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
    if not getattr(jt.compiler, "has_acl", 0):
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
    if not getattr(jt.compiler, "has_acl", 0):
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


if __name__ == "__main__":
    test_acl_indexing()
    raise SystemExit(1 if FAIL else 0)
