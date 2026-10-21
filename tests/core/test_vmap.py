# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# Native jt.vmap (python/jittor/vmap.py): builds one real batched op graph
# instead of looping over the batch in Python. This repo previously had no
# native vmap at all -- only an approximate, CPU-only, loop/stack-based
# `torch.vmap` compat shim in a completely separate namespace
# (compat/torch/installers/numerical.py).
import unittest
from importlib import import_module
from unittest.mock import patch

import numpy as np

import jittor as jt


class TestVmap(unittest.TestCase):
    def test_elementwise(self):
        x = jt.array(np.random.RandomState(0).randn(5, 3).astype("float32"))
        out = jt.vmap(lambda v: jt.sin(v) * 2 + 1)(x)
        ref = np.sin(x.numpy()) * 2 + 1
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-5)

    def test_reduce_and_grad_composition(self):
        # vmap(f) where f reduces to a per-example scalar, then jt.grad
        # through the whole batched graph, must give each example its own
        # gradient (not the sum collapsed across the batch).
        x = jt.array(np.random.RandomState(1).randn(4, 3).astype("float32"))
        x.requires_grad = True
        y = jt.vmap(lambda v: (v * v).sum())(x)
        self.assertEqual(tuple(y.shape), (4,))
        g = jt.grad(y.sum(), [x])[0]
        np.testing.assert_allclose(g.numpy(), 2 * x.numpy(), atol=1e-5)

    def test_cast_entry_points_and_plain_calls(self):
        casts = (
            lambda value, dtype: value.cast(dtype),
            lambda value, dtype: value.astype(dtype),
            lambda value, dtype: jt.cast(value, dtype),
        )
        # complex128 is not registered in this core baseline. Cast batching
        # must preserve native support rather than claim a new dtype family.
        pairs = (
            ("float32", "float64"),
            ("float32", "complex64"),
            ("complex64", "float32"),
            ("int32", "float32"),
            ("bool", "float32"),
        )
        with jt.flag_scope(auto_convert_64_to_32=0):
            for source, target in pairs:
                data = (np.arange(8).reshape(4, 2) / 4 - 0.5).astype(source)
                if "complex" in source:
                    data += 0.25j
                expected = data.real if "complex" in source else data
                expected = expected.astype(target)
                for entry, cast in enumerate(casts):
                    with self.subTest(source=source, target=target, entry=entry):
                        calls = []

                        def convert(value):
                            calls.append(tuple(value.shape))
                            return cast(value, target)

                        x = jt.array(data, dtype=source)
                        mapped = jt.vmap(convert)(x)
                        self.assertEqual(calls, [(2,)])
                        for result in (mapped, cast(x, target)):
                            self.assertEqual(str(result.dtype), target)
                            np.testing.assert_array_equal(result.numpy(), expected)

    def test_cast_keyword_arguments_and_unmapped_scalar_array(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3)
        x = jt.array(data)
        jt.vmap(lambda value: value)(x).sync()
        casts = (
            lambda value: value.cast(op="float64"),
            lambda value: value.astype(op="float64"),
            lambda value: jt.cast(value, op="float64"),
            lambda value: jt.Var.cast(value, op="float64"),
            lambda value: jt.cast(x=value, op="float64"),
        )
        for entry, cast in enumerate(casts):
            with self.subTest(entry=entry):
                for result in (cast(x), jt.vmap(cast)(x)):
                    self.assertEqual(str(result.dtype), "float64")
                    np.testing.assert_array_equal(
                        result.numpy(), data.astype("float64")
                    )
        # The free function accepts promotion from scalars/arrays. Its
        # fallback must not accidentally become a Var method descriptor.
        for source in (3, 1.25, [1, 2], np.array([0.25, -0.5], dtype="float32")):
            for result in (
                jt.cast(source, op="float64"),
                jt.cast(x=source, op="float64"),
            ):
                self.assertEqual(str(result.dtype), "float64")
                np.testing.assert_array_equal(
                    result.numpy(), np.asarray(source, dtype="float64")
                )
        with self.assertRaises(TypeError):
            jt.Var.cast(3, op="float64")

    def test_cast_nested_vmap_nondefault_axes(self):
        data = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 8
        calls = []

        def convert(value):
            calls.append(tuple(value.shape))
            return value.astype("float64")

        inner = jt.vmap(convert, in_dims=-1, out_dims=0)
        result = jt.vmap(inner, in_dims=1, out_dims=-1)(jt.array(data))
        self.assertEqual(calls, [(2,)])
        self.assertEqual(str(result.dtype), "float64")
        np.testing.assert_array_equal(
            result.numpy(), data.transpose(2, 0, 1).astype("float64")
        )

    def test_cast_gradient_and_second_derivative(self):
        data = np.arange(6, dtype=np.float32).reshape(2, 3) / 4 - 0.5
        x = jt.array(data)
        values = jt.vmap(lambda row: (row.cast("float64") ** 2).sum())(x)
        gradient = jt.grad(values.sum(), x)
        second = jt.grad(gradient.sum(), x)
        self.assertEqual(str(gradient.dtype), "float32")
        np.testing.assert_allclose(gradient.numpy(), 2 * data, atol=1e-6)
        np.testing.assert_allclose(second.numpy(), np.full_like(data, 2), atol=1e-6)

    def test_vmap_functional_jvp_vjp(self):
        data = np.asarray([[0.4, -0.7], [0.2, 0.3]], dtype="float32")
        direction = np.asarray([[0.5, 0.2], [-0.4, 0.6]], dtype="float32")
        for operation in (jt.autograd.jvp, jt.autograd.vjp):
            for create_graph in (False, True):
                with self.subTest(
                    operation=operation.__name__, create_graph=create_graph
                ):
                    calls = []

                    def transform(value, tangent):
                        calls.append(tuple(value.shape))
                        return operation(
                            lambda row: row**3,
                            value,
                            tangent,
                            create_graph=create_graph,
                        )

                    x = jt.array(data)
                    values, derivatives = jt.vmap(transform)(x, jt.array(direction))
                    self.assertEqual(calls, [(2,)])
                    np.testing.assert_allclose(values.numpy(), data**3, atol=1e-6)
                    np.testing.assert_allclose(
                        derivatives.numpy(), 3 * data**2 * direction, atol=1e-6
                    )
                    if create_graph:
                        second = jt.grad(derivatives.sum(), x)
                        np.testing.assert_allclose(
                            second.numpy(), 6 * data * direction, atol=1e-6
                        )
                    else:
                        # Native detach stops the producer's gradient path,
                        # but the detached result remains a grad-enabled leaf.
                        # Check isolation from the original input, not Torch's
                        # different requires_grad metadata convention.
                        plain_x = jt.array(data)
                        plain_values, plain_derivatives = operation(
                            lambda row: row**3,
                            plain_x,
                            jt.array(direction),
                            create_graph=False,
                        )
                        for mapped_result, plain_result in (
                            (values, plain_values),
                            (derivatives, plain_derivatives),
                        ):
                            np.testing.assert_allclose(
                                mapped_result.numpy(), plain_result.numpy(), atol=1e-6
                            )
                            mapped_gradient = jt.grad(mapped_result.sum(), x)
                            plain_gradient = jt.grad(plain_result.sum(), plain_x)
                            np.testing.assert_array_equal(
                                mapped_gradient.numpy(), np.zeros_like(data)
                            )
                            np.testing.assert_array_equal(
                                plain_gradient.numpy(), np.zeros_like(data)
                            )

    def test_vmap_functional_strict_still_rejected(self):
        x = jt.ones((2, 3))
        for operation in (jt.autograd.jvp, jt.autograd.vjp):
            with self.subTest(operation=operation.__name__):
                with self.assertRaisesRegex(NotImplementedError, "strict"):
                    jt.vmap(
                        lambda row: operation(
                            lambda value: value**2, row, row, strict=True
                        )
                    )(x)

    def test_native_maxmin_values_shape_gradient_and_keepdims(self):
        data = np.asarray(
            [[[1, 4], [3, 2]], [[6, -1], [2, 5]], [[0, 7], [8, 1]]],
            dtype="float32",
        )
        for name in ("min", "max"):
            reference = getattr(np, name)
            for axis in (None, 0, -1):
                for keepdims in (False, True):
                    with self.subTest(name=name, axis=axis, keepdims=keepdims):
                        x = jt.array(data)
                        result = jt.vmap(
                            lambda row: getattr(row, name)(dim=axis, keepdims=keepdims)
                        )(x)
                        self.assertIsInstance(result, jt.Var)
                        gradient = jt.grad(result.sum(), x)
                        for output in (result, gradient):
                            output.sync()
                            if jt.flags.use_cuda:
                                self.assertEqual(output.location(), "device")
                        expected = np.stack(
                            [
                                reference(row, axis=axis, keepdims=keepdims)
                                for row in data
                            ]
                        )
                        expected_gradient = np.stack(
                            [
                                row == reference(row, axis=axis, keepdims=True)
                                for row in data
                            ]
                        ).astype("float32")
                        self.assertEqual(tuple(result.shape), expected.shape)
                        np.testing.assert_array_equal(result.numpy(), expected)
                        np.testing.assert_array_equal(
                            gradient.numpy(), expected_gradient
                        )

    def test_maxmin_retains_optional_paired_result(self):
        # Exercise the legacy paired-result contract without loading the
        # optional Torch frontend into a native test process.
        data = np.asarray([[1, 4], [3, 2], [0, 5]], dtype="float32")
        x = jt.array(data)
        jt.vmap(lambda row: row)(x)
        batching = import_module("jittor.vmap")
        native_max = batching._ORIG_REDUCE["max"]

        def paired_max(value, dim, keepdims):
            result = native_max(value, dim, keepdims)
            return result, jt.zeros(result.shape, dtype="int32")

        with patch.dict(batching._ORIG_REDUCE, {"max": paired_max}):
            values, indices = jt.vmap(lambda row: row.max(0))(x)
        np.testing.assert_array_equal(values.numpy(), data.max(axis=1))
        np.testing.assert_array_equal(indices.numpy(), np.zeros(3, dtype="int32"))

    def test_matmul_with_shared_weight(self):
        # The single most common vmap pattern: an unbatched (shared) weight
        # matrix applied to a batched vector. Plain jt.matmul broadcasting
        # has no notion of "this leading axis is a batch axis I introduced",
        # so this needs the dedicated matmul batching rule (_apply_matmul),
        # not just generic broadcasting.
        w = jt.array(np.random.RandomState(2).randn(3, 3).astype("float32"))
        xb = jt.array(np.random.RandomState(3).randn(6, 3).astype("float32"))
        out = jt.vmap(lambda v: jt.matmul(w, v))(xb)
        ref = np.einsum("ij,bj->bi", w.numpy(), xb.numpy())
        np.testing.assert_allclose(out.numpy(), ref, atol=1e-4)

    def test_nested_vmap(self):
        x = jt.array(np.random.RandomState(4).randn(2, 3, 4).astype("float32"))
        out = jt.vmap(jt.vmap(lambda v: v * 2 + 1))(x)
        np.testing.assert_allclose(out.numpy(), x.numpy() * 2 + 1, atol=1e-5)

    def test_getitem_under_vmap(self):
        y = jt.array(np.random.RandomState(5).randn(5, 6).astype("float32"))
        out = jt.vmap(lambda v: v[1:4])(y)
        np.testing.assert_allclose(out.numpy(), y.numpy()[:, 1:4])

    def test_qr_batching_rule(self):
        # qr/svd/svdvals/eigh/inv are already (..., M, N)-batched numpy_code
        # ops with no op-specific batch-axis handling of their own, so
        # vmap's extra leading physical axis should be accepted as just one
        # more leading batch dim for free (_apply_linalg).
        a = jt.array(np.random.RandomState(6).randn(3, 4, 4).astype("float32"))

        def qrf(x):
            return jt.linalg.qr(x)

        q, r = jt.vmap(qrf)(a)
        self.assertEqual(tuple(q.shape), (3, 4, 4))
        recon = np.einsum("bij,bjk->bik", q.numpy(), r.numpy())
        np.testing.assert_allclose(recon, a.numpy(), atol=1e-3, rtol=1e-3)

    def test_random_different_gives_independent_draws(self):
        r = jt.vmap(lambda v: v + jt.random(v.shape), randomness="different")(
            jt.zeros((8, 3))
        )
        rn = r.numpy()
        self.assertFalse(np.allclose(rn[0], rn[1]))

    def test_random_same_gives_identical_draws(self):
        r = jt.vmap(lambda v: v + jt.random(v.shape), randomness="same")(
            jt.zeros((8, 3))
        )
        rn = r.numpy()
        np.testing.assert_array_equal(rn[0], rn[1])

    def test_unmapped_call_is_unaffected(self):
        # install_batching_patches() monkeypatches jt.Var/jt entry points
        # process-wide the first time jt.vmap() is used; a call with no
        # BatchedVar involved must still route to the same underlying op
        # (not silently get misdispatched) -- allclose, not exact equality,
        # since jt.matmul's own backend (MKL here) legitimately differs from
        # numpy's BLAS at the ~1e-7 float32 ulp level regardless of vmap.
        x = jt.array(np.random.RandomState(7).randn(4, 4).astype("float32"))
        y = jt.array(np.random.RandomState(8).randn(4, 4).astype("float32"))
        np.testing.assert_array_equal((x + y).numpy(), x.numpy() + y.numpy())
        np.testing.assert_allclose(
            jt.matmul(x, y).numpy(), x.numpy() @ y.numpy(), atol=1e-5, rtol=1e-5
        )

    def test_unsupported_op_raises(self):
        # cumsum now has a real batching rule (test_cumsum_batching_rule
        # below) -- cumprod is the same "scan" op family, genuinely still
        # unsupported, so this keeps testing the fail-fast contract itself
        # rather than asserting something now false.
        with self.assertRaises(NotImplementedError):
            jt.vmap(lambda v: v.cumprod(0))(
                jt.array(np.zeros((3, 4), dtype="float32"))
            )

    def test_cumsum_batching_rule(self):
        # jittor-core-gaps.md §3.4: axis-sensitive single-input rule.
        x = jt.array(np.random.RandomState(9).randn(5, 4).astype("float32"))
        out = jt.vmap(lambda v: v.cumsum(0))(x)
        np.testing.assert_allclose(out.numpy(), np.cumsum(x.numpy(), axis=1), atol=1e-5)

        # free-function spelling
        out2 = jt.vmap(lambda v: jt.cumsum(v, dim=0))(x)
        np.testing.assert_allclose(out2.numpy(), np.cumsum(x.numpy(), axis=1), atol=1e-5)

        # default dim (None -> last logical axis)
        out3 = jt.vmap(lambda v: v.cumsum())(x)
        np.testing.assert_allclose(out3.numpy(), np.cumsum(x.numpy(), axis=1), atol=1e-5)

        # negative dim, on a rank-3 logical input
        y = jt.array(np.random.RandomState(10).randn(3, 2, 4).astype("float32"))
        out4 = jt.vmap(lambda v: v.cumsum(-1))(y)
        np.testing.assert_allclose(out4.numpy(), np.cumsum(y.numpy(), axis=-1), atol=1e-5)

        # nested (2-level) vmap
        z = jt.array(np.random.RandomState(11).randn(2, 3, 4).astype("float32"))
        out5 = jt.vmap(jt.vmap(lambda v: v.cumsum(0)))(z)
        np.testing.assert_allclose(out5.numpy(), np.cumsum(z.numpy(), axis=2), atol=1e-5)

        # grad composition
        g = jt.array(np.random.RandomState(12).randn(5, 4).astype("float32"))
        g.requires_grad = True
        out6 = jt.vmap(lambda v: v.cumsum(0))(g)
        loss = out6.sum()
        grad = jt.grad(loss, [g])[0]
        # d(sum(cumsum(x)))/dx_i = (n - i), independent of batch
        n = g.shape[1]
        expect = np.tile(np.arange(n, 0, -1, dtype="float32"), (g.shape[0], 1))
        np.testing.assert_allclose(grad.numpy(), expect, atol=1e-4)

    def test_solve_batching_rule(self):
        # jittor-core-gaps.md §3.4: two-input linalg, max-level dispatch.
        # `b` is shaped (...,M,K) (a proper matrix RHS, K=2 here), not the
        # bare (...,M) vector-stack form jt.linalg.solve's own docstring
        # implies is supported: this numpy version's `np.linalg.solve`
        # gufunc only takes the "b as a vector" path when b.ndim==1 exactly
        # (a single, wholly unbatched vector) -- a batched (...,M) vector
        # stack is misparsed as a 2-D (m,n) CORE matrix, not a batch of
        # vectors, and raises a core-dimension mismatch independent of vmap
        # entirely (confirmed with a plain, non-vmap jt.linalg.solve call).
        # This is a real, separate solve() limitation, out of scope here --
        # this rule only has to correctly batch whatever solve() already
        # handles correctly, and (...,M,K) is that shape.
        rng = np.random.RandomState(13)

        def rand_spd(n):
            m = rng.randn(n, n).astype("float32")
            return m @ m.T + n * np.eye(n, dtype="float32")

        # both operands batched
        a = np.stack([rand_spd(3) for _ in range(4)])
        b = rng.randn(4, 3, 2).astype("float32")
        A, B = jt.array(a), jt.array(b)
        out = jt.vmap(lambda av, bv: jt.linalg.solve(av, bv))(A, B)
        expect = np.stack([np.linalg.solve(a[i], b[i]) for i in range(4)])
        np.testing.assert_allclose(out.numpy(), expect, atol=1e-3, rtol=1e-3)

        # `a` shared (unbatched), `b` batched -- the doc's own "shared
        # parameter" composition concern (confirmed independently to work
        # on plain, non-vmap jt.linalg.solve first).
        a0 = rand_spd(3)
        A0 = jt.array(a0)
        out2 = jt.vmap(lambda bv: jt.linalg.solve(A0, bv))(B)
        expect2 = np.stack([np.linalg.solve(a0, b[i]) for i in range(4)])
        np.testing.assert_allclose(out2.numpy(), expect2, atol=1e-3, rtol=1e-3)

        # grad composition
        B.requires_grad = True
        out3 = jt.vmap(lambda av, bv: jt.linalg.solve(av, bv))(A, B)
        loss = out3.sum()
        gB = jt.grad(loss, [B])[0]
        ones = np.ones((3, 2), dtype="float32")
        expect_g = np.stack([np.linalg.solve(a[i].T, ones) for i in range(4)])
        np.testing.assert_allclose(gB.numpy(), expect_g, atol=1e-3, rtol=1e-3)

    def test_numpy_code_raises_clear_error_under_vmap(self):
        def f(v):
            return jt.numpy_code(v.shape, v.dtype, [v],
                                  lambda np_, data: np_.copyto(data["outputs"][0], data["inputs"][0]))
        with self.assertRaisesRegex(NotImplementedError, "numpy_code"):
            jt.vmap(f)(jt.array(np.zeros((3, 4), dtype="float32")))

    def test_sparse_raises_clear_error_under_vmap(self):
        # No code change here -- SparseVar/spmm already hard isinstance(x,
        # jt.Var)-check their arguments and raise before any op runs, which
        # already meets the fail-fast bar (jittor-core-gaps.md §3.4). This
        # locks that existing behavior with an explicit test.
        def f(v):
            return jt.sparse.SparseVar(
                jt.array([[0, 0], [1, 1]]), v, jt.NanoVector([2, 2]))
        with self.assertRaises(TypeError):
            jt.vmap(f)(jt.array(np.ones((3, 2), dtype="float32")))


class TestVmapPatchTiming(unittest.TestCase):
    # jittor-core-gaps.md §3.4: install_batching_patches() runs lazily, only
    # inside vmap()'s own body -- a reference to a linalg/Var function saved
    # BEFORE any jt.vmap() call in the process stays bound to the pre-patch
    # function even after vmap() runs elsewhere. Making the install eager
    # (running it once, unconditionally, at the end of `import jittor`) was
    # tried and reverted: this project's own alias-composition system
    # (`_runtime/composition.py`'s `_make_inplace_alias`/`_publish`) gives
    # `jittor.Var.matmul`, `nn.matmul`, `jittor.matmul`, `jittor.grad`, etc.
    # a SINGLE shared object identity across every entry point, asserted
    # directly by tests/structure/nn/test_nn_structure.py::
    # TestPublicBindings::test_tensor_method_bindings_stay_on_public_functions
    # and tests/structure/runtime/test_runtime_composition_structure.py::
    # test_compat_composition_keeps_native_core_implementations_available.
    # Eager installation only patches SOME of those aliased entry points
    # (e.g. `jt.Var.matmul`/`jt.matmul`, not `nn.matmul`), breaking that
    # identity for every user, not just vmap users -- a bigger, existing,
    # explicitly-tested invariant this project keeps for a reason a narrow
    # vmap-timing fix should not sacrifice. The residual risk stays exactly
    # what `vmap()`'s own docstring already documents and works around
    # (prefer `vmap(lambda x: jt.linalg.qr(x))` over a bare function
    # reference); this test locks that documented, still-current behavior
    # rather than a fix that was tried and found to cost more than it closed.
    #
    # The linalg functions (qr/svd/svdvals/eigh/inv/solve) no longer depend
    # on the patch: each checks for a BatchedVar argument itself and routes
    # to its rule (jittor/linalg/_helpers.py::_batching_aware), which keeps
    # every alias the same object. A saved linalg reference therefore works
    # before the first vmap() call; other saved references keep the
    # limitation above.
    def test_linalg_reference_captured_before_first_vmap_call_dispatches(self):
        from tests._helpers.child_process import run_child_script
        script = """
import numpy as np
import jittor as jt

# Captured before any jt.vmap() call in this process.
solve_ref = jt.linalg.solve

def rand_spd(n, rng):
    m = rng.randn(n, n).astype("float32")
    return m @ m.T + n * np.eye(n, dtype="float32")

rng = np.random.RandomState(0)
a = np.stack([rand_spd(3, rng) for _ in range(2)])
b = rng.randn(2, 3, 1).astype("float32")
A, B = jt.array(a), jt.array(b)
expect = np.stack([np.linalg.solve(a[i], b[i]) for i in range(2)])
out = jt.vmap(lambda av, bv: solve_ref(av, bv))(A, B)
np.testing.assert_allclose(out.numpy(), expect, atol=1e-3, rtol=1e-3)
print("SAVED REFERENCE OK")

out = jt.vmap(lambda av, bv: jt.linalg.solve(av, bv))(A, B)
np.testing.assert_allclose(out.numpy(), expect, atol=1e-3, rtol=1e-3)
print("LAMBDA OK")
"""
        result = run_child_script(script, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("SAVED REFERENCE OK", result.stdout)
        self.assertIn("LAMBDA OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
