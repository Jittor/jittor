"""Fidelity battery for the family that ``_install_reductions`` used to own.

Every API here used to be a closure built inside
``installers/tensor.py::_install_reductions``, and several were built *twice* --
once as ``torch.foo`` and once as ``Tensor.foo`` -- from two different function
objects. Two objects for one API is not a cosmetic problem: the two spellings
had measurably different behaviour, and the difference was silent.

Measured on this worktree before the promotion, against PyTorch 2.12.1 as the
oracle:

``torch.var(t, axis=0)`` / ``torch.std(t, axis=0)``
    returned the *full* reduction (13.0 / 3.6055512 for a 3x4 arange) where
    Torch and ``t.var(axis=0)`` both return the per-column ``[16, 16, 16, 16]``
    / ``[4, 4, 4, 4]``. The module-level spelling swallowed ``axis`` in
    ``**kw``; the method spelling got it translated by ``_axis_to_dim``.
``torch.max(t, axis=0)`` / ``torch.min(t, axis=0)``
    returned a 0-dim scalar where Torch and ``t.max(axis=0)`` return the
    ``(values, indices)`` pair.
``torch.softmax(t, dim, dtype=...)`` / ``torch.log_softmax(...)``
    ignored ``dtype`` and returned float32 where Torch and
    ``t.softmax(dim, dtype=...)`` return float64.

The device-parameterized class runs on every device the session selected,
because this is a reduction family and the previous two promotions (cumulative
scan, ordering) both found real CPU/CUDA divergence. Measured here on sm_89:
the *arg* reductions are bit-identical across the two backends even on 8x512
rows with about five duplicates per key -- unlike ``argsort``, which is not --
while ``var``/``std`` are not bit-identical, differing by ~1.3e-07 relative,
with CUDA the closer of the two to a float64 reference.
"""

import ast
import importlib
import inspect
import unittest

import numpy as np
import torch
import jittor as _native_jittor

from _helpers import common as cu
from _helpers.device_types import instantiate_device_type_tests


#: Promoted into ``installers/tensor.py`` and bound to both spellings.
TENSOR_OWNED_BOTH = ("argmax", "argmin", "max", "min", "var", "std",
                     "broadcast_to")
#: Promoted into ``installers/tensor.py``; Torch has no module-level spelling
#: for ``unfold`` at all, and the remaining ones stay methods so this promotion
#: does not widen the published surface.
TENSOR_OWNED_METHOD_ONLY = ("masked_scatter", "masked_scatter_", "unfold",
                            "addcmul", "addcdiv")
#: Already had a final owner elsewhere; install only re-exports it.
REEXPORTED = {
    "diagonal": "jittor.ops.shape_ops",
    "masked_select": "jittor.compat.torch.installers.numerical.indexing",
    "softmax": "jittor.compat.torch.installers.numerical.elementwise",
    "log_softmax": "jittor.compat.torch.installers.numerical.elementwise",
}

MATRIX = np.arange(12, dtype="float32").reshape(3, 4)
#: 8x512 with ~5 duplicates per key: enough for two reduction implementations
#: to disagree about which of several equal maxima they report.
DUPLICATED = (np.arange(4096, dtype="float32") % 97).reshape(8, 512)
#: Large magnitudes so a float32 sum of squares actually loses bits.
SPREAD = (np.arange(4096, dtype="float32") - 2048.0).reshape(8, 512) * 977.0


def _tensor_owner():
    return importlib.import_module("jittor.compat.torch.installers.tensor")


def _numerical_owner():
    return importlib.import_module("jittor.compat.torch.installers.numerical")


class TestReductionOwnerMetadata(unittest.TestCase):
    """Identity and fidelity metadata, which do not depend on the device."""

    def test_promoted_apis_are_one_module_level_object_per_api(self):
        owner = _tensor_owner()
        for name in TENSOR_OWNED_BOTH:
            with self.subTest(name=name):
                implementation = getattr(owner, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch, name), implementation)
                self.assertIs(getattr(torch.Var, name), implementation)
                family = ".indexing" if name == "broadcast_to" else ".reductions"
                self.assertEqual(implementation.__module__, owner.__name__ + family)
                self.assertEqual(implementation.__name__, name)

    def test_method_only_apis_are_module_level_objects_too(self):
        owner = _tensor_owner()
        for name in TENSOR_OWNED_METHOD_ONLY:
            with self.subTest(name=name):
                implementation = getattr(owner, name)
                self.assertTrue(callable(implementation))
                self.assertIs(getattr(torch.Var, name), implementation)
                self.assertEqual(implementation.__module__, owner.__name__ + ".indexing")
                self.assertEqual(implementation.__name__, name)

    def test_reexported_apis_point_at_their_existing_owner(self):
        """No second copy of an API that already had a final owner.

        ``diagonal`` had a full reindex implementation here on top of the one
        in ``jittor.misc.tensor_ops`` that ``torch.diagonal`` was already
        resolving to; ``masked_select``/``softmax``/``log_softmax`` each had a
        method-side copy on top of the ``numerical`` owner.
        """
        for name, module_name in REEXPORTED.items():
            with self.subTest(name=name):
                owner = importlib.import_module(module_name)
                implementation = getattr(owner, name)
                self.assertIs(getattr(torch, name), implementation)
                self.assertIs(getattr(torch.Var, name), implementation)
                self.assertEqual(implementation.__module__, module_name)

    def test_the_installer_holds_no_closures_at_all(self):
        """The installer has no nested def/class/lambda left in it.

        This is the closable half of task 7.03 for this installer: not "a few
        more APIs promoted" but "this installer is empty", which is a fact a
        test can hold on to.
        """
        owner = _tensor_owner()
        tree = ast.parse(inspect.getsource(owner))
        installer = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "_install_reductions")
        nested = [
            node for node in ast.walk(installer) if node is not installer
            and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.ClassDef, ast.Lambda))]
        self.assertEqual(
            [getattr(node, "name", "<lambda>") for node in nested], [])

    def test_the_captured_natives_are_still_the_natives(self):
        """The import-time capture has to happen before anything patches them.

        ``_maxmin`` reduces through ``jt.max``/``jt.Var.max``, which install
        then rebinds -- a late lookup would recurse forever. Capturing at
        import time is only correct while no earlier install step has already
        replaced them, so pin that rather than assume it.
        """
        owner = _tensor_owner()
        promoted = {id(getattr(owner, name)) for name in TENSOR_OWNED_BOTH}
        for attribute in ("_NATIVE_ARGMAX", "_NATIVE_ARGMIN",
                          "_NATIVE_MAXIMUM", "_NATIVE_MINIMUM",
                          "_NATIVE_MAX", "_NATIVE_MIN",
                          "_NATIVE_MAX_METHOD", "_NATIVE_MIN_METHOD",
                          "_NATIVE_VAR_METHOD"):
            with self.subTest(attribute=attribute):
                captured = getattr(owner, attribute)
                self.assertTrue(callable(captured))
                self.assertNotIn(id(captured), promoted)
                self.assertFalse(
                    getattr(captured, "__module__", "").startswith(
                        "jittor.compat"),
                    "%s captured a compat object, so something had already "
                    "patched it by import time" % attribute)

    def test_fidelity_records_the_axis_alias_and_the_backend_spread(self):
        owner = _tensor_owner()
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("argmax", "argmin"):
            record = fidelity.fidelity_of("torch." + name)
            with self.subTest(name=name):
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIs(record.level, fidelity.Fidelity.APPROXIMATE)
                self.assertIn("int64", record.detail)
                self.assertIn("axis", record.detail)
                self.assertIn("duplicate keys", record.detail)
                self.assertIn("device", record.detail)
                self.assertIn("out", record.detail)
        for name in ("max", "min"):
            record = fidelity.fidelity_of("torch." + name)
            with self.subTest(name=name):
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIn("axis", record.detail)
                self.assertIn("keepdims", record.detail)
                self.assertIn("device", record.detail)
        for name in ("var", "std"):
            record = fidelity.fidelity_of("torch." + name)
            with self.subTest(name=name):
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIn("axis", record.detail)
                self.assertIn("correction", record.detail)
                self.assertIn("summation order", record.detail)
                self.assertIn("device", record.detail)

    def test_fidelity_records_the_mask_and_view_limits(self):
        owner = _tensor_owner()
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        for name in ("masked_scatter", "masked_scatter_"):
            record = fidelity.fidelity_of("torch.Tensor." + name)
            with self.subTest(name=name):
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIn("row-major", record.detail)
                self.assertIn("differentiable", record.detail)
                self.assertIn("device", record.detail)
        record = fidelity.fidelity_of("torch.Tensor.unfold")
        self.assertIs(record.implementation, owner.unfold)
        self.assertIn("stride view", record.detail)
        self.assertIn("device", record.detail)
        for name in ("addcmul", "addcdiv"):
            record = fidelity.fidelity_of("torch.Tensor." + name)
            with self.subTest(name=name):
                self.assertIs(record.implementation, getattr(owner, name))
                self.assertIn("out", record.detail)
                self.assertIn("device", record.detail)
        record = fidelity.fidelity_of("torch.broadcast_to")
        self.assertIs(record.implementation, owner.broadcast_to)
        self.assertIn("device", record.detail)

    def test_fidelity_records_the_reexported_owners(self):
        fidelity = importlib.import_module("jittor.compat.torch.fidelity")
        record = fidelity.fidelity_of("torch.diagonal")
        self.assertIs(record.implementation, _tensor_owner().diagonal)
        self.assertIn("re-exports", record.detail)
        for name in ("softmax", "log_softmax"):
            record = fidelity.fidelity_of("torch." + name)
            with self.subTest(name=name):
                self.assertIs(record.implementation,
                              getattr(_numerical_owner(), name))
                self.assertIn("dtype", record.detail)
                self.assertIn("device", record.detail)


class TestReductionOwner(cu.JittorTestCase):
    # ---------------------------------------------------------- axis= alias
    # Torch accepts ``axis`` as a NumPy-compatible alias for ``dim`` on these
    # reductions (verified on 2.12.1). The module-level spellings used to drop
    # it into ``**kw`` and silently reduce over everything.

    def test_var_and_std_honour_the_axis_alias_like_the_method_does(self, device):
        tensor = torch.tensor(MATRIX)
        for name, reference in (
                ("var", np.var(MATRIX, axis=0, ddof=1)),
                ("std", np.std(MATRIX, axis=0, ddof=1))):
            with self.subTest(name=name):
                through_module = getattr(torch, name)(tensor, axis=0).numpy()
                through_method = getattr(tensor, name)(axis=0).numpy()
                np.testing.assert_allclose(
                    through_module, reference, rtol=1e-6)
                np.testing.assert_array_equal(through_module, through_method)

    def test_max_and_min_honour_the_axis_alias_like_the_method_does(self, device):
        tensor = torch.tensor(MATRIX)
        for name, values, indices in (
                ("max", np.max(MATRIX, axis=0), np.argmax(MATRIX, axis=0)),
                ("min", np.min(MATRIX, axis=0), np.argmin(MATRIX, axis=0))):
            with self.subTest(name=name):
                result = getattr(torch, name)(tensor, axis=0)
                self.assertEqual(result._fields, ("values", "indices"))
                np.testing.assert_array_equal(result.values.numpy(), values)
                np.testing.assert_array_equal(result.indices.numpy(), indices)
                self.assertEqual(str(result.indices.dtype), "int64")
                np.testing.assert_array_equal(
                    getattr(tensor, name)(axis=0).values.numpy(),
                    result.values.numpy())

    def test_argmax_and_argmin_accept_the_axis_alias(self, device):
        tensor = torch.tensor(MATRIX)
        for name, reference in (
                ("argmax", np.argmax(MATRIX, axis=0)),
                ("argmin", np.argmin(MATRIX, axis=0))):
            with self.subTest(name=name):
                through_module = getattr(torch, name)(tensor, axis=0)
                self.assertEqual(str(through_module.dtype), "int64")
                np.testing.assert_array_equal(through_module.numpy(), reference)
                np.testing.assert_array_equal(
                    getattr(tensor, name)(axis=0).numpy(), reference)

    def test_softmax_family_honours_the_dtype_keyword(self, device):
        """``dtype=`` casts the input first, as Torch's own ``dtype=`` does.

        vLLM's sampler spells it ``logits.softmax(dim=-1,
        dtype=torch.float32)``; the module-level spelling used to drop it, so
        the same call written ``torch.softmax(logits, -1, dtype=...)`` computed
        in the input's narrow dtype instead.
        """
        tensor = torch.tensor(MATRIX)
        for name in ("softmax", "log_softmax"):
            with self.subTest(name=name):
                through_module = getattr(torch, name)(
                    tensor, dim=-1, dtype="float64")
                through_method = getattr(tensor, name)(
                    dim=-1, dtype="float64")
                self.assertEqual(str(through_module.dtype), "float64")
                self.assertEqual(str(through_method.dtype), "float64")
                np.testing.assert_array_equal(
                    through_module.numpy(), through_method.numpy())

    # ------------------------------------------------- values against NumPy

    def test_variance_defaults_to_the_unbiased_estimator(self, device):
        """Torch's default is ``correction=1``; Jittor's native var is biased."""
        tensor = torch.tensor(MATRIX)
        np.testing.assert_allclose(
            torch.var(tensor, dim=1).numpy(),
            np.var(MATRIX, axis=1, ddof=1), rtol=1e-6)
        np.testing.assert_allclose(
            torch.var(tensor, dim=1, correction=0).numpy(),
            np.var(MATRIX, axis=1, ddof=0), rtol=1e-6)
        np.testing.assert_allclose(
            torch.var(tensor, dim=1, unbiased=False).numpy(),
            np.var(MATRIX, axis=1, ddof=0), rtol=1e-6)
        self.assertEqual(
            tuple(torch.var(tensor, dim=1, keepdim=True).shape), (3, 1))

    def test_variance_over_a_tuple_of_dims(self, device):
        """The native ``dim=`` slot is scalar-only, so this path is compat's."""
        cube = (np.arange(24, dtype="float32") * 0.5).reshape(2, 3, 4)
        tensor = torch.tensor(cube)
        np.testing.assert_allclose(
            tensor.var(dim=(1, 2)).numpy(),
            np.var(cube.reshape(2, -1), axis=1, ddof=1), rtol=1e-6)
        np.testing.assert_allclose(
            tensor.var(dim=(1, 2), keepdim=True).numpy().reshape(-1),
            np.var(cube.reshape(2, -1), axis=1, ddof=1), rtol=1e-6)

    def test_std_has_no_floor_under_a_constant_row(self, device):
        """Jittor's native std clamps at 1e-6; Torch's is a real zero."""
        flat = np.full((2, 5), 3.0, dtype="float32")
        actual = torch.tensor(flat).std(dim=1).numpy()
        np.testing.assert_array_equal(actual, np.zeros(2, dtype="float32"))

    def test_max_and_min_keep_their_three_torch_shapes(self, device):
        tensor = torch.tensor(MATRIX)
        np.testing.assert_array_equal(torch.max(tensor).numpy(), MATRIX.max())
        np.testing.assert_array_equal(torch.min(tensor).numpy(), MATRIX.min())
        flipped = MATRIX[::-1].copy()
        other = torch.tensor(flipped)
        np.testing.assert_array_equal(
            torch.max(tensor, other).numpy(), np.maximum(MATRIX, flipped))
        np.testing.assert_array_equal(
            torch.min(tensor, other).numpy(), np.minimum(MATRIX, flipped))
        kept = torch.max(tensor, dim=1, keepdim=True)
        self.assertEqual(tuple(kept.values.shape), (3, 1))

    def test_max_keeps_the_values_only_keepdims_spelling_for_jittor(self, device):
        """jittor's own softmax/layernorm call ``x.max(dim, keepdims=True)``."""
        tensor = torch.tensor(MATRIX)
        values = tensor.max(1, keepdims=True)
        self.assertNotIsInstance(values, tuple)
        np.testing.assert_array_equal(
            values.numpy(), MATRIX.max(axis=1, keepdims=True))

    def test_masked_scatter_consumes_the_source_in_row_major_order(self, device):
        mask = MATRIX % 3 == 0
        source = np.arange(int(mask.sum()), dtype="float32") + 100.0
        expected = MATRIX.copy()
        expected[mask] = source
        actual = torch.tensor(MATRIX).masked_scatter(
            torch.tensor(mask), torch.tensor(source))
        np.testing.assert_array_equal(actual.numpy(), expected)
        self.assertEqual(str(actual.dtype), "float32")

    def test_masked_scatter_inplace_returns_the_same_var(self, device):
        mask = MATRIX % 3 == 0
        source = np.arange(int(mask.sum()), dtype="float32") + 100.0
        expected = MATRIX.copy()
        expected[mask] = source
        tensor = torch.tensor(MATRIX)
        returned = tensor.masked_scatter_(
            torch.tensor(mask), torch.tensor(source))
        self.assertIs(returned, tensor)
        np.testing.assert_array_equal(tensor.numpy(), expected)

    def test_masked_scatter_broadcasts_the_mask(self, device):
        mask = np.array([[True, False, True, False]])
        full = np.broadcast_to(mask, MATRIX.shape)
        source = np.arange(int(full.sum()), dtype="float32") + 100.0
        expected = MATRIX.copy()
        expected[full] = source
        actual = torch.tensor(MATRIX).masked_scatter(
            torch.tensor(mask), torch.tensor(source))
        np.testing.assert_array_equal(actual.numpy(), expected)

    def test_masked_select_flattens_the_selection(self, device):
        mask = MATRIX > 5
        tensor = torch.tensor(MATRIX)
        np.testing.assert_array_equal(
            torch.masked_select(tensor, torch.tensor(mask)).numpy(),
            MATRIX[mask])
        np.testing.assert_array_equal(
            tensor.masked_select(torch.tensor(mask)).numpy(), MATRIX[mask])

    def test_unfold_slides_a_window_along_one_dim(self, device):
        tensor = torch.tensor(MATRIX)
        actual = tensor.unfold(1, 2, 2).numpy()
        self.assertEqual(actual.shape, (3, 2, 2))
        np.testing.assert_array_equal(actual, MATRIX.reshape(3, 2, 2))
        stepped = tensor.unfold(1, 3, 1).numpy()
        self.assertEqual(stepped.shape, (3, 2, 3))
        np.testing.assert_array_equal(stepped[:, 0, :], MATRIX[:, 0:3])
        np.testing.assert_array_equal(stepped[:, 1, :], MATRIX[:, 1:4])
        np.testing.assert_array_equal(
            tensor.unfold(-1, 2, 2).numpy(), MATRIX.reshape(3, 2, 2))

    def test_diagonal_matches_numpy_including_negative_axes(self, device):
        cube = np.arange(24, dtype="float32").reshape(2, 3, 4)
        tensor = torch.tensor(cube)
        for offset, dim1, dim2 in ((0, 0, 2), (1, 0, 2), (-1, 0, 2),
                                   (0, -2, -1), (2, 1, 2), (-5, 0, 1)):
            with self.subTest(offset=offset, dim1=dim1, dim2=dim2):
                expected = np.diagonal(cube, offset, dim1, dim2)
                np.testing.assert_array_equal(
                    torch.diagonal(tensor, offset, dim1, dim2).numpy(),
                    expected)
                np.testing.assert_array_equal(
                    tensor.diagonal(offset, dim1, dim2).numpy(), expected)

    def test_addcmul_and_addcdiv_scale_the_product(self, device):
        first = np.array([[2.0, 4.0], [8.0, 16.0]], dtype="float32")
        second = np.array([[1.0, 2.0], [4.0, 8.0]], dtype="float32")
        base = MATRIX[:2, :2]
        tensor = torch.tensor(base)
        np.testing.assert_allclose(
            tensor.addcmul(torch.tensor(first), torch.tensor(second),
                           value=0.5).numpy(),
            base + 0.5 * (first * second), rtol=1e-6)
        np.testing.assert_allclose(
            tensor.addcdiv(torch.tensor(first), torch.tensor(second),
                           value=2).numpy(),
            base + 2 * (first / second), rtol=1e-6)

    def test_broadcast_to_expands_without_copying_values(self, device):
        row = np.array([[1.0, 2.0, 3.0]], dtype="float32")
        tensor = torch.tensor(row)
        expected = np.broadcast_to(row, (4, 3))
        np.testing.assert_array_equal(
            torch.broadcast_to(tensor, (4, 3)).numpy(), expected)
        np.testing.assert_array_equal(
            tensor.broadcast_to((4, 3)).numpy(), expected)

    # ------------------------------------------------- this device vs the CPU

    def test_arg_reductions_agree_with_the_cpu_path_bit_for_bit(self, device):
        """Unlike ``argsort``, the arg-reduce path is stable across backends.

        Measured on sm_89 with 8x512 rows drawn from ``arange % 97`` (about
        five duplicates per key): CPU and CUDA return *the same* index for
        every row, for all four of argmax/argmin/max/min. That is a stronger
        invariant than the ordering family gets, so pin it exactly -- if a
        future kernel starts breaking ties differently, this test is where it
        shows up, rather than in a model that silently gathers other rows.
        """
        tensor_here = torch.tensor(DUPLICATED)
        rows = np.arange(DUPLICATED.shape[0])
        for name in ("argmax", "argmin"):
            with self.subTest(name=name):
                here = getattr(torch, name)(tensor_here, dim=1).numpy()
                with _native_jittor.flag_scope(use_cuda=0):
                    on_cpu = getattr(torch, name)(
                        torch.tensor(DUPLICATED), dim=1).numpy()
                np.testing.assert_array_equal(here, on_cpu)
                np.testing.assert_array_equal(
                    DUPLICATED[rows, here],
                    getattr(np, name.replace("arg", ""))(DUPLICATED, axis=1))
        for name in ("max", "min"):
            with self.subTest(name=name):
                here = getattr(torch, name)(tensor_here, dim=1)
                with _native_jittor.flag_scope(use_cuda=0):
                    on_cpu = getattr(torch, name)(
                        torch.tensor(DUPLICATED), dim=1)
                    cpu_values = on_cpu.values.numpy()
                    cpu_indices = on_cpu.indices.numpy()
                np.testing.assert_array_equal(here.values.numpy(), cpu_values)
                np.testing.assert_array_equal(
                    here.indices.numpy(), cpu_indices)

    def test_variance_agrees_with_the_cpu_path_only_to_float32_rounding(self, device):
        """A variance is a sum, so the backends' summation orders differ.

        Measured on sm_89 over 512-element float32 rows spanning +-2e6: CPU is
        1.2e-07 relative from a float64 reference, CUDA 9.6e-08, and the two
        differ from each other by 1.3e-07 -- the parallel reduction is the more
        accurate, the same way the parallel prefix sum beat the sequential scan
        for ``cumsum``. So this pins a *bounded* inconsistency plus the fact
        that both sides track the float64 reference, not bit equality.
        """
        reference = np.var(SPREAD.astype("float64"), axis=1, ddof=1)
        here = torch.var(torch.tensor(SPREAD), dim=1).numpy()
        with _native_jittor.flag_scope(use_cuda=0):
            on_cpu = torch.var(torch.tensor(SPREAD), dim=1).numpy()
        np.testing.assert_allclose(here, reference, rtol=1e-6)
        np.testing.assert_allclose(on_cpu, reference, rtol=1e-6)
        np.testing.assert_allclose(here, on_cpu, rtol=1e-5)

    def test_mask_and_view_paths_agree_with_the_cpu_path_bit_for_bit(self, device):
        """No accumulation in any of these, so they must be exact both sides."""
        mask = DUPLICATED % 3 == 0
        source = np.arange(int(mask.sum()), dtype="float32")

        def scatter():
            return torch.tensor(DUPLICATED).masked_scatter(
                torch.tensor(mask), torch.tensor(source)).numpy()

        def unfold():
            return torch.tensor(DUPLICATED).unfold(1, 4, 2).numpy()

        def diagonal():
            return torch.tensor(DUPLICATED[:, :8]).diagonal(0, 0, 1).numpy()

        for probe in (scatter, unfold, diagonal):
            with self.subTest(probe=probe.__name__):
                here = probe()
                with _native_jittor.flag_scope(use_cuda=0):
                    on_cpu = probe()
                np.testing.assert_array_equal(here, on_cpu)


instantiate_device_type_tests(TestReductionOwner, globals())


if __name__ == "__main__":
    unittest.main()
