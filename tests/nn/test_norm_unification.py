# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The normalization modules hold parameters; the arithmetic is in one place.

``BatchNorm`` was the worst of the four. Its training path had two branches
that were not the same function:

* ``sync`` (i.e. under MPI): statistics from ``E[x^2] - E[x]^2``, output built
  as ``x * (weight/sqrt(var+eps)) + (bias - mean*that)``, backward by composite
  autodiff through that expression;
* non-``sync``: statistics recomputed inside ``_ln_normalize`` from the
  two-pass formula ``mean((x-mean)^2)``, output ``(x-mean)*rstd``, backward
  from the closed form.

So whether MPI was initialised decided both the numbers and the gradient
formula -- the same model, the same weights, a different answer once it was
launched with ``mpirun``. And ``E[x^2] - E[x]^2`` cancels catastrophically when
the mean is large next to the standard deviation, which is exactly the branch
that only ran in distributed jobs, where it is hardest to notice.

``nn.functional.batch_norm(training=True)`` was a third transcription, and it
never reached the fused CUDA kernel at all.

All of them now go through ``_batch_norm_train`` / ``_batch_norm_eval``, whose
only ``sync``-dependent step is whether the two statistics are all-reduced.
``LayerNorm``, ``GroupNorm`` and ``InstanceNorm`` likewise delegate to their
functionals.
"""

import unittest

import numpy as np

import jittor as jt


def _reference_batch_norm(x, weight, bias, eps):
    """Training-mode batch norm in float64, straight from the definition."""
    dims = (0,) + tuple(range(2, x.ndim))
    mean = x.mean(axis=dims, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=dims, keepdims=True)
    shape = [1, x.shape[1]] + [1] * (x.ndim - 2)
    return (x - mean) / np.sqrt(var + eps) * weight.reshape(shape) + bias.reshape(shape)


def _reference_grad(x, weight, seed, eps):
    """dL/dx for ``L = sum(seed * batch_norm(x))``, closed form, float64."""
    dims = (0,) + tuple(range(2, x.ndim))
    mean = x.mean(axis=dims, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=dims, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * rstd
    shape = [1, x.shape[1]] + [1] * (x.ndim - 2)
    g = seed * weight.reshape(shape)
    return rstd * (g - g.mean(axis=dims, keepdims=True)
                   - xhat * (g * xhat).mean(axis=dims, keepdims=True))


class _NormParity:
    """Plain mixin; the CPU and CUDA classes below pick ``use_cuda``."""

    use_cuda = 0

    def setUp(self):
        self.rng = np.random.default_rng(20260903)

    def _batch_norm_pieces(self, shape, offset=0.0, scale=1.0):
        x = (self.rng.standard_normal(shape) * scale + offset).astype("float32")
        weight = self.rng.standard_normal(shape[1]).astype("float32")
        bias = self.rng.standard_normal(shape[1]).astype("float32")
        seed = self.rng.standard_normal(shape).astype("float32")
        return x, weight, bias, seed

    def test_module_and_functional_agree(self):
        for shape in ((4, 3, 5, 6), (4, 3, 7), (2, 5, 3, 4, 3)):
            with self.subTest(shape=shape):
                x, weight, bias, seed = self._batch_norm_pieces(shape)
                with jt.flag_scope(use_cuda=self.use_cuda):
                    layer = jt.nn.BatchNorm(shape[1])
                    layer.weight.update(jt.array(weight))
                    layer.bias.update(jt.array(bias))

                    xa = jt.array(x)
                    ya = layer(xa)
                    ga = jt.grad((ya * jt.array(seed)).sum(), [xa])[0]
                    running = (layer.running_mean.numpy().copy(),
                               layer.running_var.numpy().copy())

                    xb = jt.array(x)
                    mean_buf = jt.zeros(shape[1])
                    var_buf = jt.ones(shape[1])
                    yb = jt.nn.batch_norm(
                        xb, mean_buf, var_buf, jt.array(weight),
                        jt.array(bias), training=True)
                    gb = jt.grad((yb * jt.array(seed)).sum(), [xb])[0]
                    values = [v.numpy() for v in (ya, ga, yb, gb)]
                    buffers = (mean_buf.numpy(), var_buf.numpy())
                ya, ga, yb, gb = values
                np.testing.assert_allclose(ya, yb, rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(ga, gb, rtol=1e-4, atol=1e-4)
                # the running buffers are updated the same way too
                np.testing.assert_allclose(running[0], buffers[0],
                                           rtol=1e-5, atol=1e-5)
                np.testing.assert_allclose(running[1], buffers[1],
                                           rtol=1e-5, atol=1e-5)

    def test_values_and_gradient_match_the_reference(self):
        x, weight, bias, seed = self._batch_norm_pieces((4, 3, 5, 6))
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.BatchNorm(3)
            layer.weight.update(jt.array(weight))
            layer.bias.update(jt.array(bias))
            xv = jt.array(x)
            y = layer(xv)
            g = jt.grad((y * jt.array(seed)).sum(), [xv])[0]
            got = (y.numpy(), g.numpy())
        expected = _reference_batch_norm(x.astype("float64"),
                                         weight.astype("float64"),
                                         bias.astype("float64"), 1e-5)
        np.testing.assert_allclose(got[0], expected, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(
            got[1],
            _reference_grad(x.astype("float64"), weight.astype("float64"),
                            seed.astype("float64"), 1e-5),
            rtol=2e-3, atol=2e-3)

    def _run_batch_norm(self, sync, x, weight, bias, seed):
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.BatchNorm(x.shape[1], sync=sync)
            layer.weight.update(jt.array(weight))
            layer.bias.update(jt.array(bias))
            xv = jt.array(x)
            y = layer(xv)
            g = jt.grad((y * jt.array(seed)).sum(), [xv])[0]
            return [v.numpy() for v in (y, g, layer.running_mean,
                                        layer.running_var)]

    def test_sync_and_local_branches_are_the_same_function(self):
        """The headline defect: whether MPI is up decided the BN numbers.

        ``jt.in_mpi`` is forced on and ``mpi_all_reduce`` replaced by the
        identity, so this runs without a launcher -- and that is exactly the
        claim: with a world of one the collective changes nothing, so
        ``sync=True`` and ``sync=False`` have to agree. They did not: the sync
        branch took its statistics from ``E[x^2]-E[x]^2`` and scale-shifted raw
        x with composite-autodiff backward, the local branch used the two-pass
        variance and the closed-form backward.
        """
        # A mean that is large next to the standard deviation: this is where
        # the two formulas visibly disagree. At mean 0 / std 1 they agree to
        # 1e-5 and the defect hides -- which is why nobody noticed it.
        x, weight, bias, seed = self._batch_norm_pieces(
            (4, 3, 5, 6), offset=100.0, scale=0.05)
        identity = lambda self, op: self  # noqa: E731
        original_reduce = getattr(jt.Var, "mpi_all_reduce", None)
        original_in_mpi = jt.compile_extern.in_mpi
        original_world = jt.compile_extern.world_size
        try:
            jt.Var.mpi_all_reduce = identity
            # jt.in_mpi / jt.world_size are served from compile_extern; writing
            # to jt directly would shadow the accessor permanently (6.B15).
            jt.compile_extern.in_mpi = True
            jt.compile_extern.world_size = 1
            self.assertTrue(jt.in_mpi)
            synced = self._run_batch_norm(True, x, weight, bias, seed)
            jt.compile_extern.in_mpi = False
            local = self._run_batch_norm(False, x, weight, bias, seed)
        finally:
            if original_reduce is None:
                del jt.Var.mpi_all_reduce
            else:
                jt.Var.mpi_all_reduce = original_reduce
            jt.compile_extern.in_mpi = original_in_mpi
            jt.compile_extern.world_size = original_world
        for a, b, name in zip(local, synced,
                              ("output", "grad", "running_mean", "running_var")):
            with self.subTest(tensor=name):
                # 1e-3 rather than 1e-5 because on CUDA the local branch takes
                # the fused kernel, which sums the same two-pass formula in a
                # different order; on this deliberately ill-conditioned input
                # that shows up at ~2e-4. Before the merge the two branches
                # used different formulas and disagreed by ~7e-2.
                np.testing.assert_allclose(a, b, rtol=1e-3, atol=1e-3)
        # both branches also have to be right, not merely equal
        expected = _reference_batch_norm(x.astype("float64"),
                                         weight.astype("float64"),
                                         bias.astype("float64"), 1e-5)
        np.testing.assert_allclose(local[0], expected, rtol=5e-3, atol=5e-3)
        np.testing.assert_allclose(synced[0], expected, rtol=5e-3, atol=5e-3)

    def test_sync_and_local_agree_at_the_helper_level_too(self):
        """Same claim one level down, where the answer must be bit-identical."""
        from jittor.nn.functional import normalization as F_norm

        x, weight, bias, seed = self._batch_norm_pieces((4, 3, 5, 6))
        results = {}
        identity = lambda self, op: self  # noqa: E731
        original = getattr(jt.Var, "mpi_all_reduce", None)
        try:
            jt.Var.mpi_all_reduce = identity
            for sync in (False, True):
                with jt.flag_scope(use_cuda=self.use_cuda):
                    xv = jt.array(x)
                    y, mean, var = F_norm._batch_norm_train(
                        xv, [0, 2, 3], jt.array(weight), jt.array(bias),
                        1e-5, sync=sync)
                    g = jt.grad((y * jt.array(seed)).sum(), [xv])[0]
                    results[sync] = [v.numpy() for v in (y, g, mean, var)]
        finally:
            if original is None:
                del jt.Var.mpi_all_reduce
            else:
                jt.Var.mpi_all_reduce = original
        for local, synced, name in zip(results[False], results[True],
                                       ("output", "grad", "mean", "var")):
            with self.subTest(tensor=name):
                if self.use_cuda:
                    # On CUDA sync=False takes the fused kernel, so the two
                    # graphs fuse and reduce differently -- same function, not
                    # the same instruction sequence, and CUDA reductions are
                    # order-dependent. On CPU they are literally the same
                    # graph and must be bit-equal, which is the assertion that
                    # would catch a second implementation creeping back.
                    np.testing.assert_allclose(local, synced, rtol=1e-5,
                                               atol=1e-5)
                else:
                    np.testing.assert_array_equal(local, synced)

    def test_two_pass_variance_where_the_old_formula_collapses(self):
        """``E[x^2] - E[x]^2`` was used *only* in the sync branch.

        float32 keeps about 7 digits, so once ``var / mean**2`` drops below
        1e-7 the two squared terms agree to the last bit they have and their
        difference is mostly rounding. This measures both formulas on the same
        input: the two-pass one is accurate, the old one is not. The old one
        ran only under MPI, which is where it is hardest to notice.
        """
        from jittor.nn.functional import normalization as F_norm

        x, _, _, _ = self._batch_norm_pieces((4, 3, 5, 6), offset=100.0,
                                             scale=0.05)
        dims = [0, 2, 3]
        with jt.flag_scope(use_cuda=self.use_cuda):
            xv = jt.array(x)
            old_mean = jt.mean(xv, dims=dims)
            old_var = (jt.mean(xv * xv, dims=dims)
                       - old_mean * old_mean).maximum(0.0)
            new_var = F_norm._batch_statistics(xv, dims, False)[1]
            old_var, new_var = old_var.numpy(), new_var.numpy()
        exact = x.astype("float64")
        exact_var = ((exact - exact.mean(axis=(0, 2, 3), keepdims=True)) ** 2
                     ).mean(axis=(0, 2, 3))
        np.testing.assert_allclose(new_var, exact_var, rtol=5e-3, atol=0)
        self.assertGreater(
            float(np.abs(old_var / exact_var - 1).max()), 0.05,
            "E[x^2]-E[x]^2 stopped losing precision here; if it is accurate "
            "now this test has lost its point and needs a harsher input")

    def test_output_is_accurate_for_a_large_mean(self):
        x, weight, bias, _ = self._batch_norm_pieces(
            (4, 3, 5, 6), offset=100.0, scale=0.05)
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.BatchNorm(3)
            layer.weight.update(jt.array(weight))
            layer.bias.update(jt.array(bias))
            got = layer(jt.array(x)).numpy()
        expected = _reference_batch_norm(x.astype("float64"),
                                         weight.astype("float64"),
                                         bias.astype("float64"), 1e-5)
        np.testing.assert_allclose(got, expected, rtol=5e-3, atol=5e-3)

    def test_eval_mode_module_and_functional_agree(self):
        x, weight, bias, _ = self._batch_norm_pieces((4, 3, 5, 6))
        mean = self.rng.standard_normal(3).astype("float32")
        var = np.abs(self.rng.standard_normal(3)).astype("float32") + 0.5
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.BatchNorm(3, is_train=False)
            layer.weight.update(jt.array(weight))
            layer.bias.update(jt.array(bias))
            layer.running_mean.update(jt.array(mean))
            layer.running_var.update(jt.array(var))
            got = layer(jt.array(x)).numpy()
            want = jt.nn.batch_norm(
                jt.array(x), jt.array(mean), jt.array(var), jt.array(weight),
                jt.array(bias), training=False).numpy()
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
        shape = (1, 3, 1, 1)
        reference = ((x - mean.reshape(shape))
                     / np.sqrt(var.reshape(shape) + 1e-5)
                     * weight.reshape(shape) + bias.reshape(shape))
        np.testing.assert_allclose(got, reference, rtol=1e-4, atol=1e-4)

    def _module_matches_functional(self, module, functional, x, seed):
        with jt.flag_scope(use_cuda=self.use_cuda):
            xa = jt.array(x)
            ya = module(xa)
            ga = jt.grad((ya * jt.array(seed)).sum(), [xa])[0]
            xb = jt.array(x)
            yb = functional(xb)
            gb = jt.grad((yb * jt.array(seed)).sum(), [xb])[0]
            values = [v.numpy() for v in (ya, ga, yb, gb)]
        ya, ga, yb, gb = values
        np.testing.assert_array_equal(ya, yb)
        if self.use_cuda:
            # CUDA reductions are not bit-reproducible call to call, so
            # bit-equality on the gradient would be flaky whatever the two
            # spellings share; the forward above is deterministic and is the
            # assertion that would catch a second implementation.
            np.testing.assert_allclose(ga, gb, rtol=1e-5, atol=1e-5)
        else:
            np.testing.assert_array_equal(ga, gb)

    def test_layer_norm_module_delegates(self):
        x = self.rng.standard_normal((3, 4, 5)).astype("float32")
        seed = self.rng.standard_normal((3, 4, 5)).astype("float32")
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.LayerNorm((4, 5))
            layer.weight.update(jt.array(
                self.rng.standard_normal((4, 5)).astype("float32")))
            layer.bias.update(jt.array(
                self.rng.standard_normal((4, 5)).astype("float32")))
        self._module_matches_functional(
            layer,
            lambda v: jt.nn.layer_norm(v, (4, 5), layer.weight, layer.bias,
                                       layer.eps),
            x, seed)

    def test_group_norm_module_delegates(self):
        x = self.rng.standard_normal((2, 6, 4, 5)).astype("float32")
        seed = self.rng.standard_normal((2, 6, 4, 5)).astype("float32")
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.GroupNorm(3, 6)
            layer.weight.update(jt.array(
                self.rng.standard_normal(6).astype("float32")))
            layer.bias.update(jt.array(
                self.rng.standard_normal(6).astype("float32")))
        self._module_matches_functional(
            layer,
            lambda v: jt.nn.group_norm(v, 3, layer.weight, layer.bias,
                                       layer.eps),
            x, seed)

    def test_instance_norm_module_delegates(self):
        x = self.rng.standard_normal((2, 3, 4, 5)).astype("float32")
        seed = self.rng.standard_normal((2, 3, 4, 5)).astype("float32")
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.InstanceNorm(3)
            layer.weight.update(jt.array(
                self.rng.standard_normal(3).astype("float32")))
            layer.bias.update(jt.array(
                self.rng.standard_normal(3).astype("float32")))
        self._module_matches_functional(
            layer,
            lambda v: jt.nn.instance_norm(v, weight=layer.weight,
                                          bias=layer.bias, eps=layer.eps),
            x, seed)

    def test_group_norm_module_rejects_a_wrong_channel_count(self):
        with jt.flag_scope(use_cuda=self.use_cuda):
            layer = jt.nn.GroupNorm(3, 6)
            with self.assertRaises(ValueError) as ctx:
                layer(jt.array(np.zeros((2, 4, 3, 3), dtype="float32")))
            self.assertIn("num_channels", str(ctx.exception))


class TestNormParityCPU(_NormParity, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not jt.has_cuda, "no CUDA")
class TestNormParityCUDA(_NormParity, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
