# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Parameters accepted for signature compatibility but not honoured.

Each of these used to be swallowed in silence: the call returned, and the only
evidence that the request had been dropped was in the numbers (or in the memory
profile).  ``jittor._arg_policy`` splits them in two:

``ignored``
    the returned values are still correct, only the promise the parameter makes
    (a memory saving, a driver choice) is not kept -> warn once per process;
``unsupported``
    honouring the parameter would change the result -> ``NotImplementedError``,
    with ``JITTOR_ALLOW_UNSUPPORTED_ARGS=1`` as the opt-in escape hatch.

One negative test per parameter.
"""

import unittest
import warnings

import numpy as np

import jittor as jt
from jittor import _arg_policy


class _PolicyCase(unittest.TestCase):
    def setUp(self):
        _arg_policy.reset_warned()
        self._saved_override = _arg_policy.set_allow_unsupported(False)

    def tearDown(self):
        _arg_policy.set_allow_unsupported(self._saved_override)
        _arg_policy.reset_warned()

    def assertWarnsOnce(self, needle, call):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            first = call()
            self.assertEqual(len(caught), 1, [str(c.message) for c in caught])
            self.assertIn(needle, str(caught[0].message))
            self.assertIs(caught[0].category, UserWarning)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            call()
            self.assertEqual(len(caught), 0, "warning must fire once per process")
        return first


class TestInplaceActivations(_PolicyCase):
    """``inplace=True`` allocates a new var anyway -- correct, but no saving."""

    def setUp(self):
        super().setUp()
        self.x = jt.array(np.array([-1.0, 0.5, 2.0], dtype="float32"))

    def _check(self, name, func):
        plain = func(self.x).numpy()
        got = self.assertWarnsOnce(
            "{}: inplace=True".format(name), lambda: func(self.x, inplace=True))
        # the value is unchanged, and the input is NOT overwritten
        np.testing.assert_allclose(got.numpy(), plain, rtol=1e-6, atol=1e-6)
        got.sync()
        np.testing.assert_allclose(
            self.x.numpy(), [-1.0, 0.5, 2.0], rtol=1e-6, atol=1e-6)

    def test_relu(self):
        self._check("jittor.nn.relu", jt.nn.relu)

    def test_leaky_relu(self):
        self._check("jittor.nn.leaky_relu",
                    lambda x, inplace=False: jt.nn.leaky_relu(x, inplace=inplace))

    def test_silu(self):
        self._check("jittor.nn.silu", jt.nn.silu)

    def test_mish(self):
        self._check("jittor.nn.mish", jt.nn.mish)


class TestInstanceNormRunningStats(_PolicyCase):
    def test_running_stats_raise(self):
        x = jt.array(np.zeros((2, 3, 4, 4), dtype="float32"))
        mean = jt.array(np.zeros(3, dtype="float32"))
        var = jt.array(np.ones(3, dtype="float32"))
        with self.assertRaises(NotImplementedError) as ctx:
            jt.nn.instance_norm(x, running_mean=mean, running_var=var)
        self.assertIn("running_mean/running_var", str(ctx.exception))
        with self.assertRaises(NotImplementedError):
            jt.nn.instance_norm(x, running_mean=mean)
        with self.assertRaises(NotImplementedError):
            jt.nn.instance_norm(x, running_var=var)

    def test_without_running_stats_is_unchanged(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((2, 3, 4, 5)).astype("float32")
        got = jt.nn.instance_norm(jt.array(x)).numpy()
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)
        np.testing.assert_allclose(
            got, (x - mean) / np.sqrt(var + 1e-5), rtol=1e-4, atol=1e-4)

    def test_momentum_warns(self):
        """momentum only ever feeds a running-stat update, and there is none."""
        x = jt.array(np.zeros((2, 3, 4, 4), dtype="float32"))
        self.assertWarnsOnce(
            "jittor.nn.instance_norm: momentum=0.5",
            lambda: jt.nn.instance_norm(x, momentum=0.5))
        # the default value must stay silent
        _arg_policy.reset_warned()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jt.nn.instance_norm(x)
            self.assertEqual([str(c.message) for c in caught], [])

    def test_escape_hatch_downgrades_to_a_warning(self):
        _arg_policy.set_allow_unsupported(True)
        x = jt.array(np.zeros((2, 3, 4, 4), dtype="float32"))
        mean = jt.array(np.zeros(3, dtype="float32"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = jt.nn.instance_norm(x, running_mean=mean)
            self.assertEqual(len(caught), 1)
            self.assertIn("running_mean/running_var", str(caught[0].message))
        self.assertEqual(tuple(out.shape), (2, 3, 4, 4))


class TestInstanceNormModuleDeadParameters(_PolicyCase):
    """``InstanceNorm`` stores momentum/is_train/sync and consults none of them."""

    def _module_warns(self, needle, **kwargs):
        self.assertWarnsOnce(needle, lambda: jt.nn.InstanceNorm(3, **kwargs))

    def test_momentum(self):
        self._module_warns("jittor.nn.InstanceNorm: momentum=0.5", momentum=0.5)

    def test_is_train(self):
        self._module_warns("jittor.nn.InstanceNorm: is_train=False",
                           is_train=False)

    def test_sync(self):
        self._module_warns("jittor.nn.InstanceNorm: sync=False", sync=False)

    def test_defaults_are_silent_and_still_normalise(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal((2, 3, 4, 5)).astype("float32")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got = jt.nn.InstanceNorm(3, affine=False)(jt.array(x)).numpy()
            self.assertEqual([str(c.message) for c in caught], [])
        mean = x.mean(axis=(2, 3), keepdims=True)
        var = x.var(axis=(2, 3), keepdims=True)
        np.testing.assert_allclose(
            got, (x - mean) / np.sqrt(var + 1e-5), rtol=1e-4, atol=1e-4)

    def test_is_train_false_really_changes_nothing(self):
        """Why these are `ignored` and not `unsupported`: the numbers agree."""
        rng = np.random.default_rng(4)
        x = jt.array(rng.standard_normal((2, 3, 4, 5)).astype("float32"))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train = jt.nn.InstanceNorm(3, affine=False, is_train=True)(x).numpy()
            evald = jt.nn.InstanceNorm(3, affine=False, is_train=False)(x).numpy()
        np.testing.assert_array_equal(train, evald)


class TestSvdDriverAndComputeUv(_PolicyCase):
    def setUp(self):
        super().setUp()
        rng = np.random.default_rng(1)
        self.x = rng.standard_normal((4, 3)).astype("float32")

    def test_compute_uv_false_warns_and_still_returns_correct_s(self):
        result = self.assertWarnsOnce(
            "jittor.linalg.svd: compute_uv=False",
            lambda: jt.linalg.svd(jt.array(self.x), compute_uv=False))
        expected = np.linalg.svd(self.x.astype("float64"), compute_uv=False)
        np.testing.assert_allclose(
            result[1].numpy(), expected, rtol=1e-4, atol=1e-4)

    def test_driver_warns(self):
        self.assertWarnsOnce(
            "jittor.linalg.svd: driver='gesvd'",
            lambda: jt.linalg.svd(jt.array(self.x), driver="gesvd"))

    def test_svdvals_driver_warns(self):
        self.assertWarnsOnce(
            "jittor.linalg.svdvals: driver='gesvdj'",
            lambda: jt.linalg.svdvals(jt.array(self.x), driver="gesvdj"))


class TestInvExInfo(_PolicyCase):
    def test_info_is_always_zero_and_says_so(self):
        x = jt.array(np.eye(3, dtype="float32"))
        result = self.assertWarnsOnce(
            "jittor.linalg.inv_ex: check_errors",
            lambda: jt.linalg.inv_ex(x))
        np.testing.assert_array_equal(result.info.numpy(), np.zeros((), "int32"))

    def test_check_errors_true_is_honoured_and_stays_silent(self):
        """torch raises for check_errors=True; so does this -- nothing to warn."""
        x = jt.array(np.eye(3, dtype="float32"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = jt.linalg.inv_ex(x, check_errors=True)
            self.assertEqual([str(c.message) for c in caught], [])
        np.testing.assert_allclose(
            result.inverse.numpy(), np.eye(3), rtol=1e-5, atol=1e-5)

    def test_singular_input_raises_instead_of_reporting_through_info(self):
        """The documented mask use case: it does not work, and now says so."""
        singular = np.zeros((3, 3), dtype="float32")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with self.assertRaises(Exception) as ctx:
                jt.linalg.inv_ex(jt.array(singular)).inverse.sync()
        self.assertNotIsInstance(ctx.exception, AssertionError)


class TestCtcLossZeroInfinity(_PolicyCase):
    def _inputs(self):
        rng = np.random.default_rng(2)
        log_probs = jt.array(
            rng.standard_normal((6, 2, 5)).astype("float32")).log_softmax(2)
        targets = jt.array(np.array([[1, 2, 3], [2, 3, 1]], dtype="int32"))
        input_lengths = jt.array(np.array([6, 6], dtype="int32"))
        target_lengths = jt.array(np.array([3, 3], dtype="int32"))
        return log_probs, targets, input_lengths, target_lengths

    def test_zero_infinity_raises(self):
        args = self._inputs()
        with self.assertRaises(NotImplementedError) as ctx:
            jt.ctc_loss(*args, zero_infinity=True)
        self.assertIn("zero_infinity", str(ctx.exception))

    def test_default_still_works(self):
        loss = jt.ctc_loss(*self._inputs())
        self.assertTrue(np.isfinite(float(loss.item())))


class TestSortStable(_PolicyCase):
    def test_stable_raises(self):
        x = jt.array(np.array([3, 1, 1, 2], dtype="int32"))
        with self.assertRaises(NotImplementedError) as ctx:
            jt.sort(x, stable=True)
        self.assertIn("stable", str(ctx.exception))

    def test_argsort_really_is_unstable_on_cpu(self):
        """The reason ``stable=True`` is refused rather than accepted."""
        saved = jt.flags.use_cuda
        jt.flags.use_cuda = 0
        try:
            rng = np.random.default_rng(0)
            keys = rng.integers(0, 4, size=1000).astype("int32")
            index, _ = jt.argsort(jt.array(keys), 0, False)
            stable_reference = np.argsort(keys, kind="stable")
            self.assertFalse(
                np.array_equal(index.numpy(), stable_reference),
                "argsort became stable on CPU -- implement sort(stable=True) "
                "instead of refusing it",
            )
            # it is still a correct sort, just not a stable one
            np.testing.assert_array_equal(
                keys[index.numpy()], keys[stable_reference])
        finally:
            jt.flags.use_cuda = saved

    def test_default_sort_is_unchanged(self):
        x = jt.array(np.array([3.0, 1.0, 2.0], dtype="float32"))
        values, index = jt.sort(x)
        np.testing.assert_allclose(values.numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(index.numpy(), [1, 2, 0])


class TestTopkSorted(_PolicyCase):
    """``sorted=False`` asks for *less*; returning sorted output is compliant."""

    def test_sorted_false_is_accepted_without_a_warning(self):
        x = jt.array(np.array([5.0, 1.0, 4.0, 2.0, 3.0], dtype="float32"))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            values, indices = jt.topk(x, 3, sorted=False)
            self.assertEqual([str(c.message) for c in caught], [])
        # torch leaves the order unspecified for sorted=False; we return the
        # stronger guarantee, so nothing the caller asked for is withheld.
        np.testing.assert_allclose(values.numpy(), [5.0, 4.0, 3.0])
        np.testing.assert_array_equal(indices.numpy(), [0, 2, 4])
        self.assertNotIn(("jittor.topk", "sorted"), _arg_policy.registry())


class TestPolicyRegistry(_PolicyCase):
    def test_every_declared_parameter_is_reachable(self):
        jt.nn.relu(jt.array(np.zeros(2, dtype="float32")), inplace=True)
        entries = _arg_policy.registry()
        self.assertIn(("jittor.nn.relu", "inplace"), entries)
        kind, consequence = entries[("jittor.nn.relu", "inplace")]
        self.assertEqual(kind, "ignored")
        self.assertTrue(consequence)


if __name__ == "__main__":
    unittest.main()
