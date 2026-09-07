"""Real tensor coverage for the initialization package's implementation owners."""

import importlib
import pickle
import unittest

import jittor as jt
import numpy as np


class TestInitPackage(unittest.TestCase):
    def test_native_public_identity_and_serialization(self):
        init = jt.init
        for name in init.__all__:
            function = getattr(init, name)
            owner = importlib.import_module(function.__module__)
            self.assertIs(getattr(owner, function.__name__), function)
            self.assertIs(pickle.loads(pickle.dumps(function)), function)
            self.assertIs(
                pickle.loads(("cjittor.init\n%s\n." % name).encode("ascii")),
                function,
            )
        for name in ("eye_", "uniform_", "kaiming_normal_", "trunc_normal_"):
            self.assertIs(getattr(jt.Var, name), getattr(init, name))
        self.assertIs(jt.Var.normal_, init.gauss_)
        self.assertIs(jt.Var.fill_, init.constant_)

    def test_initializers_execute_in_each_domain(self):
        init = jt.init
        np.testing.assert_array_equal(init.eye((2, 3)).numpy(), np.eye(2, 3))
        weight = jt.zeros((8, 4))
        self.assertIs(weight.constant_(2), weight)
        np.testing.assert_array_equal(weight.numpy(), np.full((8, 4), 2))
        self.assertIs(init.gauss_(weight, mean=3, std=0), weight)
        np.testing.assert_array_equal(weight.numpy(), np.full((8, 4), 3))
        for initializer in (
            init.invariant_uniform_, init.relu_invariant_gauss_,
            init.kaiming_uniform_, init.kaiming_normal_, init.xavier_uniform_,
            init.xavier_gauss_,
        ):
            initializer(weight)
            values = weight.numpy()
            self.assertEqual(values.shape, (8, 4))
            self.assertTrue(np.isfinite(values).all())
            self.assertGreater(float(np.std(values)), 0)
        self.assertIs(init.trunc_normal_(weight, a=-0.5, b=0.5), weight)
        values = weight.numpy()
        self.assertTrue(np.all(values >= -0.5))
        self.assertTrue(np.all(values <= 0.5))
