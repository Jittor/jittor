"""Initialization ownership and legacy globals without starting Jittor."""

import importlib.util
import math
from pathlib import Path
import pickle
import sys
from types import ModuleType
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "python" / "jittor" / "init"


class TestInitPackage(unittest.TestCase):
    def setUp(self):
        self.jt = ModuleType("jittor")
        self.jt.__path__ = []
        self.jt.Var = type("Var", (), {})
        self.jt.NanoVector = tuple
        self.jt._arg_policy = ModuleType("jittor._arg_policy")
        self.modules = patch.dict(sys.modules, {"jittor": self.jt})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        for name in tuple(sys.modules):
            if name == "jittor.init" or name.startswith("jittor.init."):
                del sys.modules[name]
        spec = importlib.util.spec_from_file_location(
            "jittor.init", PACKAGE / "__init__.py",
            submodule_search_locations=[str(PACKAGE)],
        )
        self.init = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.init
        self.jt.init = self.init
        spec.loader.exec_module(self.init)

    def test_public_functions_have_one_owner_and_pickle_both_paths(self):
        owners = set()
        for name in self.init.__all__:
            function = getattr(self.init, name)
            owners.add(function.__module__)
            owner = sys.modules[function.__module__]
            self.assertIs(getattr(owner, function.__name__), function)
            self.assertIs(pickle.loads(pickle.dumps(function)), function)
            legacy = ("cjittor.init\n%s\n." % name).encode("ascii")
            self.assertIs(pickle.loads(legacy), function)
        self.assertEqual(owners, {
            "jittor.init.basic", "jittor.init.scaling",
            "jittor.init._fan", "jittor.init.truncated",
        })
        self.assertFalse((PACKAGE.parent / "init.py").exists())

    def test_var_bindings_keep_public_callable_identity(self):
        names = (
            "eye_", "constant_", "zero_", "random_", "one_", "uniform_",
            "gauss_", "invariant_uniform_", "relu_invariant_gauss_",
            "kaiming_uniform_", "kaiming_normal_", "xavier_uniform_",
            "xavier_gauss_", "trunc_normal_",
        )
        for name in names:
            self.assertIs(getattr(self.jt.Var, name), getattr(self.init, name))
        self.assertIs(self.jt.Var.fill_, self.init.fill)
        self.assertIs(self.init.fill, self.init.constant_)
        self.assertIs(self.jt.Var.normal_, self.init.gauss_)

    def test_shared_fan_gain_and_scaling_math(self):
        self.assertEqual(self.init._calculate_fan_in_and_fan_out((8, 4, 3)), (12, 24))
        self.assertEqual(self.init.calculate_gain("selu"), 0.75)
        weight = type("Weight", (), {"shape": (8, 4, 3), "ndim": 3})()
        self.assertAlmostEqual(
            self.init.calculate_std(weight, "fan_out", "relu"), math.sqrt(2 / 24),
        )
        with self.assertRaises(ValueError):
            self.init._fan_for_mode((8, 4), "invalid")
