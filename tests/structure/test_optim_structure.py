"""Architecture and compatibility contracts for ``jittor.optim``."""

import ast
import inspect
import pickle
from pathlib import Path
import unittest

import jittor as jt
from jittor import nn, optim
from jittor.optim import base, schedulers
from jittor.optim.algorithms import adam, adan, rmsprop, sgd


_CLASSES = {
    "Optimizer": base.Optimizer,
    "SGD": sgd.SGD,
    "RMSprop": rmsprop.RMSprop,
    "Adam": adam.Adam,
    "AdamW": adam.AdamW,
    "Adan": adan.Adan,
    "LRScheduler": schedulers.LRScheduler,
    "LambdaLR": schedulers.LambdaLR,
}


class TestOptimStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.package = cls.repo / "python" / "jittor" / "optim"

    def test_physical_package_replaces_the_legacy_module(self):
        self.assertFalse((self.package.parent / "optim.py").exists())
        expected = {
            "__init__.py", "base.py", "schedulers.py",
        }
        self.assertEqual(
            {path.name for path in self.package.glob("*.py")}, expected,
        )
        self.assertEqual(
            {path.name for path in (self.package / "algorithms").glob("*.py")},
            {"__init__.py", "sgd.py", "rmsprop.py", "adam.py", "adan.py"},
        )

    def test_facade_preserves_the_historical_public_surface(self):
        native = {
            "jt", "np", "deepcopy", "Optimizer", "opt_grad", "SGD",
            "RMSprop", "Adam", "AdamW", "Adan", "LRScheduler", "LambdaLR",
        }
        runtime = native | {
            "LBFGS", "adam", "adamw", "lr_scheduler", "optimizer", "rmsprop",
            "sgd", "swa_utils",
        }
        self.assertEqual(set(optim._NATIVE_EXPORTS), native)
        self.assertEqual(set(optim.__all__), runtime)
        self.assertFalse({"base", "algorithms", "schedulers"} & set(optim.__all__))
        for name, implementation in _CLASSES.items():
            with self.subTest(name=name):
                self.assertIs(getattr(optim, name), implementation)
                self.assertIs(getattr(nn, name), implementation)
        self.assertIs(jt.Var.opt_grad, base.opt_grad)
        self.assertIs(optim.opt_grad, base.opt_grad)

    def test_implementations_have_real_module_paths_and_legacy_pickles_load(self):
        expected_modules = {
            "Optimizer": "jittor.optim.base",
            "SGD": "jittor.optim.algorithms.sgd",
            "RMSprop": "jittor.optim.algorithms.rmsprop",
            "Adam": "jittor.optim.algorithms.adam",
            "AdamW": "jittor.optim.algorithms.adam",
            "Adan": "jittor.optim.algorithms.adan",
            "LRScheduler": "jittor.optim.schedulers",
            "LambdaLR": "jittor.optim.schedulers",
        }
        for name, implementation in _CLASSES.items():
            with self.subTest(name=name):
                module_name = expected_modules[name]
                self.assertEqual(implementation.__module__, module_name)
                self.assertIs(pickle.loads(pickle.dumps(implementation)), implementation)
                current = pickle.dumps(implementation, protocol=0)
                legacy = current.replace(
                    ("c" + module_name + "\n").encode(),
                    b"cjittor.optim\n",
                    1,
                )
                self.assertIs(pickle.loads(legacy), implementation)

    def test_core_signatures_are_unchanged(self):
        signatures = {
            "Optimizer": "(*a, **k)",
            "SGD": (
                "(params, lr, momentum=0, weight_decay=0, dampening=0, "
                "nesterov=False)"
            ),
            "RMSprop": "(params, lr=0.001, *a, **k)",
            "Adam": "(params, lr=0.001, *a, **k)",
            "AdamW": "(params, lr=0.001, *a, **k)",
            "Adan": "(params, lr=0.001, *a, **k)",
        }
        for name, signature in signatures.items():
            with self.subTest(name=name):
                self.assertEqual(str(inspect.signature(_CLASSES[name])), signature)

    def test_package_parses_as_python37(self):
        for path in self.package.rglob("*.py"):
            with self.subTest(path=path.name):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )


if __name__ == "__main__":
    unittest.main()
