"""Structural contracts for the canonical torch compatibility domain."""

import ast
import importlib
import inspect
import os
import pickle
from pathlib import Path
import subprocess
import sys
import unittest

import jittor
import jittor.torch_compat as legacy_compat
from jittor.compat import torch as compat
from jittor.compat.torch import functional
from jittor.compat.torch import grad
from jittor.compat.torch import lr_scheduler
from jittor.compat.torch import nested
from jittor.compat.torch import optimizers
from jittor.compat.torch import serialization
from jittor.compat.torch import types


def _class_callables(cls):
    for member in vars(cls).values():
        if isinstance(member, (classmethod, staticmethod)):
            yield member.__func__
        elif isinstance(member, property):
            for value in (member.fget, member.fset, member.fdel):
                if value is not None:
                    yield value
        elif callable(member):
            yield member


class TestTorchCompatStructure(unittest.TestCase):
    def test_legacy_import_is_the_canonical_module(self):
        self.assertIs(legacy_compat, compat)
        self.assertIs(jittor.torch_compat, compat)
        self.assertIs(sys.modules["jittor.torch_compat"], compat)
        self.assertIs(importlib.import_module("jittor.torch_compat"), compat)
        self.assertIs(importlib.import_module("jittor.compat.torch"), compat)
        self.assertEqual(compat.__name__, "jittor.compat.torch")

    def test_fresh_legacy_import_installs_without_warning(self):
        code = (
            "import sys\n"
            "import jittor.torch_compat as legacy\n"
            "from jittor.compat import torch as canonical\n"
            "import jittor\n"
            "assert legacy is canonical is jittor.torch_compat\n"
            "assert sys.modules['jittor.torch_compat'] is canonical\n"
            "assert jittor._torch_compat_install_complete\n"
            "assert canonical._INSTALL_COMPLETE\n"
            "assert sys.modules['torch'] is jittor\n"
            "assert sys.modules['torch.nn'] is jittor.nn\n"
            "assert sys.modules['torch.nn.functional'] is jittor.nn.functional\n"
        )
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("torch_compat not fully installed", result.stdout)

    def test_legacy_physical_scaffolding_is_absent(self):
        package_root = Path(types.__file__).resolve().parent
        jittor_root = package_root.parents[1]
        self.assertFalse((jittor_root / "torch_compat.py").exists())
        self.assertFalse((jittor_root / "_torch_compat").exists())
        self.assertTrue((jittor_root / "compat" / "__init__.py").is_file())

    def test_canonical_package_reexports_domain_symbols(self):
        expected = {
            "_clip_grad_norm_device": grad._clip_grad_norm_device,
            "_GradScaler": grad._GradScaler,
            "_install_lr_scheduler": lr_scheduler._install_lr_scheduler,
            "_install_optimizers": optimizers._install_optimizers,
            "_install_safetensors_shim": serialization._install_safetensors_shim,
            "_NestedTensor": nested._NestedTensor,
            "_torch_norm_impl": functional._torch_norm_impl,
            "device": types.device,
            "dtype": types.dtype,
        }
        for name, implementation in expected.items():
            with self.subTest(name=name):
                self.assertIs(getattr(compat, name), implementation)

    def test_new_definitions_report_canonical_origins(self):
        expected = {
            compat.dtype: "jittor.compat.torch.types",
            compat.device: "jittor.compat.torch.types",
            compat._GradScaler: "jittor.compat.torch.grad",
            compat._NestedTensor: "jittor.compat.torch.nested",
            compat._TorchSize: "jittor.compat.torch.nested",
            compat._torch_norm_impl: "jittor.compat.torch.functional",
        }
        for value, module_name in expected.items():
            with self.subTest(name=value.__name__):
                self.assertEqual(value.__module__, module_name)
                if isinstance(value, type):
                    for member in _class_callables(value):
                        self.assertEqual(member.__module__, module_name)

    def test_installed_definitions_report_canonical_origins(self):
        scheduler_names = (
            "LRScheduler", "LambdaLR", "MultiplicativeLR", "ConstantLR",
            "LinearLR", "StepLR", "MultiStepLR", "ExponentialLR",
            "CosineAnnealingLR", "PolynomialLR", "OneCycleLR",
            "SequentialLR", "ChainedScheduler", "ReduceLROnPlateau",
        )
        for name in scheduler_names:
            value = getattr(jittor.optim.lr_scheduler, name)
            with self.subTest(name=name):
                self.assertEqual(value.__module__, "jittor.compat.torch.lr_scheduler")
        expected = (
            (jittor.optim.LBFGS, "jittor.compat.torch.optimizers"),
            (jittor.optim.swa_utils.SWALR, "jittor.compat.torch.lr_scheduler"),
            (jittor.optim.swa_utils.AveragedModel, "jittor.compat.torch.lr_scheduler"),
            (jittor.optim.swa_utils.get_swa_avg_fn, "jittor.compat.torch.lr_scheduler"),
            (jittor.optim.swa_utils.get_ema_avg_fn, "jittor.compat.torch.lr_scheduler"),
            (jittor.optim.swa_utils.update_bn, "jittor.compat.torch.lr_scheduler"),
            (jittor.optim.Optimizer.state.fget, "jittor.compat.torch.optimizers"),
            (jittor.optim.Adam.__init__, "jittor.compat.torch.optimizers"),
            (jittor.optim.Adam.step, "jittor.compat.torch.optimizers"),
            (jittor.optim.AdamW.step, "jittor.compat.torch.optimizers"),
            (jittor.optim.SGD.step, "jittor.compat.torch.optimizers"),
        )
        for value, module_name in expected:
            with self.subTest(name=value.__name__):
                self.assertEqual(value.__module__, module_name)

    def test_no_public_symbol_exposes_a_legacy_origin(self):
        for value in compat._COMPAT_PUBLIC_SYMBOLS:
            module_name = getattr(value, "__module__", "")
            with self.subTest(name=value.__name__):
                self.assertTrue(module_name.startswith("jittor.compat.torch"))
                self.assertNotEqual(module_name, "jittor.torch_compat")

    def test_public_symbols_are_pickleable_from_canonical_origins(self):
        for value in compat._COMPAT_PUBLIC_SYMBOLS:
            with self.subTest(name=value.__name__):
                self.assertIs(pickle.loads(pickle.dumps(value)), value)

    def test_legacy_pickle_global_paths_still_resolve(self):
        for name, value in (
            ("_TorchSize", compat._TorchSize),
            ("_GradScaler", compat._GradScaler),
            ("dtype", compat.dtype),
            ("device", compat.device),
        ):
            payload = ("cjittor.torch_compat\n" + name + "\n.").encode("ascii")
            with self.subTest(name=name):
                self.assertIs(pickle.loads(payload), value)

    def test_install_is_idempotent(self):
        self.assertTrue(jittor._torch_compat_install_complete)
        self.assertTrue(compat._INSTALL_COMPLETE)
        before = {
            "grad": jittor.grad,
            "no_grad": jittor.no_grad,
            "interpolate": jittor.nn.functional.interpolate,
        }
        self.assertIs(compat.install(jittor), jittor)
        self.assertIs(compat.install(jittor), jittor)
        self.assertIs(jittor.grad, before["grad"])
        self.assertIs(jittor.no_grad, before["no_grad"])
        self.assertIs(jittor.nn.functional.interpolate, before["interpolate"])

    def test_dtype_objects_preserve_jittor_constructors(self):
        self.assertTrue(callable(jittor.float32))
        self.assertTrue(callable(jittor.int32))
        fp = jittor.float32([1, 2])
        integer = jittor.int32([1, 2])
        self.assertIsInstance(fp, jittor.Var)
        self.assertIsInstance(integer, jittor.Var)
        self.assertEqual(str(fp.dtype), "float32")
        self.assertEqual(str(integer.dtype), "int32")

    def test_real_nn_functional_receives_torch_semantics(self):
        jittor_functional = importlib.import_module("jittor.nn.functional")
        self.assertIs(jittor.nn.functional, jittor_functional)
        self.assertIs(sys.modules["torch.nn.functional"], jittor_functional)
        signature = inspect.signature(jittor_functional.interpolate)
        self.assertEqual(signature.parameters["mode"].default, "nearest")
        x = jittor.array([[[[1.0, 2.0], [3.0, 4.0]]]])
        actual = jittor_functional.interpolate(x, scale_factor=2).numpy()
        expected = [
            [
                [1.0, 1.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 2.0],
                [3.0, 3.0, 4.0, 4.0],
                [3.0, 3.0, 4.0, 4.0],
            ]
        ]
        self.assertEqual(actual[0].tolist(), expected)

    def test_real_nn_modules_package_is_not_replaced(self):
        jittor_modules = importlib.import_module("jittor.nn.modules")
        self.assertIs(jittor.nn.modules, jittor_modules)
        self.assertIs(sys.modules["torch.nn.modules"], jittor_modules)
        self.assertEqual(jittor_modules.__name__, "jittor.nn.modules")
        self.assertIs(
            sys.modules["torch.nn.modules.module"].Module,
            jittor.Module,
        )

    def test_domain_modules_import_the_root_directly(self):
        package_root = Path(types.__file__).resolve().parent
        for path in package_root.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn(".runtime import", source)
                self.assertNotIn("preserve_facade_origins", source)
                self.assertIn("import jittor as jt", source)

    def test_package_discovery_includes_only_canonical_compat_packages(self):
        package_root = Path(types.__file__).resolve().parent
        repo_root = package_root.parents[3]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages

        packages = find_packages(where=str(repo_root / "python"))
        self.assertIn("jittor.compat", packages)
        self.assertIn("jittor.compat.torch", packages)
        self.assertNotIn("jittor._torch_compat", packages)

    def test_installation_order_remains_stable(self):
        source_path = Path(compat.__file__).resolve()
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        install = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "install"
        )
        calls = []

        class InstallCallVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                return

            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and (
                    node.func.id.startswith("_install_")
                    or node.func.id == "_wrap_constructors"
                ):
                    calls.append(node.func.id)
                self.generic_visit(node)

        visitor = InstallCallVisitor()
        for statement in install.body:
            visitor.visit(statement)
        self.assertEqual(calls, [
            "_wrap_constructors",
            "_install_random_and_linspace",
            "_install_reductions",
            "_install_nn_extras",
            "_install_cuda",
            "_install_version",
            "_install_distributed",
            "_install_tensor_methods",
            "_install_misc",
            "_install_torchdata_stateful_dataloader",
            "_install_torchmetrics_fastpaths",
            "_install_optimizers",
            "_install_lr_scheduler",
            "_install_autograd_function",
            "_install_autograd",
            "_install_tensordict_compat",
            "_install_safetensors_shim",
            "_install_flash_attn_shim",
        ])

    def test_canonical_module_line_budgets(self):
        package_root = Path(types.__file__).resolve().parent
        self.assertLessEqual(
            len(Path(compat.__file__).read_text(encoding="utf-8").splitlines()),
            8700,
        )
        for path in package_root.glob("*.py"):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    800,
                    f"split {path.name} before adding more implementation",
                )


if __name__ == "__main__":
    unittest.main()
