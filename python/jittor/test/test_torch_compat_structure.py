"""Structural contracts for the modular torch compatibility implementation."""

import ast
import pickle
from pathlib import Path
import types as python_types
import unittest

import jittor
import jittor.torch_compat as compat
from jittor._torch_compat import functional
from jittor._torch_compat import grad
from jittor._torch_compat import lr_scheduler
from jittor._torch_compat import nested
from jittor._torch_compat import optimizers
from jittor._torch_compat import serialization
from jittor._torch_compat import types


class _ModuleImportVisitor(ast.NodeVisitor):
    """Inspect imports executed at module load while ignoring function bodies."""

    def __init__(self):
        self.violations = []

    def visit_FunctionDef(self, node):
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name == "jittor" or alias.name.startswith("jittor."):
                self.violations.append((node.lineno, alias.name))

    def visit_ImportFrom(self, node):
        if node.level >= 2:
            target = "." * node.level + (node.module or "")
            self.violations.append((node.lineno, target))
        elif node.level == 0 and node.module and (
            node.module == "jittor" or node.module.startswith("jittor.")
        ):
            self.violations.append((node.lineno, node.module))


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
    def test_private_implementation_package_is_not_shadowed(self):
        self.assertIsInstance(jittor._torch_compat, python_types.ModuleType)
        self.assertEqual(jittor._torch_compat.__name__, "jittor._torch_compat")

    def test_facade_reexports_implementation_symbols(self):
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

    def test_public_class_module_names_remain_stable(self):
        for cls in (
            compat.dtype,
            compat.device,
            compat._GradScaler,
            compat._NestedTensor,
            compat._TorchSize,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.__module__, "jittor.torch_compat")
                for member in _class_callables(cls):
                    self.assertEqual(member.__module__, "jittor.torch_compat")

    def test_facade_callable_module_names_remain_stable(self):
        for name, value in vars(compat).items():
            module_name = getattr(value, "__module__", "")
            if module_name.startswith("jittor._torch_compat"):
                self.fail(f"{name} exposes private implementation module {module_name}")

    def test_moved_facade_symbols_remain_pickleable_by_identity(self):
        for value in compat._COMPAT_FACADE_SYMBOLS:
            with self.subTest(name=value.__name__):
                self.assertIs(pickle.loads(pickle.dumps(value)), value)

    def test_installed_optimizer_module_names_remain_stable(self):
        scheduler_names = (
            "LRScheduler", "LambdaLR", "MultiplicativeLR", "ConstantLR",
            "LinearLR", "StepLR", "MultiStepLR", "ExponentialLR",
            "CosineAnnealingLR", "PolynomialLR", "OneCycleLR",
            "SequentialLR", "ChainedScheduler", "ReduceLROnPlateau",
        )
        scheduler_classes = []
        for name in scheduler_names:
            with self.subTest(name=name):
                cls = getattr(jittor.optim.lr_scheduler, name)
                scheduler_classes.append(cls)
                self.assertEqual(cls.__module__, "jittor.torch_compat")
        scheduler_classes.extend((
            jittor.optim.swa_utils.SWALR,
            jittor.optim.swa_utils.AveragedModel,
        ))
        for cls in scheduler_classes:
            for member in _class_callables(cls):
                with self.subTest(cls=cls.__name__, member=member.__name__):
                    self.assertEqual(member.__module__, "jittor.torch_compat")
        for value in (
            jittor.optim.LBFGS,
            jittor.optim.swa_utils.SWALR,
            jittor.optim.swa_utils.AveragedModel,
            jittor.optim.swa_utils.get_swa_avg_fn,
            jittor.optim.swa_utils.get_ema_avg_fn,
            jittor.optim.swa_utils.update_bn,
            jittor.optim.swa_utils.get_swa_avg_fn(),
            jittor.optim.swa_utils.get_ema_avg_fn(),
            jittor.optim.Optimizer.state.fget,
            jittor.optim.Adam.__init__,
            jittor.optim.Adam.step,
            jittor.optim.AdamW.step,
            jittor.optim.SGD.step,
        ):
            with self.subTest(name=value.__name__):
                self.assertEqual(value.__module__, "jittor.torch_compat")
        for cls_name in ("Optimizer", "Adam", "AdamW", "SGD", "RMSprop", "Adan", "LBFGS"):
            cls = getattr(jittor.optim, cls_name)
            for member in _class_callables(cls):
                with self.subTest(cls=cls_name, member=member.__name__):
                    self.assertFalse(
                        member.__module__.startswith("jittor._torch_compat"),
                        f"{cls_name}.{member.__name__} exposes {member.__module__}",
                    )

    def test_safetensors_public_callables_keep_facade_origin(self):
        try:
            import safetensors.torch as safetensors_torch
        except ImportError:
            self.skipTest("safetensors is optional")
        for value in (
            safetensors_torch.safe_open,
            safetensors_torch.load,
            safetensors_torch.load_file,
            safetensors_torch.save,
            safetensors_torch.save_file,
        ):
            with self.subTest(name=value.__name__):
                self.assertEqual(value.__module__, "jittor.torch_compat")

    def test_torch_size_pickle_contract_remains_stable(self):
        original = compat._TorchSize((2, 3, 4))
        restored = pickle.loads(pickle.dumps(original))
        self.assertIs(type(restored), compat._TorchSize)
        self.assertEqual(restored, original)
        self.assertEqual(restored.numel(), 24)

    def test_private_modules_do_not_import_root_at_module_scope(self):
        package_root = Path(types.__file__).resolve().parent
        for path in package_root.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = _ModuleImportVisitor()
            visitor.visit(tree)
            with self.subTest(path=path.name):
                self.assertFalse(
                    visitor.violations,
                    f"{path.name} adds root import(s) at module load: "
                    f"{visitor.violations}",
                )

    def test_import_guard_covers_nested_module_scope_and_parent_imports(self):
        tree = ast.parse(
            "try:\n"
            "    import jittor\n"
            "except ImportError:\n"
            "    from .. import nn\n"
        )
        visitor = _ModuleImportVisitor()
        visitor.visit(tree)
        self.assertEqual(
            visitor.violations,
            [(2, "jittor"), (4, "..")],
        )

    def test_setup_declares_private_implementation_package(self):
        package_root = Path(types.__file__).resolve().parent
        setup_path = package_root.parents[2] / "setup.py"
        if not setup_path.is_file():
            self.skipTest("setup.py is only available in a source checkout")

        tree = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
        packages = None
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function_name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if function_name != "setup":
                continue
            for keyword in node.keywords:
                if keyword.arg == "packages":
                    packages = ast.literal_eval(keyword.value)
                    break
        self.assertIsNotNone(packages, "setup.py must declare packages explicitly")
        self.assertIn("jittor._torch_compat", packages)

    def test_facade_installation_order_remains_stable(self):
        package_root = Path(types.__file__).resolve().parent
        facade = package_root.parent / "torch_compat.py"
        tree = ast.parse(facade.read_text(encoding="utf-8"), filename=str(facade))
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

    def test_facade_and_implementation_line_budgets(self):
        package_root = Path(types.__file__).resolve().parent
        facade = package_root.parent / "torch_compat.py"
        self.assertLessEqual(len(facade.read_text(encoding="utf-8").splitlines()), 8700)
        for path in package_root.glob("*.py"):
            with self.subTest(path=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    800,
                    f"split {path.name} before adding more implementation",
                )


if __name__ == "__main__":
    unittest.main()
