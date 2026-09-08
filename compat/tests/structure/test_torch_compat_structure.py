"""Structural contracts for the canonical torch compatibility domain."""

import ast
import importlib
import inspect
import os
import pickle
from pathlib import Path
import sys
import unittest

import jittor
import jittor.torch_compat as legacy_compat
import torch
from jittor.compat import torch as compat
from jittor.compat.torch import functional
from jittor.compat.torch import grad
from jittor.compat.torch import lr_scheduler
from jittor.compat.torch import nested
from jittor.compat.torch import optimizers
from jittor.compat.torch import serialization
from jittor.compat.torch import types

from _helpers.child_process import run_python_child


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
    def test_sys_modules_publication_has_an_exact_owner_whitelist(self):
        compat_root = Path(compat.__file__).resolve().parent.parent
        assignments = []
        mutation_calls = []
        import_fallbacks = []

        for path in sorted(compat_root.rglob("*.py")):
            if path.relative_to(compat_root).parts[0] == "tests":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            parents = {}
            for node in ast.walk(tree):
                for child in ast.iter_child_nodes(node):
                    parents[child] = node

            def owner(node):
                while node is not None:
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        return node.name
                    node = parents.get(node)
                return "<module>"

            def expression_key(node):
                if hasattr(ast, "Index") and isinstance(node, ast.Index):
                    node = node.value
                if isinstance(node, ast.Name):
                    return node.id
                if isinstance(node, (ast.Str, ast.Constant)) and isinstance(
                    getattr(node, "s", None), str
                ):
                    return repr(getattr(node, "s"))
                if isinstance(node, ast.Constant) and node.value is None:
                    return "None"
                if isinstance(node, ast.JoinedStr):
                    return "<f-string>"
                return "<dynamic>"

            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "modules":
                    value = node.value
                    if (
                        isinstance(value, ast.Call)
                        and isinstance(value.func, ast.Name)
                        and value.func.id == "__import__"
                        and value.args
                        and isinstance(value.args[0], (ast.Str, ast.Constant))
                        and getattr(value.args[0], "s", None) == "sys"
                    ):
                        import_fallbacks.append(
                            (path.relative_to(compat_root).as_posix(), node.lineno)
                        )

                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in ("clear", "pop", "popitem", "setdefault", "update")
                    and isinstance(node.func.value, ast.Attribute)
                    and node.func.value.attr == "modules"
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id in ("sys", "_sys")
                ):
                    mutation_calls.append(
                        (
                            path.relative_to(compat_root).as_posix(),
                            owner(node),
                            node.func.attr,
                            tuple(expression_key(arg) for arg in node.args),
                        )
                    )

                if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                    continue
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if not isinstance(target, ast.Subscript):
                        continue
                    modules = target.value
                    if not (
                        isinstance(modules, ast.Attribute)
                        and modules.attr == "modules"
                        and isinstance(modules.value, ast.Name)
                        and modules.value.id in ("sys", "_sys")
                    ):
                        continue
                    assignments.append(
                        (
                            path.relative_to(compat_root).as_posix(),
                            owner(node),
                            expression_key(target.slice),
                        )
                    )

        self.assertEqual(import_fallbacks, [])
        self.assertEqual(sorted(assignments), sorted([
            ("external_backend.py", "_restore_source_import_state", "name"),
            ("external_backend.py", "import_local", "key"),
            ("external_backend.py", "publish_source_module", "name"),
            ("shim/resources/stubs/torchaudio/__init__.py", "__getattr__", "<f-string>"),
            ("shim/resources/stubs/torchdata/__init__.py", "__getattr__", "<f-string>"),
            ("shim/resources/torch/__init__.py", "<module>", "__name__"),
            ("shim/runtime.py", "_activate_once", repr("torch")),
            ("shim/runtime.py", "_publish_torch_module", repr("torch")),
            ("torch/__init__.py", "_restore_namespace", "name"),
            ("triton/__init__.py", "install", "name"),
        ]))
        self.assertEqual(sorted(mutation_calls), sorted([
            ("external_backend.py", "_restore_source_import_state", "pop", ("name", "None")),
            ("external_backend.py", "import_local", "pop", ("key",)),
            ("external_backend.py", "import_local", "pop", ("key",)),
            (
                "shim/resources/stubs/torchvision/__init__.py",
                "<module>",
                "setdefault",
                (repr("torchvision.models"), "_models"),
            ),
            (
                "shim/resources/stubs/torchvision/__init__.py",
                "<module>",
                "setdefault",
                (repr("torchvision.transforms"), "transforms"),
            ),
            (
                "shim/resources/stubs/torchvision/__init__.py",
                "<module>",
                "setdefault",
                (repr("torchvision.transforms.functional"), "<dynamic>"),
            ),
            (
                "shim/resources/stubs/torchvision/__init__.py",
                "<module>",
                "setdefault",
                (repr("torchvision.utils"), "_utils"),
            ),
            ("torch/__init__.py", "_restore_namespace", "pop", ("name", "None")),
        ]))

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
            "import jittor\n"
            "before = {t: dict(vars(t)) for t in (jittor.Var, jittor.Module, jittor.Function)}\n"
            "import jittor.torch_compat as legacy\n"
            "from jittor.compat import torch as canonical\n"
            "import jittor\n"
            "assert legacy is canonical is jittor.torch_compat\n"
            "assert sys.modules['jittor.torch_compat'] is canonical\n"
            "import torch\n"
            "assert torch._torch_compat_install_complete\n"
            "assert torch._torch_compat_install_context.complete\n"
            "assert sys.modules['torch'] is torch and torch is not jittor\n"
            "assert sys.modules['torch.nn'] is torch.nn and torch.nn is not jittor.nn\n"
            "assert sys.modules['torch.nn.functional'] is torch.nn.functional\n"
            "assert torch.nn.functional is not jittor.nn.functional\n"
            "assert all(dict(vars(t)) == state for t, state in before.items())\n"
            "assert '_torch_compat_install_context' not in vars(jittor)\n"
        )
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = run_python_child(
            ["-c", code], env=env, inherit=False, merge_stderr=True, without_torch_mode=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertNotIn("torch_compat not fully installed", result.stdout)

    def test_legacy_physical_scaffolding_is_absent(self):
        package_root = Path(types.__file__).resolve().parent
        jittor_root = Path(__file__).resolve().parents[3] / "python/jittor"
        self.assertFalse((jittor_root / "torch_compat.py").exists())
        self.assertFalse((jittor_root / "_torch_compat").exists())
        self.assertTrue((package_root.parent / "__init__.py").is_file())

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
            value = getattr(torch.optim.lr_scheduler, name)
            with self.subTest(name=name):
                self.assertEqual(value.__module__, "jittor.compat.torch.lr_scheduler")
        expected = (
            (torch.optim.LBFGS, "torch.optim"),
            (torch.optim.LBFGS.step, "jittor.compat.torch.optimizer_api"),
            (torch.optim.swa_utils.SWALR, "jittor.compat.torch.lr_scheduler"),
            (torch.optim.swa_utils.AveragedModel, "jittor.compat.torch.lr_scheduler"),
            (torch.optim.swa_utils.get_swa_avg_fn, "jittor.compat.torch.lr_scheduler"),
            (torch.optim.swa_utils.get_ema_avg_fn, "jittor.compat.torch.lr_scheduler"),
            (torch.optim.swa_utils.update_bn, "jittor.compat.torch.lr_scheduler"),
            (torch.optim.Optimizer.state.fget, "jittor.compat.torch.optimizer_api"),
            (torch.optim.Adam.__init__, "jittor.compat.torch.optimizer_api"),
            (torch.optim.Adam.step, "jittor.compat.torch.optimizer_api"),
            (torch.optim.AdamW.step, "jittor.compat.torch.optimizer_api"),
            (torch.optim.SGD.step, "jittor.compat.torch.optimizer_api"),
        )
        for value, module_name in expected:
            with self.subTest(name=value.__name__):
                self.assertEqual(value.__module__, module_name)
        self.assertTrue(issubclass(torch.optim.LBFGS, torch.optim.Optimizer))
        self.assertIsNot(torch.optim.Optimizer, jittor.optim.Optimizer)
        self.assertTrue(jittor.optim.Optimizer.__module__.startswith("jittor.optim"))

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
        from jittor.compat.shim.runtime import activation_status
        status = activation_status(jittor)
        self.assertTrue(status.active)
        self.assertIs(status.result["torch"], torch)
        native = {name: getattr(jittor, name) for name in ("Var", "Module", "Function", "grad", "no_grad")}
        before = {
            "grad": torch.grad,
            "no_grad": torch.no_grad,
            "interpolate": torch.nn.functional.interpolate,
        }
        self.assertIs(compat.install(torch), torch)
        self.assertIs(compat.install(torch), torch)
        self.assertIs(torch.grad, before["grad"])
        self.assertIs(torch.no_grad, before["no_grad"])
        self.assertIs(torch.nn.functional.interpolate, before["interpolate"])
        for name, value in native.items():
            self.assertIs(getattr(jittor, name), value)
        with self.assertRaisesRegex(RuntimeError, "independent TorchNamespace"):
            compat.install(jittor)

    def test_dtype_objects_preserve_jittor_constructors(self):
        self.assertTrue(callable(jittor.float32))
        self.assertTrue(callable(jittor.int32))
        fp = jittor.float32([1, 2])
        integer = jittor.int32([1, 2])
        self.assertIsInstance(fp, jittor.Var)
        self.assertIsInstance(integer, jittor.Var)
        self.assertNotIsInstance(jittor.float32, str)
        self.assertNotIsInstance(jittor.int32, str)
        self.assertNotIsInstance(fp.dtype, torch.dtype)
        self.assertEqual(str(fp.dtype), "float32")
        self.assertEqual(str(integer.dtype), "int32")
        self.assertEqual(fp.numpy().tolist(), [1.0, 2.0])
        self.assertEqual(integer.numpy().tolist(), [1, 2])
        tensor = torch.tensor([1, 2], dtype=torch.float32)
        self.assertIs(type(tensor), torch.Tensor)
        self.assertIs(tensor.dtype, torch.float32)
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertIs(pickle.loads(pickle.dumps(tensor.dtype)), torch.float32)
        self.assertFalse(callable(torch.float32))

    def test_real_nn_functional_receives_torch_semantics(self):
        jittor_functional = importlib.import_module("jittor.nn.functional")
        self.assertIs(jittor.nn.functional, jittor_functional)
        self.assertIs(sys.modules["torch.nn.functional"], torch.nn.functional)
        self.assertIsNot(torch.nn.functional, jittor_functional)
        self.assertEqual(inspect.signature(jittor_functional.interpolate).parameters["mode"].default, "bilinear")
        signature = inspect.signature(torch.nn.functional.interpolate)
        self.assertEqual(signature.parameters["mode"].default, "nearest")
        x = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
        output = torch.nn.functional.interpolate(x, scale_factor=2)
        self.assertIs(type(output), torch.Tensor)
        actual = output.numpy()
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
        self.assertIs(sys.modules["torch.nn.modules"], torch.nn.modules)
        self.assertIsNot(torch.nn.modules, jittor_modules)
        self.assertEqual(jittor_modules.__name__, "jittor.nn.modules")
        self.assertIs(
            sys.modules["torch.nn.modules.module"].Module,
            torch.nn.Module,
        )
        self.assertIs(jittor_modules.Module, jittor.Module)
        self.assertIsNot(torch.nn.Module, jittor.Module)

    def test_domain_modules_import_the_root_directly(self):
        package_root = Path(types.__file__).resolve().parent
        for name in (
            "functional.py",
            "grad.py",
            "lr_scheduler.py",
            "nested.py",
            "optimizers.py",
            "serialization/portable.py",
            "serialization/torch_archive.py",
            "serialization/safetensors.py",
            "types.py",
        ):
            path = package_root / name
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn(".runtime import", source)
                self.assertNotIn("preserve_facade_origins", source)
                self.assertIn("import jittor as jt", source)

    def test_package_discovery_includes_only_canonical_compat_packages(self):
        package_root = Path(types.__file__).resolve().parent
        repo_root = Path(__file__).resolve().parents[3]
        if not (repo_root / "pyproject.toml").is_file():
            self.skipTest("packaging metadata is only available in a source checkout")
        from setuptools import find_packages

        native_packages = find_packages(where=str(repo_root / "python"))
        self.assertNotIn("jittor.compat", native_packages)
        packages = ["jittor.compat"] + [
            "jittor.compat." + name
            for name in find_packages(where=str(repo_root / "compat"))
        ]
        self.assertIn("jittor.compat", packages)
        self.assertIn("jittor.compat.torch", packages)
        self.assertIn("jittor.compat.torch.serialization", packages)
        self.assertNotIn("jittor._torch_compat", packages)

    def test_installation_order_remains_stable(self):
        self.assertEqual([name for name, _installer in compat._REQUIRED_STEPS], [
            "core",
            "tensor.base",
            "tensor.methods",
            "nn",
            "optim",
            "autograd",
            "cuda",
            "distributed",
            "core.extended",
            "serialization",
            "utilities",
            "utilities.runtime-knobs",
            "data",
            "distributions",
            "compiler",
            "numerical",
            "numerical.signal",
            "autograd.module-keys",
            "nn.module-keys",
            "optim.module-keys",
            "distributions.module-keys",
            "compiler.module-keys",
            "numerical.module-keys",
            "utilities.module-keys",
        ])
        self.assertEqual(
            [name for name, _installer in compat._OPTIONAL_STEPS],
            [
                "optional.torchmetrics",
                "optional.transformers",
                "optional.tensordict",
                "optional.safetensors",
                "optional.flash-attn",
                "optional.vllm",
            ],
        )

    def test_canonical_module_line_budgets(self):
        package_root = Path(types.__file__).resolve().parent
        source = Path(compat.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(set(functions), {
            "_install_optim_and_schedulers", "_install_serialization",
            "_install_optional_safetensors", "_install_optional_vllm",
            "_same_namespace", "_restore_namespace", "_abandon", "install",
        })
        # Separate the parent-aware transaction coordinator from composition:
        # API implementations must remain in their domain modules, not grow
        # inside install(). Each responsibility has a 300-line ceiling.
        install = functions["install"]
        install_lines = install.end_lineno - install.lineno + 1
        self.assertLessEqual(install_lines, 300)
        self.assertLessEqual(len(source.splitlines()) - install_lines, 300)
        self.assertFalse(any(isinstance(node, (ast.FunctionDef, ast.ClassDef))
                             for statement in install.body for node in ast.walk(statement)))
        for path in package_root.glob("*.py"):
            if path.name == "__init__.py":
                continue
            with self.subTest(path=path.name):
                self.assertLessEqual(
                    len(path.read_text(encoding="utf-8").splitlines()),
                    800,
                    f"split {path.name} before adding more implementation",
                )
        installers = package_root / "installers"
        expected = {
            "core.py", "tensor", "factories.py", "autograd.py", "nn",
            "nn_init.py", "cuda", "distributed.py", "data.py",
            "distributions.py", "numerical", "compiler.py", "utilities.py",
        }
        self.assertEqual(
            {path.name for path in installers.iterdir()
             if path.name != "__init__.py" and (path.suffix == ".py" or
                (path.is_dir() and (path / "__init__.py").is_file()))},
            expected,
        )
        for path in installers.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            with self.subTest(installer=path.name):
                # No line budget: it fails on growth, not on a boundary violation.
                self.assertNotIn("exec(", source)
                self.assertNotIn("sys.modules", source)


if __name__ == "__main__":
    unittest.main()
