"""Static boundaries for Stage 7 compatibility composition."""

from __future__ import print_function

import ast
import json
import os
import pickle
from pathlib import Path
import tempfile
import unittest

from _helpers.child_process import run_python_child


class TestRuntimeCompositionStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(__file__).resolve().parents[2]
        cls.jittor = cls.repo / "python" / "jittor"
        cls.compat = cls.jittor / "compat"

    def test_root_contains_only_preflight_and_post_core_composition(self):
        path = self.jittor / "__init__.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definitions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        forbidden = {
            "_TorchShimFlagsProxy",
            "_apply_external_runtime_patches",
            "_install_torch_shim_runtime",
        }
        self.assertFalse(definitions & forbidden)
        self.assertFalse(any(name.startswith("_jt_torch_") for name in definitions))
        self.assertFalse(definitions)
        # No line budget here. "The root defines nothing" (above) and the import
        # ordering (below) are the architecture contract; a line count is a proxy
        # that goes red when someone adds a necessary comment and stays green when
        # someone adds a wrong one.
        self.assertNotIn("jittor.torch_shim", source)
        self.assertIn("prepare_import_environment as _prepare_compat_import", source)
        self.assertIn("if _compat_preflight_result.active:", source)
        self.assertIn("_configure_compat_math_flags(_sys.modules[__name__])", source)
        self.assertIn("from ._runtime.core_api import *", source)
        self.assertIn("compose as _compose_compat_runtime", source)
        self.assertIn("JITTOR_TORCH_STRICT_BOOTSTRAP", source)
        self.assertIn("strict=_compat_is_truthy(", source)
        self.assertLess(
            source.index("_prepare_compat_import("),
            source.index("from jittor_utils import lock"),
        )
        self.assertLess(
            source.index("_configure_compat_math_flags(_sys.modules[__name__])"),
            source.index("from ._runtime.core_api import *"),
        )
        self.assertLess(
            source.index("from ._runtime.core_api import *"),
            source.index("from . import nn"),
        )
        self.assertLess(
            source.index("from . import nn"),
            source.index("_compose_compat_runtime("),
        )

        runtime_source = (self.compat / "runtime.py").read_text(encoding="utf-8")
        self.assertIn("torch_compat_requested(root_module, preflight)", runtime_source)
        self.assertIn("if torch_mode:", runtime_source)

    def test_core_api_identity_and_legacy_pickle_paths_are_stable(self):
        import jittor
        from jittor._runtime import core_api

        for name in (
            "Module",
            "Function",
            "flag_scope",
            "array",
            "make_module",
        ):
            with self.subTest(name=name):
                self.assertIs(getattr(jittor, name), getattr(core_api, name))
        self.assertIs(jittor.flags, core_api.flags)
        for name in ("Module", "Function"):
            implementation = getattr(jittor, name)
            current = pickle.dumps(implementation, protocol=0)
            legacy = current.replace(
                b"cjittor._runtime.core_api\n",
                b"cjittor\n",
                1,
            )
            self.assertIs(pickle.loads(legacy), implementation)

    def test_compat_composition_keeps_native_core_implementations_available(self):
        import jittor
        from jittor._runtime import core_api

        self.assertIsNot(jittor.grad, core_api.grad)
        self.assertEqual(
            jittor.grad.__module__,
            "jittor.compat.torch.installers.tensor",
        )
        self.assertIsNot(jittor.save, core_api.save)
        self.assertIsNot(jittor.load, core_api.load)
        required = [
            report for report in jittor._compat_composition_report.torch_reports if report.required
        ]
        self.assertTrue(required)
        self.assertTrue(all(report.status == "complete" for report in required))

    def _run_mode_probe(self, source, torch_mode=False):
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["nvcc_path"] = ""
        if torch_mode:
            env["JITTOR_TORCH_SHIM"] = "1"
        else:
            env.pop("JITTOR_TORCH_SHIM", None)
            env.pop("JITTOR_TORCH_PROJECT_ROOT", None)
            env.pop("JITTOR_TORCH_RUNTIME_ROOT", None)
        result = run_python_child(
            ["-c", source], env=env, inherit=False, merge_stderr=True)
        self.assertEqual(result.returncode, 0, result.stdout)
        line = next(
            line for line in result.stdout.splitlines() if line.startswith("RESULT=")
        )
        return json.loads(line[len("RESULT="):])

    def test_plain_jittor_preserves_native_data_and_namespace(self):
        result = self._run_mode_probe(r'''
import json
import sys
import numpy as np
import jittor as jt

x = jt.array([1, 2, 3])
x.data[1] = 7
c = jt.ones(10)
jt.sync_all()
with jt.profile_scope() as report:
    b = c - 1
    assert b.data[1] == 0
print("RESULT=" + json.dumps({
    "data_is_numpy": isinstance(x.data, np.ndarray),
    "profile_entries": len(report),
    "shared_write": x.numpy().tolist(),
    "torch_registered": "torch" in sys.modules,
    "torch_installed": bool(getattr(jt, "_torch_compat_install_complete", False)),
}))
''')
        self.assertEqual(
            result,
            {
                "data_is_numpy": True,
                "profile_entries": 2,
                "shared_write": [1, 7, 3],
                "torch_registered": False,
                "torch_installed": False,
            },
        )

    def test_explicit_torch_mode_uses_detached_data_alias(self):
        result = self._run_mode_probe(r'''
import json
import sys
import jittor as jt

x = jt.array([1.0, 2.0])
data = x.data
data.fill_(3.0)
print("RESULT=" + json.dumps({
    "data_is_var": isinstance(data, jt.Var),
    "data_is_distinct": data is not x,
    "data_is_detached": data.is_stop_grad(),
    "shared_write": x.numpy().tolist(),
    "torch_is_jittor": sys.modules.get("torch") is jt,
    "torch_installed": jt._torch_compat_install_complete,
}))
''', torch_mode=True)
        self.assertEqual(
            result,
            {
                "data_is_var": True,
                "data_is_distinct": True,
                "data_is_detached": True,
                "shared_write": [3.0, 3.0],
                "torch_is_jittor": True,
                "torch_installed": True,
            },
        )

    def test_indirect_deployed_torch_import_does_not_switch_native_mode(self):
        torch_source = (
            self.compat / "shim" / "resources" / "torch_init.py"
        ).read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            torch_dir = root / "torch"
            triton_dir = root / "triton"
            torch_dir.mkdir()
            triton_dir.mkdir()
            (torch_dir / "__init__.py").write_text(torch_source, encoding="utf-8")
            (triton_dir / "__init__.py").write_text(
                "import torch\n__version__ = 'test-real-triton'\n",
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["nvcc_path"] = ""
            env.pop("JITTOR_TORCH_SHIM", None)
            env.pop("JITTOR_TORCH_PROJECT_ROOT", None)
            env.pop("JITTOR_TORCH_RUNTIME_ROOT", None)
            env["PYTHONPATH"] = os.pathsep.join(
                [str(root), str(self.repo / "python"), env.get("PYTHONPATH", "")]
            ).rstrip(os.pathsep)

            native = run_python_child(
                ["-c", r'''
import json, sys
import jittor as jt
print("RESULT=" + json.dumps({
    "torch_installed": "_torch_compat_install_context" in jt.__dict__,
    "median_owner": jt.median.__module__,
    "triton_is_shim": bool(getattr(sys.modules.get("triton"), "__triton_shim__", False)),
}))
'''],
                env=env, inherit=False, merge_stderr=True)
            self.assertEqual(native.returncode, 0, native.stdout)
            native_line = next(
                line for line in native.stdout.splitlines() if line.startswith("RESULT=")
            )
            self.assertEqual(
                json.loads(native_line[len("RESULT="):]),
                {
                    "torch_installed": False,
                    "median_owner": "jittor.misc.tensor_ops",
                    "triton_is_shim": True,
                },
            )

            explicit = run_python_child(
                ["-c", r'''
import json, sys
import torch
import jittor as jt
print("RESULT=" + json.dumps({
    "torch_is_jittor": torch is jt and sys.modules.get("torch") is jt,
    "torch_installed": jt._torch_compat_install_complete,
}))
'''],
                env=env, inherit=False, merge_stderr=True)
            self.assertEqual(explicit.returncode, 0, explicit.stdout)
            explicit_line = next(
                line for line in explicit.stdout.splitlines() if line.startswith("RESULT=")
            )
            self.assertEqual(
                json.loads(explicit_line[len("RESULT="):]),
                {"torch_is_jittor": True, "torch_installed": True},
            )

            late = run_python_child(
                ["-c", r'''
import json, sys
import numpy as np
import jittor as jt
native_data = jt.ones(2).data
import torch
torch_data = jt.ones(2).data
print("RESULT=" + json.dumps({
    "native_data_is_numpy": isinstance(native_data, np.ndarray),
    "torch_data_is_var": isinstance(torch_data, jt.Var),
    "torch_is_jittor": torch is jt and sys.modules.get("torch") is jt,
}))
'''],
                env=env, inherit=False, merge_stderr=True)
            self.assertEqual(late.returncode, 0, late.stdout)
            late_line = next(
                line for line in late.stdout.splitlines() if line.startswith("RESULT=")
            )
            self.assertEqual(
                json.loads(late_line[len("RESULT="):]),
                {
                    "native_data_is_numpy": True,
                    "torch_data_is_var": True,
                    "torch_is_jittor": True,
                },
            )

    def test_moved_scope_state_stays_synchronized_with_the_root(self):
        import jittor
        from jittor._runtime import core_api

        self.assertIsNone(core_api.single_log_capture)
        self.assertIsNone(jittor.single_log_capture)
        with jittor.log_capture_scope(log_v=0):
            self.assertIs(core_api.single_log_capture, True)
            self.assertIs(jittor.single_log_capture, True)
        self.assertIsNone(core_api.single_log_capture)
        self.assertIsNone(jittor.single_log_capture)

        class FakeMPI(object):
            def __init__(self):
                self.state = True

            def get_state(self):
                return self.state

            def set_state(self, state):
                self.state = state

            def world_rank(self):
                return 0

        old_mpi = core_api.mpi
        old_compile_in_mpi = core_api.compile_extern.in_mpi
        fake_mpi = FakeMPI()
        try:
            core_api.mpi = fake_mpi
            # Write the ONE owner and check every reader follows. Setting all
            # three by hand (as this used to) passes just as happily when the
            # three are independent snapshots, so it could not catch the bug it
            # looked like it was guarding: core_api kept its own `in_mpi` copy
            # from `from jittor import *`, and anything that later turned
            # distributed on left that copy stale -- with
            # Module.mpi_param_broadcast() reading the stale one and silently
            # broadcasting nothing. Assigning jittor.in_mpi / core_api.in_mpi
            # here would also shadow the module __getattr__ that now serves
            # them, putting the snapshot back. 6.B15.
            core_api.compile_extern.in_mpi = True
            scope_type = getattr(core_api, "__single_process_scope")
            with scope_type(rank=0) as selected:
                self.assertTrue(selected)
                self.assertFalse(core_api.in_mpi)
                self.assertFalse(jittor.in_mpi)
                self.assertFalse(core_api.compile_extern.in_mpi)
            self.assertTrue(core_api.in_mpi)
            self.assertTrue(jittor.in_mpi)
            self.assertTrue(core_api.compile_extern.in_mpi)
            self.assertTrue(fake_mpi.state)
        finally:
            core_api.mpi = old_mpi
            core_api.compile_extern.in_mpi = old_compile_in_mpi

    def test_core_api_is_the_only_large_python_api_implementation(self):
        path = self.jittor / "_runtime" / "core_api.py"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        definitions = {
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        self.assertIn("Module", definitions)
        self.assertIn("Function", definitions)
        self.assertIn("array", definitions)
        self.assertIn("flag_scope", definitions)
        self.assertNotIn("compose", definitions)

    def test_source_architecture_names_the_core_api_owner(self):
        source = (
            self.repo / "docs" / "architecture" / "source-architecture.md"
        ).read_text(encoding="utf-8")
        normalized = " ".join(source.split())
        self.assertIn("`jittor._runtime.core_api`", normalized)
        self.assertIn("Public root exports retain object identity", normalized)

    def test_preflight_and_lazy_shim_are_stdlib_only(self):
        stdlib = {
            "__future__",
            "collections",
            "dataclasses",
            "glob",
            "hashlib",
            "importlib",
            "os",
            "pathlib",
            "sys",
            "traceback",
            "warnings",
        }
        # `diagnostics` is this layer's own recorder, and it is on this list
        # only because it is itself stdlib-only -- which the loop below checks
        # rather than assumes. Preflight runs before the compiler and the
        # native core exist, and that is what must not change; being unable to
        # record what preflight swallowed would be the wrong way to keep it.
        allowed = stdlib | {"diagnostics"}
        self.assertTrue(
            {node.module.split(".", 1)[0] if isinstance(node, ast.ImportFrom)
             else "" for node in ast.walk(
                 ast.parse((self.compat / "diagnostics.py").read_text(encoding="utf-8")))
             if isinstance(node, (ast.Import, ast.ImportFrom))} <= stdlib | {""},
            "diagnostics.py must stay stdlib-only to be importable this early")
        for relative in ("shim/preflight.py", "shim/__init__.py"):
            path = self.compat / relative
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imports = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".", 1)[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    imports.add((node.module or "").split(".", 1)[0])
            with self.subTest(path=relative):
                self.assertTrue(imports.issubset(allowed), imports - allowed)

    def test_shim_runtime_is_orchestration_only(self):
        runtime = self.compat / "shim" / "runtime.py"
        tree = ast.parse(runtime.read_text(encoding="utf-8"), filename=str(runtime))
        definitions = [
            node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        self.assertEqual(definitions, ["enable"])
        composition = (self.compat / "runtime.py").read_text(encoding="utf-8")
        self.assertNotIn("apply_external_runtime_patches", composition)
        for name in ("preflight.py", "discovery.py", "build.py", "control.py"):
            self.assertTrue((runtime.parent / name).is_file())

    def test_alias_ownership_is_central(self):
        aliases = (self.compat / "_aliases.py").read_text(encoding="utf-8")
        for name in (
            "jittor.attention",
            "jittor.torch_compat",
            "jittor.torch_fsdp2_compat",
            "jittor.torch_shim",
            "jittor.triton_shim",
            "jittor.depthwise_conv",
        ):
            self.assertIn(name, aliases)
        for path in (
            self.compat / "shim" / "__init__.py",
            self.compat / "triton" / "__init__.py",
            self.compat / "fsdp2" / "__init__.py",
            self.jittor / "nn" / "modules" / "depthwise.py",
        ):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=str(path.relative_to(self.repo))):
                self.assertNotIn("MetaPathFinder", source)
                self.assertNotIn("_LegacyAliasFinder", source)

    def test_new_modules_parse_as_python_37(self):
        paths = [
            self.jittor / "_runtime" / "__init__.py",
            self.jittor / "_runtime" / "core_api.py",
            self.compat / "_aliases.py",
            self.compat / "integrations.py",
            self.compat / "runtime.py",
            self.compat / "torch" / "context.py",
        ]
        paths.extend((self.compat / "torch" / "installers").glob("*.py"))
        paths.extend((self.compat / "shim").glob("*.py"))
        for path in paths:
            with self.subTest(path=str(path.relative_to(self.repo))):
                ast.parse(
                    path.read_text(encoding="utf-8"),
                    filename=str(path),
                    feature_version=(3, 7),
                )


if __name__ == "__main__":
    unittest.main()
