"""Boundary rules left over from the Stage 6 tools, examples and wheel cleanup.

Two kinds of assertion used to live here, and only one of them earned its place.

**Rules** stay: no module/package path collisions, no shadowed definitions, no
unreviewed cross-file duplicates, imports pointing only at canonical runtime
modules, development trees that are not import packages, shell tools that parse,
tool imports without side effects. These say something true about the design and
survive a file moving.

**Manifests** are gone: the exact entry set of ``python/jittor``, the exact set of
root-level modules, the list of documentation files that must exist, the list of
moved tool and example paths, the list of ``extern`` subdirectories. Every one of
them was a copy of the directory tree written a second time, so relocating a
single file meant editing this test -- and none of them would have caught a wrong
answer from any of that code.

**Migration guards** -- assertions that something already deleted stays deleted --
are true forever the moment the migration lands, and then cost the gate time for
nothing. They now carry an expiry date; see
``test_migration_guards_have_not_outlived_their_purpose``.
"""

from __future__ import print_function

import ast
import copy
import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Dict, List, Tuple
import unittest


#: After this date the migration guards in this file must be deleted rather than
#: carried forever. Moving the date is a decision someone makes on purpose, with
#: a reason; leaving them in place silently is what the audit found.
MIGRATION_GUARD_EXPIRY = datetime.date(2027, 3, 1)


class TestCleanupStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        if not (cls.repo_root / "pyproject.toml").is_file():
            raise unittest.SkipTest("cleanup contracts require a source checkout")

    def test_migration_guards_have_not_outlived_their_purpose(self):
        guards = (
            "test_retired_runtime_payloads_are_absent",
            "test_legacy_fsdp2_path_names_are_absent_everywhere",
            "test_vcompiler_retirement_is_documented",
        )
        self.assertLess(
            datetime.date.today(),
            MIGRATION_GUARD_EXPIRY,
            "these one-shot migration guards have been true since the migration "
            "landed and are now pure gate weight: %s. Delete them together with "
            "this test, or move MIGRATION_GUARD_EXPIRY with a written reason."
            % ", ".join(guards),
        )

    def test_retired_runtime_payloads_are_absent(self):
        retired = (
            "python/jittor/attention.py",
            "python/jittor/contrib.py",
            "python/jittor/gradfunctional",
            "python/jittor/lr_scheduler.py",
            "python/jittor/nn/sparse.py",
            "python/jittor/other",
            "python/jittor/script",
            "python/jittor/demo",
            "python/jittor/notebook",
            "python/jittor/optim.py",
            "python/jittor/weightnorm.py",
            "python/jittor/sparse.py",
            "python/jittor/vcompiler",
            "python/jittor/version",
            "python/jittor/utils/polish.py",
            "python/jittor/utils/polish_centos.py",
            "python/jittor/extern/llvm",
            "python/jittor_utils/translator.py",
            "python/jittor_utils/pack_offline.py",
            "tools/docs/legacy/make_doc.py",
        )
        for relative in retired:
            with self.subTest(path=relative):
                self.assertFalse((self.repo_root / relative).exists())

    def test_runtime_tree_has_no_module_package_path_collisions(self):
        runtime_root = self.repo_root / "python" / "jittor"
        collisions = []
        for module_path in runtime_root.rglob("*.py"):
            if module_path.name == "__init__.py":
                continue
            package_path = module_path.with_suffix("")
            if package_path.is_dir():
                collisions.append(
                    (
                        module_path.relative_to(self.repo_root).as_posix(),
                        package_path.relative_to(self.repo_root).as_posix() + "/",
                    )
                )
        self.assertEqual(collisions, [])

    def test_runtime_files_have_no_shadowed_top_level_definitions(self):
        runtime_root = self.repo_root / "python" / "jittor"
        duplicates = []
        for path in sorted(runtime_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            definitions: Dict[str, List[int]] = {}
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                is_overload = any(
                    (isinstance(decorator, ast.Name) and decorator.id == "overload")
                    or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
                    for decorator in node.decorator_list
                )
                if is_overload:
                    continue
                definitions.setdefault(node.name, []).append(node.lineno)
            for name, lines in definitions.items():
                if len(lines) > 1:
                    duplicates.append(
                        (
                            path.relative_to(self.repo_root).as_posix(),
                            name,
                            lines,
                        )
                    )
        self.assertEqual(duplicates, [])

    def test_cross_file_duplicate_implementations_are_reviewed(self):
        source_root = self.repo_root / "python"
        implementations: Dict[str, List[Tuple[str, str]]] = {}
        for path in sorted(source_root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if any(
                    (isinstance(decorator, ast.Name) and decorator.id == "overload")
                    or (isinstance(decorator, ast.Attribute) and decorator.attr == "overload")
                    for decorator in node.decorator_list
                ):
                    continue
                if getattr(node, "end_lineno", node.lineno) - node.lineno < 3:
                    continue
                normalized = copy.deepcopy(node)
                normalized.name = "_"
                normalized.decorator_list = []
                fingerprint = ast.dump(normalized, include_attributes=False)
                implementations.setdefault(fingerprint, []).append(
                    (
                        path.relative_to(self.repo_root).as_posix(),
                        node.name,
                    )
                )

        duplicate_groups = [
            frozenset(group)
            for group in implementations.values()
            if len({path for path, _name in group}) > 1
        ]
        deploy_helpers = frozenset(
            {
                ("python/jittor/compat/shim/deploy.py", "_default_site_packages"),
                ("python/jittor/compat/triton/deploy.py", "_default_site_packages"),
            }
        )
        stub_fallbacks = frozenset(
            {
                (
                    "python/jittor/compat/shim/resources/stubs/torchaudio/__init__.py",
                    "_AnyModule",
                ),
                (
                    "python/jittor/compat/shim/resources/stubs/torchdata/__init__.py",
                    "_AnyModule",
                ),
            }
        )
        tuple_helpers = frozenset(
            {
                ("python/jittor/extern/acl/acl_compiler.py", "_ntuple"),
                ("python/jittor/extern/acl/aclops/conv_op.py", "_ntuple"),
                ("python/jittor/misc/tensor_ops.py", "_ntuple"),
            }
        )
        model_local_groups = {
            frozenset(
                {
                    ("python/jittor/models/convnext.py", "StochasticDepth"),
                    ("python/jittor/models/maxvit.py", "StochasticDepth"),
                }
            ),
            frozenset(
                {
                    ("python/jittor/models/efficientnet.py", "_make_divisible"),
                    ("python/jittor/models/mobilenet_v3.py", "_make_divisible"),
                    ("python/jittor/models/regnet.py", "_make_divisible"),
                }
            ),
            frozenset(
                {
                    ("python/jittor/models/googlenet.py", "BasicConv2d"),
                    ("python/jittor/models/inception.py", "BasicConv2d"),
                }
            ),
        }
        acl_shared_helpers = {
            frozenset(
                {
                    ("python/jittor/extern/acl/aclops/getitem_op.py", name),
                    ("python/jittor/extern/acl/aclops/setitem_op.py", name),
                }
            )
            for name in ("caculate_shape", "can_broadcast_and_shape")
        }
        acl_forward_helpers = frozenset(
            {
                ("python/jittor/extern/acl/aclops/index_op.py", "range_forward"),
                ("python/jittor/extern/acl/aclops/setitem_op.py", "setitem_forward"),
            }
        )
        legacy_loader_names = {
            "_maybe_decode_ascii",
            "persistent_load",
            "_storage_type_to_dtype_map",
            "_get_dtype_from_pickle_storage_type",
            "StorageType",
            "jittor_rebuild_var",
            "ArrayWrapper",
            "jittor_rebuild_direct",
            "_check_seekable",
            "_is_compressed_file",
            "_should_read_directly",
            "persistent_load_direct",
        }

        def reviewed(group):
            if group in (
                deploy_helpers,
                stub_fallbacks,
                tuple_helpers,
                acl_forward_helpers,
            ):
                return True
            if group in model_local_groups or group in acl_shared_helpers:
                return True
            paths = {path for path, _name in group}
            names = {name for _path, name in group}
            if (
                paths
                == {
                    "python/jittor_utils/load_pytorch.py",
                    "python/jittor_utils/load_pytorch_old.py",
                }
                and names <= legacy_loader_names
            ):
                return True
            if all(path.startswith("python/jittor/extern/acl/aclops/") for path in paths):
                return len(group) > 10 and all(name.endswith("_cmd") for _path, name in group)
            return False

        unreviewed = sorted(
            (sorted(group) for group in duplicate_groups if not reviewed(group)),
            key=repr,
        )
        self.assertEqual(unreviewed, [])

    def test_legacy_fsdp2_path_names_are_absent_everywhere(self):
        forbidden_names = {
            "jittor_fsdp2",
            "torch_fsdp2_compat.py",
            "torch_fsdp2_compat",
        }
        found = []
        for path in self.repo_root.rglob("*"):
            if ".git" in path.parts:
                continue
            if path.name in forbidden_names:
                found.append(path.relative_to(self.repo_root).as_posix())
        self.assertEqual(found, [])

    def test_active_python_imports_use_canonical_runtime_modules(self):
        retired_modules = (
            "jittor._misc",
            "jittor._nn",
            "jittor._pool",
            "jittor._torch_compat",
            "jittor._torch_fsdp2",
            "jittor.depthwise_conv",
            "jittor.monkeypatch_ops",
            "jittor.torch_compat",
            "jittor.torch_fsdp2_compat",
            "jittor.torch_shim",
            "jittor.triton_shim",
            "jittor_fsdp2",
        )
        intentional_compatibility_imports = [
            (
                "tests/compat/torch/test_torch_shim_aliases.py",
                "jittor.torch_shim.flashattn_jittor",
            ),
            (
                "tests/compat/torch/test_torch_shim_aliases.py",
                "jittor.triton_shim",
            ),
            (
                "tests/structure/test_nn_structure.py",
                "jittor.depthwise_conv",
            ),
            (
                "tests/structure/test_torch_compat_structure.py",
                "jittor.torch_compat",
            ),
            (
                "tests/structure/test_torch_compat_structure.py",
                "jittor.torch_compat",
            ),
            (
                "tests/structure/test_torch_fsdp2_structure.py",
                "jittor.torch_fsdp2_compat",
            ),
            (
                "tests/structure/test_torch_fsdp2_structure.py",
                "jittor.torch_fsdp2_compat.api",
            ),
            (
                "tests/structure/test_triton_structure.py",
                "jittor.triton_shim",
            ),
            (
                "tests/structure/test_triton_structure.py",
                "jittor.triton_shim.*",
            ),
        ]
        roots = (
            "agent/scripts",
            "agent/skills",
            "docs",
            "examples",
            "python",
            "tests",
            "tools",
        )
        paths = [self.repo_root / "noxfile.py", self.repo_root / "setup.py"]
        for root in roots:
            paths.extend((self.repo_root / root).rglob("*.py"))

        found = []
        for path in sorted(paths):
            relative = path.relative_to(self.repo_root).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
            for node in ast.walk(tree):
                imported: List[str] = []
                if isinstance(node, ast.Import):
                    imported.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.append(node.module)
                    imported.extend(node.module + "." + alias.name for alias in node.names)
                elif isinstance(node, ast.Call) and node.args:
                    is_import_module = (
                        isinstance(node.func, ast.Attribute) and node.func.attr == "import_module"
                    ) or (isinstance(node.func, ast.Name) and node.func.id == "import_module")
                    is_builtin_import = (
                        isinstance(node.func, ast.Name) and node.func.id == "__import__"
                    )
                    argument = node.args[0]
                    if (is_import_module or is_builtin_import) and isinstance(argument, ast.Str):
                        imported.append(argument.s)
                    elif (
                        (is_import_module or is_builtin_import)
                        and isinstance(argument, ast.BinOp)
                        and isinstance(argument.op, ast.Add)
                        and isinstance(argument.left, ast.Str)
                    ):
                        imported.append(argument.left.s + "*")
                for module in imported:
                    if any(
                        module == retired or module.startswith(retired + ".")
                        for retired in retired_modules
                    ):
                        found.append((relative, module))

        # A subset, not an exact match: removing one of these compatibility
        # imports is progress and must not turn the gate red. Adding a new import
        # of a retired module still fails, which is the rule being enforced.
        unexpected = sorted(set(found) - set(intentional_compatibility_imports))
        self.assertEqual(unexpected, [])

    def test_documentation_governance_checker(self):
        checker = self.repo_root / "agent" / "scripts" / "check_docs_governance.py"
        result = subprocess.run(
            [sys.executable, str(checker)],
            cwd=str(self.repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_development_trees_are_not_import_packages(self):
        initializers = []
        for relative in ("examples", "tools"):
            initializers.extend((self.repo_root / relative).rglob("__init__.py"))
        self.assertEqual(initializers, [])

    def test_shell_tools_are_executable_and_parse(self):
        scripts = sorted((self.repo_root / "tools").rglob("*.sh"))
        scripts.append(self.repo_root / "tools" / "distributed" / "tmpi")
        self.assertTrue(scripts)
        for path in scripts:
            with self.subTest(path=path.relative_to(self.repo_root).as_posix()):
                self.assertTrue(os.access(str(path), os.X_OK))
                result = subprocess.run(
                    ["bash", "-n", str(path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout)

    def test_example_and_release_tool_imports_are_side_effect_free(self):
        probe = r"""
import importlib.util
from pathlib import Path
import sys

path = Path(sys.argv[1]).resolve()
work = Path.cwd()
before = sorted(item.relative_to(work).as_posix() for item in work.rglob('*'))
spec = importlib.util.spec_from_file_location('stage6_import_probe', str(path))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
after = sorted(item.relative_to(work).as_posix() for item in work.rglob('*'))
assert before == after, (before, after)
assert callable(module.main)
for forbidden in ('jittor', 'PIL', 'pywebio'):
    assert forbidden not in sys.modules, forbidden
"""
        targets = (
            self.repo_root / "examples" / "gan" / "simple_cgan.py",
            self.repo_root / "tools" / "release" / "pack_offline.py",
        )
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            work = Path(temporary) / "work"
            home.mkdir()
            work.mkdir()
            env = os.environ.copy()
            env.update(
                {
                    "HOME": str(home),
                    "JITTOR_HOME": str(Path(temporary) / "jittor-home"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                }
            )
            for target in targets:
                with self.subTest(path=target.relative_to(self.repo_root).as_posix()):
                    result = subprocess.run(
                        [sys.executable, "-c", probe, str(target)],
                        cwd=str(work),
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        universal_newlines=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stdout)

    def test_pack_offline_dry_run_writes_nothing(self):
        script = self.repo_root / "tools" / "release" / "pack_offline.py"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            env = os.environ.copy()
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, str(script), "--dry-run", "--output-dir", str(output)],
                cwd=temporary,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertFalse(output.exists(), result.stdout)

    def test_vcompiler_retirement_is_documented(self):
        release_note = self.repo_root / "docs" / "releases" / "2.0.md"
        source = release_note.read_text(encoding="utf-8")
        self.assertIn("jittor.vcompiler", source)
        self.assertIn("breaking", source.lower())
        self.assertIn("compile_custom_op", source)

if __name__ == "__main__":
    unittest.main()
