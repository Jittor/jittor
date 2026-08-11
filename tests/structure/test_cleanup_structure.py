"""Contracts for the Stage 6 tools, examples, and runtime-wheel cleanup."""

from __future__ import print_function

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class TestCleanupStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        if not (cls.repo_root / "pyproject.toml").is_file():
            raise unittest.SkipTest("cleanup contracts require a source checkout")

    def test_retired_runtime_payloads_are_absent(self):
        retired = (
            "python/jittor/script",
            "python/jittor/demo",
            "python/jittor/notebook",
            "python/jittor/vcompiler",
            "python/jittor/version",
            "python/jittor/utils/polish.py",
            "python/jittor/utils/polish_centos.py",
            "python/jittor_utils/pack_offline.py",
        )
        for relative in retired:
            with self.subTest(path=relative):
                self.assertFalse((self.repo_root / relative).exists())

    def test_moved_tools_and_example_targets_exist(self):
        required = (
            "examples/gan/simple_cgan.py",
            "tools/benchmarks/legacy/inference_perf.py",
            "tools/build/build_aarch64_mkl.sh",
            "tools/distributed/tmpi",
            "tools/docs/legacy/make_doc.py",
            "tools/install/legacy/install.sh",
            "tools/install/legacy/install_llvm.sh",
            "tools/install/legacy/install_mkl.sh",
            "tools/release/legacy/polish.py",
            "tools/release/legacy/polish_centos.py",
            "tools/release/pack_offline.py",
            "tools/services/legacy/converter_server.sh",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((self.repo_root / relative).is_file())

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

    def test_extern_runtime_contract_and_llvm_defer_are_intact(self):
        required = (
            "python/jittor/extern/__init__.py",
            "python/jittor/extern/acl/aclops",
            "python/jittor/extern/acl/aclnn",
            "python/jittor/extern/acl/hccl",
            "python/jittor/extern/corex/corex_compiler.py",
            "python/jittor/extern/cuda/inc",
            "python/jittor/extern/cuda/src",
            "python/jittor/extern/cuda/cub",
            "python/jittor/extern/cuda/cublas",
            "python/jittor/extern/cuda/cudnn",
            "python/jittor/extern/cuda/cufft",
            "python/jittor/extern/cuda/curand",
            "python/jittor/extern/cuda/cusparse",
            "python/jittor/extern/cuda/cutt",
            "python/jittor/extern/cuda/nccl",
            "python/jittor/extern/mkl/ops",
            "python/jittor/extern/mpi/inc",
            "python/jittor/extern/mpi/ops",
            "python/jittor/extern/mpi/src",
            "python/jittor/extern/rocm",
            "python/jittor/extern/llvm/jt_alignment_from_assumptions.cc",
        )
        for relative in required:
            with self.subTest(path=relative):
                self.assertTrue((self.repo_root / relative).exists())
        compiler = (self.repo_root / "python" / "jittor" / "compiler.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def compile_extern():", compiler)


if __name__ == "__main__":
    unittest.main()
