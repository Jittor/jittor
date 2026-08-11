"""Standard-library tests for the source-distribution contents gate."""

from __future__ import print_function

import contextlib
import importlib.util
import io
from pathlib import Path
import shutil
import tarfile
import tempfile
import unittest
from unittest import mock


_SCRIPT = Path(__file__).resolve().with_name("check_sdist_contents.py")
_SPEC = importlib.util.spec_from_file_location("_check_sdist_contents_test", _SCRIPT)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)


class TestSourceDistributionContents(unittest.TestCase):
    def setUp(self):
        temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_directory.cleanup)
        self.root = Path(temporary_directory.name)
        self.members = {path: b"content\n" for path in checker.REQUIRED_SOURCE_PATHS}

    def _sdist(self, name, members):
        source = self.root / "source"
        shutil.rmtree(str(source), ignore_errors=True)
        for relative, content in members.items():
            path = source / "jittor-1.0" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        archive_path = self.root / name
        with tarfile.open(str(archive_path), "w:gz") as archive:
            archive.add(str(source / "jittor-1.0"), arcname="jittor-1.0")
        return archive_path

    def _run(self, path):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with mock.patch.object(
            checker, "_expected_source_paths", return_value=self.members
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = checker.main([str(path)])
        return status, stdout.getvalue(), stderr.getvalue()

    def test_complete_source_distribution_passes(self):
        status, stdout, stderr = self._run(self._sdist("complete.tar.gz", self.members))
        self.assertEqual(status, 0, stderr)
        self.assertIn("source distribution OK", stdout)

    def test_checkout_inventory_includes_python_and_excludes_caches(self):
        tracked = set(checker.REQUIRED_SOURCE_PATHS)
        tracked.update(
            (
                "python/jittor/runtime_source.py",
                "python/jittor/__pycache__/runtime_source.cpython-311.pyc",
                "python/jittor.egg-info/PKG-INFO",
            )
        )
        result = mock.Mock(
            returncode=0,
            stdout=("\0".join(sorted(tracked)) + "\0").encode("utf-8"),
            stderr=b"",
        )
        with mock.patch.object(checker.subprocess, "run", return_value=result) as run:
            paths = checker._expected_source_paths(self.root)

        command = run.call_args.args[0]
        self.assertIn("python", command)
        self.assertIn("python/jittor/runtime_source.py", paths)
        self.assertNotIn("python/jittor/__pycache__/runtime_source.cpython-311.pyc", paths)
        self.assertNotIn("python/jittor.egg-info/PKG-INFO", paths)

    def test_canonical_generated_egg_info_members_pass(self):
        members = dict(self.members)
        members.update(
            {path: b"generated metadata\n" for path in checker.CANONICAL_EGG_INFO_MEMBERS}
        )
        status, stdout, stderr = self._run(self._sdist("canonical-egg-info.tar.gz", members))
        self.assertEqual(status, 0, stderr)
        self.assertIn("source distribution OK", stdout)

    def test_unexpected_canonical_egg_info_member_fails(self):
        members = dict(self.members)
        members["python/jittor.egg-info/not-zip-safe"] = b"unexpected\n"
        status, _stdout, stderr = self._run(self._sdist("extra-egg-info.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("unapproved generated .egg-info metadata", stderr)
        self.assertIn("python/jittor.egg-info/not-zip-safe", stderr)

    def test_other_egg_info_directory_fails(self):
        members = dict(self.members)
        members["python/other.egg-info/PKG-INFO"] = b"unexpected\n"
        status, _stdout, stderr = self._run(self._sdist("other-egg-info.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("unapproved generated .egg-info metadata", stderr)
        self.assertIn("python/other.egg-info/PKG-INFO", stderr)

    def test_missing_tools_build_source_fails(self):
        members = dict(self.members)
        del members["tools/build/build_aarch64_mkl.sh"]
        status, _stdout, stderr = self._run(self._sdist("missing.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("missing required source-distribution member", stderr)
        self.assertIn("tools/build/build_aarch64_mkl.sh", stderr)

    def test_missing_documentation_source_fails(self):
        members = dict(self.members)
        del members["docs/index.md"]
        status, _stdout, stderr = self._run(self._sdist("missing-docs.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("missing required source-distribution member", stderr)
        self.assertIn("docs/index.md", stderr)

    def test_missing_non_sentinel_source_file_fails(self):
        self.members["tools/build/README.md"] = b"build documentation\n"
        archive_members = dict(self.members)
        del archive_members["tools/build/README.md"]
        status, _stdout, stderr = self._run(self._sdist("missing-readme.tar.gz", archive_members))
        self.assertEqual(status, 1)
        self.assertIn("missing required source-distribution member", stderr)
        self.assertIn("tools/build/README.md", stderr)

    def test_missing_runtime_python_source_fails(self):
        self.members["python/jittor/runtime_source.py"] = b"RUNTIME = True\n"
        archive_members = dict(self.members)
        del archive_members["python/jittor/runtime_source.py"]
        status, _stdout, stderr = self._run(self._sdist("missing-python.tar.gz", archive_members))
        self.assertEqual(status, 1)
        self.assertIn("missing required source-distribution member", stderr)
        self.assertIn("python/jittor/runtime_source.py", stderr)

    def test_generated_cache_payload_fails(self):
        members = dict(self.members)
        members["examples/notebooks/.ipynb_checkpoints/basics.ipynb"] = b"cache\n"
        status, _stdout, stderr = self._run(self._sdist("polluted.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("polluting sdist member", stderr)
        self.assertIn(".ipynb_checkpoints", stderr)

    def test_retired_source_paths_are_rejected(self):
        retired = (
            "doc/legacy.md",
            "jittor_fsdp2/__init__.py",
            "README.cn",
            "README.cn.md",
            "python/jittor/_nn",
            "python/jittor/_misc/legacy.py",
            "python/jittor/_nn/legacy.py",
            "python/jittor/_pool/legacy.py",
            "python/jittor/_torch_compat/legacy.py",
            "python/jittor/_torch_fsdp2/legacy.py",
            "python/jittor/torch_fsdp2_compat/__init__.py",
            "python/jittor/torch_shim/__init__.py",
            "python/jittor/triton_shim/__init__.py",
            "python/jittor/depthwise_conv.py",
            "python/jittor/extern/llvm/jt_alignment_from_assumptions.cc",
            "python/jittor/misc.py",
            "python/jittor/monkeypatch_ops.py",
            "python/jittor/nn.py",
            "python/jittor/optim.py",
            "python/jittor/pool.py",
            "python/jittor/torch_compat.py",
            "python/jittor/torch_fsdp2_compat.py",
            "python/jittor_utils/translator.py",
        )
        for index, relative in enumerate(retired):
            with self.subTest(path=relative):
                members = dict(self.members)
                members[relative] = b"retired\n"
                status, _stdout, stderr = self._run(
                    self._sdist("retired-{}.tar.gz".format(index), members)
                )
                self.assertEqual(status, 1)
                self.assertIn("polluting sdist member", stderr)
                self.assertIn(relative, stderr)

    def test_notebook_source_and_products_are_rejected(self):
        for index, relative in enumerate(("docs/tutorial.ipynb", "docs/tutorial.src.md")):
            with self.subTest(path=relative):
                members = dict(self.members)
                members[relative] = b"generated\n"
                status, _stdout, stderr = self._run(
                    self._sdist("notebook-{}.tar.gz".format(index), members)
                )
                self.assertEqual(status, 1)
                self.assertIn("forbidden notebook source/product suffix", stderr)

    def test_unexpected_ignored_tool_file_fails(self):
        members = dict(self.members)
        members["tools/.venv/bin/python"] = b"generated environment\n"
        status, _stdout, stderr = self._run(self._sdist("unexpected.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("unexpected source-distribution member", stderr)
        self.assertIn("tools/.venv/bin/python", stderr)


if __name__ == "__main__":
    unittest.main()
