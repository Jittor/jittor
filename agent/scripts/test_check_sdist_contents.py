"""Standard-library tests for the source-distribution contents gate."""

from __future__ import print_function

import contextlib
import importlib.util
import io
from pathlib import Path
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

    def test_missing_tools_build_source_fails(self):
        members = dict(self.members)
        del members["tools/build/build_aarch64_mkl.sh"]
        status, _stdout, stderr = self._run(self._sdist("missing.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("missing required source-distribution member", stderr)
        self.assertIn("tools/build/build_aarch64_mkl.sh", stderr)

    def test_missing_non_sentinel_source_file_fails(self):
        self.members["tools/build/README.md"] = b"build documentation\n"
        archive_members = dict(self.members)
        del archive_members["tools/build/README.md"]
        status, _stdout, stderr = self._run(self._sdist("missing-readme.tar.gz", archive_members))
        self.assertEqual(status, 1)
        self.assertIn("missing required source-distribution member", stderr)
        self.assertIn("tools/build/README.md", stderr)

    def test_generated_cache_payload_fails(self):
        members = dict(self.members)
        members["examples/notebooks/.ipynb_checkpoints/basics.ipynb"] = b"cache\n"
        status, _stdout, stderr = self._run(self._sdist("polluted.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("polluting sdist member", stderr)
        self.assertIn(".ipynb_checkpoints", stderr)

    def test_unexpected_ignored_tool_file_fails(self):
        members = dict(self.members)
        members["tools/.venv/bin/python"] = b"generated environment\n"
        status, _stdout, stderr = self._run(self._sdist("unexpected.tar.gz", members))
        self.assertEqual(status, 1)
        self.assertIn("unexpected source-distribution member", stderr)
        self.assertIn("tools/.venv/bin/python", stderr)


if __name__ == "__main__":
    unittest.main()
