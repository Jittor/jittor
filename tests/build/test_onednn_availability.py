"""Whether oneDNN can be obtained is decided before a CPU product needs it.

``use_mkl`` is on by default and the default install builds oneDNN from source
on first use. Without cmake that build cannot start, and every first CPU matmul
used to fail with "building oneDNN v3 requires cmake". The library is now
reported unavailable up front, and the generic kernels are used instead.
"""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from jittor.build import onednn


ASSET = SimpleNamespace(sha256="0" * 64, url="unused", filename="unused.tar.gz")
VERSION = "3.0"


class TestSourceBuildAvailable(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.compiler = __import__("sys").executable

    def test_no_cmake_and_no_build_is_unavailable(self):
        with mock.patch.object(onednn.shutil, "which", return_value=None):
            self.assertFalse(onednn.source_build_available(
                self.root, ASSET, VERSION, self.compiler))

    def test_cmake_makes_it_available(self):
        with mock.patch.object(onednn.shutil, "which", return_value="/usr/bin/cmake"):
            self.assertTrue(onednn.source_build_available(
                self.root, ASSET, VERSION, self.compiler))

    def test_an_installed_matching_build_needs_no_cmake(self):
        prefix, metadata, _, _, _ = onednn._source_build(
            self.root, ASSET, VERSION, self.compiler)
        (prefix / "lib").mkdir(parents=True)
        (prefix / "lib" / "libdnnl.so").write_bytes(b"")
        (prefix / "jittor-build.json").write_text(json.dumps(metadata, sort_keys=True))
        with mock.patch.object(onednn.shutil, "which", return_value=None):
            self.assertTrue(onednn.source_build_available(
                self.root, ASSET, VERSION, self.compiler))
        # A build for different options/compiler is not a substitute.
        (prefix / "jittor-build.json").write_text(json.dumps({"other": 1}))
        with mock.patch.object(onednn.shutil, "which", return_value=None):
            self.assertFalse(onednn.source_build_available(
                self.root, ASSET, VERSION, self.compiler))

    def test_install_source_still_raises_without_cmake(self):
        # The policy above keeps the loader from being called; called anyway,
        # the build keeps its explicit error rather than degrading silently.
        with mock.patch.object(onednn.shutil, "which",
                               side_effect=lambda name: None if name == "cmake" else name):
            with self.assertRaisesRegex(RuntimeError, "requires cmake"):
                onednn.install_source(self.root, ASSET, VERSION, self.compiler,
                                      download=None, safe_extract=None)


if __name__ == "__main__":
    unittest.main()
