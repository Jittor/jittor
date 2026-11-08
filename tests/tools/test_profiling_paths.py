"""Profiling output boundaries can be checked without loading model packages."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from benchmarks.profiling import _paths


class TestProfilingPaths(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.source = self.base / "source"
        (self.source / "python/jittor").mkdir(parents=True)
        (self.source / "AGENTS.md").write_text("Project rules\n")
        self.lab = self.base / "lab"
        self.lab.mkdir()

    def test_explicit_and_default_source_roots(self):
        self.assertEqual(_paths.resolve_source_root(self.source), self.source)
        with patch.object(_paths, "REPO_ROOT", self.source):
            self.assertEqual(_paths.resolve_source_root(), self.source)
        with self.assertRaisesRegex(ValueError, "source checkout"):
            _paths.resolve_source_root(self.lab)

    def test_report_below_external_lab_is_accepted_without_creating_it(self):
        report = self.lab / "run/report.json"
        with patch.dict(os.environ, JITTOR_LAB_ROOT=str(self.lab)):
            self.assertEqual(_paths.lab_output_path(report, self.source), report)
        self.assertFalse(report.exists())

    def test_report_requires_an_explicit_lab_directory(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "must name an external"):
                _paths.lab_output_path(self.lab / "report.json", self.source)

    def test_report_cannot_escape_lab_or_replace_its_root(self):
        with patch.dict(os.environ, JITTOR_LAB_ROOT=str(self.lab)):
            for target in (self.lab, self.lab / "../report.json"):
                with self.subTest(target=target), self.assertRaises(ValueError):
                    _paths.lab_output_path(target, self.source)

    def test_lab_inside_checkout_does_not_allow_checkout_output(self):
        with patch.dict(os.environ, JITTOR_LAB_ROOT=str(self.source)):
            with self.assertRaisesRegex(ValueError, "outside the source checkout"):
                _paths.lab_output_path(self.source / "report.json", self.source)

    def test_symlink_cannot_redirect_report_outside_lab(self):
        (self.lab / "redirect").symlink_to(self.source, target_is_directory=True)
        with patch.dict(os.environ, JITTOR_LAB_ROOT=str(self.lab)):
            with self.assertRaisesRegex(ValueError, "below JITTOR_LAB_ROOT"):
                _paths.lab_output_path(self.lab / "redirect/report.json", self.source)


if __name__ == "__main__":
    unittest.main()
