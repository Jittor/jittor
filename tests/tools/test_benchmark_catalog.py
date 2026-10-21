"""Host-only benchmark discovery: no tensor runtime or optional model packages."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from _helpers.child_process import run_python_child
from benchmarks import catalog


REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "benchmarks/catalog.py"


class BenchmarkCatalogTests(unittest.TestCase):
    def test_full_listing_without_site_packages_or_runtime_state(self):
        with tempfile.TemporaryDirectory() as directory:
            env = dict(os.environ, HOME=directory, TMPDIR=directory,
                       JITTOR_HOME=directory, JITTOR_LAB_ROOT=directory)
            env.pop("PYTHONPATH", None)
            result = run_python_child(["-S", str(SCRIPT), "--json"],
                                      cwd=directory, env=env, inherit=False,
                                      repo_paths=False, check=True)
            listing = json.loads(result.stdout)
            self.assertEqual(listing["schema_version"], 1)
            self.assertEqual({suite["id"] for suite in listing["suites"]},
                             {suite["id"] for suite in catalog.SUITES})
            self.assertTrue(all(suite["workloads"] for suite in listing["suites"]))
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_asv_inventory_reads_source_without_executing_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benchmarks/asv/future_suite.py"
            source.parent.mkdir(parents=True)
            source.write_text('''
import missing_tensor_runtime
raise RuntimeError("listing must not execute this")
DEVICES = ("cpu", "future-device")
CASES = {"new-shape": (1, 2), "future-shape": (3, 4)}
class FutureBenchmarks:
    params = (["jittor"], DEVICES, list(CASES))
    param_names = ["backend", "device", "case"]
    def time_new_workload(self, backend, device):
        pass
    def setup(self, backend, device):
        pass
''', encoding="utf-8")
            self.assertEqual(catalog.asv_workloads(root), [{
                "name": "future_suite.FutureBenchmarks.time_new_workload",
                "source": "benchmarks/asv/future_suite.py",
                "parameters": {"backend": ["jittor"], "device": ["cpu", "future-device"],
                               "case": ["new-shape", "future-shape"]},
            }])

    def test_dynamic_axes_are_reported_as_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benchmarks/asv/dynamic.py"
            source.parent.mkdir(parents=True)
            source.write_text('''
class DynamicBenchmarks:
    params = (["jittor"], get_devices())
    param_names = ["backend", "device"]
    def time_workload(self, backend, device):
        pass
''', encoding="utf-8")
            with self.assertRaises(ValueError):
                catalog.asv_workloads(root)

    def test_parameter_shape_mismatch_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "benchmarks/asv/broken.py"
            source.parent.mkdir(parents=True)
            source.write_text('''
class BrokenBenchmarks:
    params = (["jittor"], ["cpu"])
    param_names = ["backend"]
    def time_workload(self, backend, device):
        pass
''', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "parameter names/axes mismatch"):
                catalog.asv_workloads(root)

    def test_stale_target_path_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Stale benchmark catalog path"):
                catalog.catalog(Path(directory), selected="asv")

    def test_text_filter_only_lists_selected_suite(self):
        result = run_python_child(["-S", str(SCRIPT), "--suite", "asv"],
                                  repo_paths=False, check=True)
        self.assertIn("operators.OperatorBenchmarks.time_operator", result.stdout)
        self.assertNotIn("torch-compat [", result.stdout)
        self.assertIn("not hardware validation", result.stdout)


if __name__ == "__main__":
    unittest.main()
