"""Host-only checks for fair comparisons and truthful worker failure rows."""

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parents[2] / "benchmarks" / "torch_compat"


def load(name):
    spec = importlib.util.spec_from_file_location("benchmark_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    before = sys.path[:]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = before
    return module


def measurement(runtime):
    result = {
        "workload": "example", "runtime": runtime, "status": "ok",
        "device": "cpu", "observed_device": "cpu", "size": "tiny",
        "dtype": "float32", "config": {"batch": 1}, "seed": 0,
        "requires": ["transformers"], "items_per_step": 1,
        "precision_policy": {"tf32": None},
        "versions": {"numpy": "1.26.4", "transformers": "4.56.2"},
        "runtime_conditions": {
            "affinity": [0, 1], "runtime_threads": 2,
            "thread_env": {"OMP_NUM_THREADS": "2", "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"}},
        "median_s": 2.0 if runtime == "torch" else 1.0,
        "values": [1.0, 2.0], "first_step_s": 3.0,
    }
    if runtime == "jittor":
        result["fallbacks"] = 0
    return result


class TestModelBenchmarkContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global report, runner, worker
        report = load("report")
        with mock.patch.dict(sys.modules, {"report": report}):
            runner = load("run")
        worker = load("worker")

    def test_matching_contract_produces_ratio(self):
        pair = {r: measurement(r) for r in ("torch", "jittor")}
        self.assertEqual(report.comparison_issues(pair), [])
        self.assertEqual(report._ratio(pair), 0.5)

    def test_mismatched_contract_never_produces_ratio_or_agreement(self):
        changes = (
            ("device", "cuda"), ("observed_device", "cuda"),
            ("dtype", "float16"), ("size", "full"), ("seed", 3),
            ("config", {"batch": 2}), ("precision_policy", {"tf32": True}),
            ("versions", {"numpy": "1.26.4", "transformers": "4.57.0"}),
            ("runtime_conditions", {"affinity": [0], "runtime_threads": 1, "thread_env": {}}),
        )
        for field, value in changes:
            with self.subTest(field=field):
                pair = {r: measurement(r) for r in ("torch", "jittor")}
                pair["jittor"][field] = value
                self.assertTrue(report.comparison_issues(pair))
                self.assertIsNone(report._ratio(pair))
                self.assertEqual(report._agree(pair["jittor"], pair["torch"]), "-")

    def test_legacy_results_render_but_do_not_claim_verified_ratio(self):
        results = [measurement(r) for r in ("torch", "jittor")]
        for result in results:
            del result["observed_device"]
        text = report.render_table({"meta": {}, "results": results})
        self.assertIn("missing observed_device", text)
        self.assertIn("0/1 workloads compared", text)
        self.assertNotIn("0.50x", text)

    def test_precision_readback_and_fallback_refuse_accelerator_ratio(self):
        pair = {r: measurement(r) for r in ("torch", "jittor")}
        for result in pair.values():
            result.update(device="cuda", observed_device="cuda", settings={"tf32_matmul": True})
        self.assertEqual(report._ratio(pair), 0.5)
        pair["jittor"]["settings"]["tf32_matmul"] = False
        self.assertIn("different CUDA precision readback", report.comparison_issues(pair))
        self.assertIsNone(report._ratio(pair))
        pair["jittor"]["settings"]["tf32_matmul"] = True
        pair["jittor"]["fallbacks"] = 1
        self.assertIsNone(report._ratio(pair))

    def test_missing_or_nonfinite_timing_never_enters_ratio(self):
        for value in (None, 0, float("nan"), float("inf")):
            with self.subTest(value=value):
                pair = {r: measurement(r) for r in ("torch", "jittor")}
                pair["torch"]["median_s"] = value
                self.assertIsNone(report._ratio(pair))

    def test_missing_or_invalid_jittor_fallback_counter_refuses_ratio(self):
        pair = {r: measurement(r) for r in ("torch", "jittor")}
        del pair["jittor"]["fallbacks"]
        self.assertIn("missing Jittor fallback count", report.comparison_issues(pair))
        self.assertIsNone(report._ratio(pair))
        for value in (-1, None, False, "0", 0.0):
            with self.subTest(value=value):
                pair["jittor"]["fallbacks"] = value
                self.assertIn("invalid Jittor fallback count", report.comparison_issues(pair))
                self.assertIsNone(report._ratio(pair))

    def test_saved_run_speedup_requires_same_device_and_contract(self):
        old = {"meta": {"device": "cpu"}, "results": [measurement("jittor")]}
        new = copy.deepcopy(old)
        new["results"][0]["median_s"] = 0.5
        self.assertIn("2.00x", report.compare(old, new))
        new["meta"]["device"] = "cuda"
        self.assertNotIn("2.00x", report.compare(old, new))
        self.assertIn("different run device", report.compare(old, new))

    def test_cpu_environment_sets_backend_and_threads_before_import(self):
        options = SimpleNamespace(device="cpu", gpu=None, threads=2)
        with mock.patch.dict("os.environ", {"JT_USE_CUDA": "1", "use_cuda": "1", "JT_BACKEND": "cuda"}, clear=True):
            jittor = runner.environment_for("jittor", options, Path("state"))
            torch = runner.environment_for("torch", options, Path("state"))
        self.assertEqual(jittor["use_cuda"], "0")
        self.assertEqual(jittor["JT_USE_CUDA"], "0")
        self.assertEqual(jittor["JT_BACKEND"], "cpu")
        self.assertNotIn("use_cuda", torch)
        self.assertNotIn("JT_USE_CUDA", torch)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"):
            self.assertEqual(jittor[name], torch[name])

    def test_device_readback_refuses_cpu_fallback(self):
        tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
        self.assertEqual(worker.output_device([tensor], "cpu"), "cpu")
        with self.assertRaisesRegex(RuntimeError, "requested cuda"):
            worker.output_device([tensor], "cuda")

    def test_listing_resolves_inherited_dependency_and_dtype(self):
        specs = {row["name"]: row for row in runner.list_workloads()}
        self.assertEqual(specs["qwen3_prefill"]["requires"], ("transformers",))
        self.assertEqual(specs["resnet50_infer_b1"]["default_dtype"], "float16")
        self.assertEqual(len(specs), 14)

    def test_worker_cannot_report_success_after_crash_or_timeout(self):
        options = SimpleNamespace(torch_python="reference-python", jittor_python="jittor-python",
                                  device="cpu", size="tiny", seed=0, dtype=None,
                                  warmup=None, repeats=None, batch=None, tf32=True,
                                  cudnn_benchmark=False, allow_fallback=False,
                                  timeout=30, gpu=None, threads=2)
        output = runner.MARKER + json.dumps({"status": "ok", "median_s": 1})
        cases = (
            (SimpleNamespace(stdout=output, returncode=-11), None, "crash"),
            (None, subprocess.TimeoutExpired("worker", 30, output=output.encode()), "timeout"),
            (SimpleNamespace(stdout=runner.MARKER + "{", returncode=1), None, "crash"),
            (SimpleNamespace(stdout=runner.MARKER + "[]", returncode=1), None, "crash"),
        )
        with tempfile.TemporaryDirectory() as directory:
            for completed, error, expected in cases:
                with self.subTest(status=expected), mock.patch.object(
                        runner.subprocess, "run", return_value=completed, side_effect=error):
                    result = runner.run_one("example", "torch", options,
                                            Path(directory), Path(directory), "default")
                    self.assertEqual(result["status"], expected)
                    self.assertEqual(result["compile"], "default")


if __name__ == "__main__":
    unittest.main()
