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
        global report, runner, worker, cuda_runner
        report = load("report")
        with mock.patch.dict(sys.modules, {"report": report}):
            runner = load("run")
        worker = load("worker")
        with mock.patch.dict(sys.modules, {"run": runner, "worker": worker}), \
                mock.patch.object(sys, "path", [str(HERE), *sys.path]):
            cuda_runner = load("benchmark_cuda_models")

    def test_cuda_inference_measures_compiled_before_eager(self):
        argv = ["bench", "--torch-python", "reference-python", "--case", "resnet50_infer_b1",
                "--case", "mobilenetv3-small", "--case", "qwen3_decode"]
        with mock.patch.object(sys, "argv", argv):
            args = cuda_runner.parse_args()
        both = [("torch", "compiled"), ("jittor", "compiled"),
                ("torch", "eager"), ("jittor", "eager")]
        self.assertEqual(cuda_runner.execution_order(args, "resnet50_infer_b1"), both)
        self.assertEqual(cuda_runner.execution_order(args, "mobilenetv3-small"), both)
        self.assertEqual(cuda_runner.execution_order(args, "qwen3_decode"),
                         [("torch", "compiled"), ("torch", "eager"), ("jittor", "eager")])

    def test_cuda_training_profiles_cover_both_gpu_counts(self):
        cases = cuda_runner.TRAINING_CASES
        for family, model, _, batch in cases.values():
            for ranks in (1, 2):
                case = f"{model}-r{ranks}-b{batch}"
                with self.subTest(case=case):
                    self.assertEqual(cases.get(case), (family, model, ranks, batch))

    def test_cuda_stock_inference_excludes_fast_output_mismatch(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as root:
            rows = []
            for runtime, mode, seconds, second_batch in (
                ("torch", "compiled", 0.001, 2.0),
                ("jittor", "compiled", 0.0005, 3.0),
                ("torch", "eager", 0.002, 2.0),
                ("jittor", "eager", 0.0015, 2.0),
            ):
                directory = Path(root) / f"{runtime}-{mode}"
                directory.mkdir()
                np.savez(directory / "outputs.npz",
                         **{"batch.0.0": np.array([1.0], dtype=np.float32),
                            "batch.1.0": np.array([second_batch], dtype=np.float32)})
                row = dict(measurement(runtime), case="resnet50_infer_b1", kind="stock", round=1,
                           median_s=seconds, checked_output_batches=2, directory=str(directory),
                           execution="cuda-graph" if mode == "compiled" else "eager")
                row["torch_mode" if runtime == "torch" else "jittor_mode"] = mode
                rows.append(row)
            result, = cuda_runner.compare_results(rows)
        self.assertEqual(result["torch_baseline_mode"], "compiled")
        self.assertEqual(result["jittor_selected_mode"], "jittor_eager")
        self.assertEqual(result["jittor_compiled_ms"], 0.5)
        self.assertEqual(result["jittor_over_torch"], 1.5)
        self.assertFalse(result["jittor_checks"]["jittor_compiled"]["parity"])
        mismatch = result["jittor_checks"]["jittor_compiled"]["output_errors"]["batch.1.0"]
        self.assertEqual(mismatch["elements_outside_tolerance"], 1)

    def test_cuda_reference_requires_three_verified_modes(self):
        rows = [
            dict(runtime=runtime, case="model", round=1, status="ok",
                 median_s=seconds, execution=mode,
                 **{"torch_mode" if runtime == "torch" else "jittor_mode": mode})
            for runtime, mode, seconds in (
                ("torch", "eager", 0.004),
                ("torch", "compiled", 0.001),
                ("jittor", "eager", 0.002),
            )
        ]
        for include_compiled in (True, False):
            with self.subTest(include_compiled=include_compiled):
                compared = rows if include_compiled else [rows[0], rows[2]]
                result_for_mode = lambda reference, candidate: {
                    "parity": candidate is rows[2], "comparable": candidate is rows[2],
                }
                with mock.patch.object(cuda_runner, "compare_pair", side_effect=result_for_mode):
                    result, = cuda_runner.compare_results(compared)
                self.assertTrue(result["jittor_checks"]["jittor_eager"]["parity"])
                self.assertIsNone(result.get("jittor_over_torch"))
                self.assertIsNone(result.get("torch_baseline_mode"))
                if include_compiled:
                    self.assertFalse(result["torch_compile_check"]["parity"])

    def test_cuda_output_checks_report_absolute_relative_and_cosine_errors(self):
        import numpy as np

        reference = np.array([2.0, 4.0, -np.inf])
        candidate = np.array([1.0, 4.0, -np.inf])
        result = cuda_runner.compare_arrays(np, reference, candidate, 0.0, 0.0)
        self.assertFalse(result["parity"])
        self.assertEqual(result["max_abs_diff"], 1.0)
        self.assertEqual(result["max_rel_diff"], 0.5)
        self.assertAlmostEqual(result["cosine_similarity"], 18 / np.sqrt(340))

    def test_cuda_training_aggregation_tracks_peak_vram_warmup_and_strategy(self):
        job = dict(case="resnet50d-r2-b64", kind="training", mode="compiled-step", ranks=2, batch=64, warmup=2, samples=2)
        records = [
            dict(records=[dict(step=i, total=t) for i, t in enumerate((1.5, 0.5, 0.2, 0.2))],
                 peak_memory_bytes=2 * 1024 ** 3, peak_reserved_bytes=3 * 1024 ** 3, dtype="float32"),
            dict(records=[dict(step=i, total=t) for i, t in enumerate((1.2, 0.6, 0.25, 0.25))],
                 peak_memory_bytes=4 * 1024 ** 3, peak_reserved_bytes=5 * 1024 ** 3, dtype="float32"),
        ]
        agg = cuda_runner.aggregate_training(job, records)
        self.assertEqual(agg["peak_memory_bytes"], 4 * 1024 ** 3)
        self.assertEqual(agg["peak_vram_gb"], 4.0)
        self.assertAlmostEqual(agg["compile_warmup_s"], 2.1)
        self.assertAlmostEqual(agg["first_call_s"], 1.5)
        self.assertEqual(agg["strategy"], "DDP")

    def test_cuda_worker_oom_error_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "worker.log"
            log.write_text("Traceback\ntorch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.00 GiB\n")
            detected = cuda_runner._worker_oom_error([Path(tmp)])
            self.assertIsNotNone(detected)
            self.assertIn("OutOfMemoryError", detected)
            log.write_text("ValueError: bad shape\n")
            self.assertIsNone(cuda_runner._worker_oom_error([Path(tmp)]))

    def test_cuda_training_parity_reports_first_failed_step(self):
        rank = dict(training_input_sha256="inputs", initial_state_sha256="weights",
                    records=[dict(step=0, loss=1.0), dict(step=1, loss=0.8), dict(step=2, loss=0.6)])
        reference = dict(kind="training", rank_results=[rank])
        candidate = copy.deepcopy(reference)
        candidate["rank_results"][0]["records"][1]["loss"] = 0.5
        result = cuda_runner.compare_pair(reference, candidate)
        self.assertTrue(result["same_inputs"])
        self.assertFalse(result["comparable"])
        self.assertEqual(result["loss_errors"], [
            dict(rank=0, first_step=1, steps_outside_tolerance=1, reference_loss=0.8,
                 candidate_loss=0.5, max_abs=0.8 - 0.5),
        ])

    def test_cuda_logit_masks_compare_infinities_at_the_same_positions(self):
        import numpy as np

        reference = np.array([1.0, -np.inf, 2.0], dtype=np.float32)
        same = cuda_runner.compare_arrays(np, reference, reference, 1e-4, 1e-3)
        self.assertTrue(same["parity"])
        self.assertEqual(same["max_abs"], 0.0)
        candidate = np.array([-np.inf, 1.0, 2.0], dtype=np.float32)
        moved_mask = cuda_runner.compare_arrays(np, reference, candidate, 1e-4, 1e-3)
        self.assertFalse(moved_mask["parity"])
        self.assertEqual(moved_mask["elements_outside_tolerance"], 2)

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

    def test_tiered_inference_tolerance_policy(self):
        import numpy as np
        sys.path.insert(0, str(HERE))
        try:
            cuda_bench = load("benchmark_cuda_models")
        finally:
            if sys.path and sys.path[0] == str(HERE):
                sys.path.pop(0)
        # float32 remains strict (any element outside tolerance fails)
        a = np.ones(1000, dtype=np.float32)
        b = a.copy()
        b[0] += 5e-3
        tol32 = cuda_bench.INFERENCE_TOLERANCES["float32"]
        res32 = cuda_bench.compare_arrays(
            np, a, b, tol32["atol"], tol32["rtol"],
            max_outside_fraction=tol32["max_outside_fraction"],
            min_cosine=tol32["min_cosine"],
            max_abs_cap=tol32["max_abs_cap"],
            max_scale_ratio=tol32["max_scale_ratio"],
        )
        self.assertFalse(res32["parity"])

        # bfloat16 allows bounded 1-ULP outliers on large tensors when cosine is high
        tol16 = cuda_bench.INFERENCE_TOLERANCES["bfloat16"]
        ref = np.linspace(-2.0, 2.0, 2000, dtype=np.float32)
        cand_ok = ref.copy()
        cand_ok[0] += 0.12  # single 1-ULP style outlier (0.05% < 0.5%, <= 0.25 cap)
        res_ok = cuda_bench.compare_arrays(
            np, ref, cand_ok, tol16["atol"], tol16["rtol"],
            max_outside_fraction=tol16["max_outside_fraction"],
            min_cosine=tol16["min_cosine"],
            max_abs_cap=tol16["max_abs_cap"],
            max_scale_ratio=tol16["max_scale_ratio"],
        )
        self.assertTrue(res_ok["parity"])
        self.assertGreater(res_ok["cosine_similarity"], 0.999)

        # bfloat16 rejects large computation errors and NaN/Inf mismatches
        cand_bad = ref.copy()
        cand_bad[:50] += 1.0  # 2.5% outliers + large abs diff
        res_bad = cuda_bench.compare_arrays(
            np, ref, cand_bad, tol16["atol"], tol16["rtol"],
            max_outside_fraction=tol16["max_outside_fraction"],
            min_cosine=tol16["min_cosine"],
            max_abs_cap=tol16["max_abs_cap"],
            max_scale_ratio=tol16["max_scale_ratio"],
        )
        self.assertFalse(res_bad["parity"])

        cand_nan = ref.copy()
        cand_nan[0] = np.nan
        res_nan = cuda_bench.compare_arrays(
            np, ref, cand_nan, tol16["atol"], tol16["rtol"],
            max_outside_fraction=tol16["max_outside_fraction"],
            min_cosine=tol16["min_cosine"],
            max_abs_cap=tol16["max_abs_cap"],
            max_scale_ratio=tol16["max_scale_ratio"],
        )
        self.assertFalse(res_nan["parity"])
        self.assertEqual(res_nan["nonfinite_mismatch"], 1)

        # multi-step diffusion sampling tier admits small 20-step rounding drift at 0.9999+ cosine
        tol_ms = cuda_bench.MULTI_STEP_INFERENCE_TOLERANCES["float16"]
        cand_ms = ref.copy()
        cand_ms[:100] += np.where(np.arange(100) % 2 == 0, 0.12, -0.12)  # 5% zero-mean rounding outliers, cosine ~0.9997
        res_ms = cuda_bench.compare_arrays(
            np, ref, cand_ms, tol_ms["atol"], tol_ms["rtol"],
            max_outside_fraction=tol_ms["max_outside_fraction"],
            min_cosine=tol_ms["min_cosine"],
            max_abs_cap=tol_ms["max_abs_cap"],
            max_scale_ratio=tol_ms["max_scale_ratio"],
        )
        self.assertTrue(res_ms["parity"])
        cand_ms_bad = ref.copy()
        cand_ms_bad[:400] += 0.8  # 20% outliers + exceeds max_abs_cap
        res_ms_bad = cuda_bench.compare_arrays(
            np, ref, cand_ms_bad, tol_ms["atol"], tol_ms["rtol"],
            max_outside_fraction=tol_ms["max_outside_fraction"],
            min_cosine=tol_ms["min_cosine"],
            max_abs_cap=tol_ms["max_abs_cap"],
            max_scale_ratio=tol_ms["max_scale_ratio"],
        )
        self.assertFalse(res_ms_bad["parity"])


if __name__ == "__main__":
    unittest.main()
