"""Pure Python contracts for the real-data HF release acceptance harness."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _release_hf_data import EncodedDataset, resolve_initial
from _release_hf_observe import observer_classes
from _release_hf_lifecycle import assert_logits_equal, assert_merge_equal
from _release_hf_precision import validate_optimizer_attempt
from _release_hf_state import compare_array, compare_runs, signature, state_items
from _release_hf_workflow_runner import parser


def _report(start=0, end=8):
    return {
        "runtime": "torch", "task": "bert", "device": "cpu", "mixed_precision": "no",
        "seed": 821, "versions": {"transformers": "fixed-version"},
        "initialization": {"task": "bert", "compact": True},
        "initial_model_dir": "fixed-initial/base", "initial_adapter_dir": None,
        "eval_samples": 7, "fixed_order": False, "sampler": "default Trainer sampler",
        "num_workers": 0, "gradient_checkpointing": False,
        "microbatches_per_epoch": 7, "updates_per_epoch": 4, "expected_updates": 8,
        "gradient_norm_algorithm": "fp64_pairwise_256_v1",
        "max_grad_norm": 1.0,
        "optimizer_gradient_phase": "postclip_unscaled",
        "global_step": end, "fallback_count": 0, "actual_parameter_devices": ["cpu"],
        "resume_checkpoint": "checkpoints/checkpoint-%d" % start if start else None,
        "resume_loaded_state_exact": True if start else None,
        "train_file_sha256": "fixed-train", "eval_file_sha256": "fixed-eval",
        "train_samples": 13, "epochs": 2, "batch_size": 2,
        "gradient_accumulation_steps": 2, "state_format": "full", "start_step": start,
        "microbatches": [{"step": step, "sample_ids": [step * 2, step * 2 + 1], "loss": 1.0 / step}
                         for step in range(start + 1, end + 1)],
        "updates": [{"step": step, "learning_rates": [0.01 / step], "microbatches": 1,
                     "sample_count": 2, "optimizer_step_skipped": False,
                     "mean_microbatch_loss": 1.0 / step}
                    for step in range(start + 1, end + 1)],
        "unscaled_gradients": [{"step": step, "l2_norm": 0.5 / step,
                                "l2_norm_algorithm": "fp64_pairwise_256_v1",
                                "gradient_phase": "postclip_unscaled"}
                               for step in range(start + 1, end + 1)],
    }


def _write_run(path, report):
    path.mkdir()
    (path / "report.json").write_text(json.dumps(report))
    np.savez(path / "final-state.npz", **{"parameter": np.asarray([1.0, 2.0])})


class TestReleaseHFWorkflowContracts(unittest.TestCase):
    def test_state_items_rejects_group_count_mismatch_in_both_directions(self):
        model = SimpleNamespace(named_parameters=lambda: iter(()),
                                named_buffers=lambda: iter(()))
        scheduler = SimpleNamespace(state_dict=lambda: {})
        for live_count, saved_count in ((0, 1), (1, 0)):
            with self.subTest(live=live_count, saved=saved_count):
                saved = {"param_groups": [{"params": []} for _ in range(saved_count)],
                         "state": {}}
                optimizer = SimpleNamespace(
                    param_groups=[{"params": []} for _ in range(live_count)],
                    state_dict=lambda: saved)
                with self.assertRaisesRegex(AssertionError, "parameter group count"):
                    list(state_items(model, optimizer, scheduler, None))

    def test_state_items_keeps_all_matching_group_metadata(self):
        model = SimpleNamespace(named_parameters=lambda: iter(()),
                                named_buffers=lambda: iter(()))
        scheduler = SimpleNamespace(state_dict=lambda: {})
        groups = [{"params": [], "lr": value} for value in (0.1, 0.2)]
        optimizer = SimpleNamespace(param_groups=groups,
                                    state_dict=lambda: {"param_groups": groups, "state": {}})
        values = dict(state_items(model, optimizer, scheduler, None))
        self.assertEqual(float(values["optimizer/group/0000/lr"]), 0.1)
        self.assertEqual(float(values["optimizer/group/0001/lr"]), 0.2)

    def test_comparison_rejects_mismatched_protocol_with_equal_final_tensors(self):
        changes = {
            "runtime": "jittor", "seed": 999, "versions": {"transformers": "other"},
            "initialization": {"task": "bert", "compact": False},
            "initial_model_dir": "other/base", "initial_adapter_dir": "other/adapter",
            "eval_samples": 9, "fixed_order": True, "sampler": "sequential fixture",
            "num_workers": 2, "gradient_checkpointing": True,
            "microbatches_per_epoch": 8, "updates_per_epoch": 5,
            "expected_updates": 10, "global_step": 9, "max_grad_norm": 0.5,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (field, value) in enumerate(changes.items()):
                with self.subTest(field=field):
                    full = root / ("full-%d" % index)
                    resume = root / ("resume-%d" % index)
                    _write_run(full, _report())
                    changed = _report(start=5)
                    changed[field] = value
                    _write_run(resume, changed)
                    args = SimpleNamespace(left=str(full), right=str(resume), first=str(full),
                                           atol=0, rtol=0, report=None)
                    with self.assertRaisesRegex(AssertionError, field):
                        compare_runs(args)

    def test_comparison_rejects_mixed_gradient_measurements(self):
        for field, changed in (("gradient_norm_algorithm", "legacy_generic_fp32_sum"),
                               ("optimizer_gradient_phase", "pre_optimizer_unscaled")):
            for change_event in (False, True):
                with self.subTest(field=field, change_event=change_event):
                    with tempfile.TemporaryDirectory() as temporary:
                        root = Path(temporary)
                        full, resumed = _report(), _report(start=5)
                        if change_event:
                            event_field = ("l2_norm_algorithm" if field == "gradient_norm_algorithm"
                                           else "gradient_phase")
                            resumed["unscaled_gradients"][0][event_field] = changed
                        else:
                            resumed[field] = changed
                        _write_run(root / "full", full)
                        _write_run(root / "resume", resumed)
                        args = SimpleNamespace(left=str(root / "full"), right=str(root / "resume"),
                                               first=str(root / "full"), atol=0, rtol=0, report=None)
                        with self.assertRaisesRegex(AssertionError, field):
                            compare_runs(args)

    def test_comparison_rejects_invalid_device_fallback_resume_or_missing_evidence(self):
        changes = [
            ("fallback_count", 1, "fallback_count"),
            ("actual_parameter_devices", ["cuda:0"], "actual_parameter_devices"),
            ("actual_parameter_devices", [], "actual_parameter_devices"),
            ("resume_loaded_state_exact", False, "resume_loaded_state_exact"),
            ("resume_loaded_state_exact", None, "resume_loaded_state_exact"),
            ("resume_checkpoint", None, "resume_loaded_state_exact"),
            ("seed", None, "missing seed"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (field, value, message) in enumerate(changes):
                with self.subTest(field=field, value=value):
                    full = root / ("full-%d" % index)
                    resume = root / ("resume-%d" % index)
                    _write_run(full, _report())
                    changed = _report(start=5)
                    if field == "seed":
                        del changed[field]
                    else:
                        changed[field] = value
                    _write_run(resume, changed)
                    args = SimpleNamespace(left=str(full), right=str(resume), first=str(full),
                                           atol=0, rtol=0, report=None)
                    with self.assertRaisesRegex(AssertionError, message):
                        compare_runs(args)

    def test_resume_comparison_rejects_wrong_first_run_protocol_and_incomplete_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root / "full", _report())
            _write_run(root / "resume", _report(start=5))
            changes = [("seed", 999, "seed"), ("global_step", 4, "resume prefix"),
                       ("start_step", 1, "resume prefix")]
            for index, (field, value, message) in enumerate(changes):
                with self.subTest(field=field):
                    first = _report()
                    first[field] = value
                    if field == "start_step":
                        first["resume_checkpoint"] = "checkpoints/checkpoint-1"
                        first["resume_loaded_state_exact"] = True
                    _write_run(root / ("first-%d" % index), first)
                    args = SimpleNamespace(left=str(root / "full"), right=str(root / "resume"),
                                           first=str(root / ("first-%d" % index)),
                                           atol=0, rtol=0, report=None)
                    with self.assertRaisesRegex(AssertionError, message):
                        compare_runs(args)

    def test_repeated_resume_segment_keeps_zero_tolerance_trajectory_comparison(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("repeat-one", "repeat-two"):
                _write_run(root / name, _report(start=5, end=6))
            output = root / "comparison.json"
            args = SimpleNamespace(left=str(root / "repeat-one"), right=str(root / "repeat-two"),
                                   first=str(root / "repeat-one"), atol=0, rtol=0,
                                   report=str(output))
            with redirect_stdout(io.StringIO()):
                compare_runs(args)
            result = json.loads(output.read_text())
            self.assertTrue(result["passed"])
            self.assertTrue(result["sample_cursor_exact"])
            self.assertTrue(result["update_count_exact"])
            self.assertEqual(result["atol"], 0)
            self.assertEqual(result["rtol"], 0)

    def test_first_run_cannot_reintroduce_events_before_its_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root / "left", _report(start=3))
            _write_run(root / "right", _report(start=5))
            first = _report(start=3)
            first["microbatches"].insert(0, {"step": 3, "sample_ids": [6, 7], "loss": 1.0})
            _write_run(root / "first", first)
            args = SimpleNamespace(left=str(root / "left"), right=str(root / "right"),
                                   first=str(root / "first"), atol=0, rtol=0, report=None)
            with self.assertRaisesRegex(AssertionError, "outside its segment"):
                compare_runs(args)

    def test_stopped_first_run_is_a_valid_resume_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, report in (("full", _report()), ("resume", _report(start=5)),
                                 ("first", _report(end=5))):
                _write_run(root / name, report)
            args = SimpleNamespace(left=str(root / "full"), right=str(root / "resume"),
                                   first=str(root / "first"), atol=0, rtol=0, report=None)
            with redirect_stdout(io.StringIO()):
                compare_runs(args)

    def test_signature_rejects_nonfinite_values_before_hashing(self):
        for value in (float("nan"), float("inf"), -float("inf"),
                      complex(1.0, float("nan"))):
            with self.subTest(value=value), self.assertRaisesRegex(
                    AssertionError, "nonfinite state tensor at optimizer/exp_avg"):
                signature([("optimizer/exp_avg", np.asarray([value]))])

    def test_signature_keeps_finite_hashes_and_metadata_stable(self):
        items = [("parameter", np.asarray([1.0, -2.0], dtype=np.float32)),
                 ("counter", np.asarray(3)), ("boolean", np.asarray(True)),
                 ("dtype", np.asarray("torch.float32")),
                 ("names", np.asarray(["first", "second"])),
                 ("empty", np.asarray([], dtype=np.float32))]
        measured = signature(items)
        self.assertEqual(measured, signature(items))
        self.assertEqual(measured["parameter"]["shape"], [2])
        self.assertEqual(measured["parameter"]["dtype"], "<f4")
        self.assertEqual(measured["parameter"]["sha256"],
                         "1229b9e656c57d0861aa8a8659d183ee72771fbf20097c5c912cb9c743adc96d")

    def test_default_fp16_overflow_requires_skip_unchanged_state_and_half_scale(self):
        before = {"scale": 65536.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                  "growth_interval": 2000, "_growth_tracker": 0}
        after = dict(before, scale=32768.0)
        health = {"nonfinite_count": 3, "l2_norm": None}
        result = validate_optimizer_attempt("fp16", True, health, True, before, after, True)
        self.assertFalse(result["successful_update"])
        # Early accumulated overflow remains binding even if a later gradient
        # observation becomes finite: the same optimizer window must skip.
        result = validate_optimizer_attempt("fp16", True, {"nonfinite_count": 0}, True,
                                            before, after, True)
        self.assertTrue(result["overflow"])
        cases = [
            (False, before, after, True, "did not skip"),
            (True, before, after, False, "changed model"),
            (True, before, before, True, "halve"),
            (True, before, dict(after, _growth_tracker=1), True, "reset growth"),
        ]
        for skipped, first, second, unchanged, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(AssertionError, message):
                validate_optimizer_attempt("fp16", True, health, skipped, first, second, unchanged)

    def test_finite_fp16_update_requires_finite_health_and_default_growth_policy(self):
        before = {"scale": 32768.0, "growth_factor": 2.0, "backoff_factor": 0.5,
                  "growth_interval": 2, "_growth_tracker": 0}
        health = {"nonfinite_count": 0, "l2_norm": 1.0}
        after = dict(before, _growth_tracker=1)
        result = validate_optimizer_attempt("fp16", False, health, False, before, after)
        self.assertTrue(result["successful_update"])
        grown = dict(before, scale=65536.0)
        self.assertTrue(validate_optimizer_attempt("fp16", False, health, False,
                                                  after, grown)["successful_update"])
        with self.assertRaisesRegex(AssertionError, "unexpectedly skipped"):
            validate_optimizer_attempt("fp16", False, health, True, before, after, True)
        with self.assertRaisesRegex(AssertionError, "growth policy"):
            validate_optimizer_attempt("fp16", False, health, False, before, grown)
        for precision in ("no", "bf16"):
            with self.subTest(precision=precision), self.assertRaisesRegex(AssertionError, "cannot overflow or skip"):
                validate_optimizer_attempt(precision, False, health, True, None, None)

    def test_merge_budget_rejects_each_independent_boundary(self):
        cases = [
            (np.asarray([100.0]), np.asarray([100.002]), 1.0, 1.0, "logits_max_abs"),
            (np.asarray([1.0]), np.asarray([1.0001]), 1.0, 1.0, "logits_max_global_relative"),
            (np.asarray([10.0]), np.asarray([10.0]), 1.0, 1.00002, "loss_abs"),
        ]
        for expected, observed, expected_loss, observed_loss, metric in cases:
            with self.subTest(metric=metric), self.assertRaisesRegex(AssertionError, metric):
                assert_merge_equal(expected, observed, expected_loss, observed_loss, "merge")
        with self.assertRaisesRegex(AssertionError, "nonfinite"):
            assert_merge_equal(np.ones(1), np.asarray([np.nan]), 1.0, 1.0, "merge")

    def test_merge_calibration_does_not_relax_public_reload(self):
        expected = np.asarray([31.088, 0.0])
        observed = expected + np.asarray([0.0, 0.000252])
        measured = assert_merge_equal(expected, observed, 1.0, 1.00000048, "merge")
        self.assertLess(measured["logits_max_global_relative"], 5e-5)
        with self.assertRaisesRegex(AssertionError, "changed full logits"):
            assert_logits_equal(expected, observed, 2e-5, 2e-4, "public reload")
        zeros = assert_merge_equal(np.zeros(2), np.zeros(2), 0.0, 0.0, "merge")
        self.assertEqual(zeros["logits_max_global_relative"], 0.0)
        with self.assertRaisesRegex(AssertionError, "logits_max_global_relative"):
            assert_merge_equal(np.zeros(2), np.asarray([1e-8, 0.0]), 0.0, 0.0, "merge")

    def test_archive_comparison_rejects_dtype_shape_nonfinite(self):
        cases = [
            (np.ones(2), np.ones(1), "shape_or_dtype"),
            (np.ones(2, dtype=np.float32), np.ones(2), "shape_or_dtype"),
            (np.asarray([np.nan]), np.asarray([np.nan]), "nonfinite"),
            (np.asarray([np.inf]), np.asarray([np.inf]), "nonfinite"),
            (np.asarray([1.0]), np.asarray([1.1]), "numeric"),
        ]
        for left, right, kind in cases:
            self.assertEqual(compare_array("state", left, right, 0, 0)["kind"], kind)
        self.assertIsNone(compare_array("dtype", np.asarray("torch.float32"),
                                        np.asarray("torch.float32"), 0, 0))

    def test_resume_comparison_uses_only_continuous_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root / "full", _report())
            _write_run(root / "resume", _report(start=5))
            output = root / "comparison.json"
            args = SimpleNamespace(left=str(root / "full"), right=str(root / "resume"),
                                   first=str(root / "full"), atol=0, rtol=0, report=str(output))
            with redirect_stdout(io.StringIO()):
                compare_runs(args)
            result = json.loads(output.read_text())
            self.assertTrue(result["passed"])
            self.assertTrue(result["sample_cursor_exact"])
            self.assertTrue(result["update_count_exact"])

    def test_resume_cursor_mismatch_fails_even_when_final_tensors_equal(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root / "full", _report())
            resumed = _report(start=5)
            resumed["microbatches"][0]["sample_ids"] = [99, 100]
            _write_run(root / "resume", resumed)
            args = SimpleNamespace(left=str(root / "full"), right=str(root / "resume"),
                                   first=str(root / "full"), atol=0, rtol=0, report=None)
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                compare_runs(args)

    def test_resume_intermediate_gradient_mismatch_fails_with_equal_final_tensors(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_run(root / "full", _report())
            resumed = _report(start=5)
            resumed["unscaled_gradients"][0]["l2_norm"] += 0.001
            _write_run(root / "resume", resumed)
            args = SimpleNamespace(left=str(root / "full"), right=str(root / "resume"),
                                   first=str(root / "full"), atol=0, rtol=0, report=None)
            with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                compare_runs(args)

    def test_upstream_4562_skip_tail_flush_boundary_is_explicit(self):
        # This arithmetic is copied from the independently inspected HF 4.56.2
        # loop; the actual PT negative run is retained in the release report.
        # Seven microbatches, G=2, resumed after two batches: local steps 0..4
        # never satisfy the upstream full-epoch tail predicate step+1 == 7.
        full_length, skipped, accumulation = 7, 2, 2
        upstream_flushes = [step for step in range(full_length - skipped)
                            if (step + 1) % accumulation == 0 or step + 1 == full_length]
        intended_flushes = [step for step in range(full_length - skipped)
                            if (step + 1) % accumulation == 0 or
                            step + 1 + skipped == full_length]
        self.assertEqual(upstream_flushes, [1, 3])
        self.assertEqual(intended_flushes, [1, 3, 4])
        # The two public-parameter workarounds retain a true data tail: resume
        # at an epoch boundary, or use an even microbatch count for G=2.
        for full_length, skipped in ((7, 0), (6, 2)):
            observed = [step for step in range(full_length - skipped)
                        if (step + 1) % accumulation == 0 or step + 1 == full_length]
            self.assertEqual(len(observed), (full_length - skipped + 1) // 2)

    def test_default_sampler_delegates_to_unmodified_trainer(self):
        class Base:
            def _get_train_sampler(self, dataset):
                return ("public-default", dataset)

        arguments = SimpleNamespace(fixed_order=False)
        trainer_class, _ = observer_classes(None, Base, object, arguments)
        trainer = object.__new__(trainer_class)
        self.assertEqual(trainer._get_train_sampler("dataset"), ("public-default", "dataset"))

    def test_sparse_callback_does_not_request_every_step_checkpoint(self):
        _, callback_class = observer_classes(None, object, object,
                                             SimpleNamespace(save_at=[5, 8], expected_updates=8,
                                                             stop_after=None, mixed_precision="no"))
        owner = SimpleNamespace(
            _pending_batches=[{"batch_size": 2, "loss": 1.0}], updates=[],
            optimizer=SimpleNamespace(param_groups=[{"lr": 0.01}]),
            accelerator=SimpleNamespace(scaler=None, optimizer_step_was_skipped=False),
            _attempt_context={"scaled_nonfinite": False, "unscaled_health": {"nonfinite_count": 0},
                              "scaler_before": None, "overflow_state": None},
            successful_optimizer_updates=0, skipped_optimizer_attempts=0)
        callback = callback_class()
        callback.owner = owner
        control = SimpleNamespace(should_save=False, should_evaluate=False)
        callback.on_step_end(None, SimpleNamespace(global_step=1, epoch=0.25), control)
        self.assertFalse(control.should_save)
        self.assertFalse(control.should_evaluate)
        owner._pending_batches = [{"batch_size": 1, "loss": 0.5}]
        owner._attempt_context = {"scaled_nonfinite": False, "unscaled_health": {"nonfinite_count": 0},
                                  "scaler_before": None, "overflow_state": None}
        callback.on_step_end(None, SimpleNamespace(global_step=5, epoch=1.25), control)
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_evaluate)

    def test_initialization_task_mismatch_and_mixed_paths_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "initialization.json").write_text(json.dumps({"task": "bert"}))
            args = SimpleNamespace(initialization_dir=temporary, task="t5",
                                   initial_model_dir=None, initial_adapter_dir=None)
            with self.assertRaisesRegex(ValueError, "task"):
                resolve_initial(args)
            args.task, args.initial_model_dir = "bert", "ambiguous"
            with self.assertRaisesRegex(ValueError, "combined"):
                resolve_initial(args)

    def test_seq2seq_target_uses_text_target_and_keeps_sample_id(self):
        class Tokenizer:
            def __call__(self, text=None, text_target=None, **kwargs):
                return {"input_ids": [7, 8] if text_target is not None else [1, 2, 3],
                        "attention_mask": [1, 1] if text_target is not None else [1, 1, 1]}

        dataset = EncodedDataset([{"source": "source", "target": "target"}],
                                 Tokenizer(), "t5", 32, 16, include_sample_id=True)
        item = dataset[0]
        self.assertEqual(item["labels"], [7, 8])
        self.assertEqual(item["input_ids"], [1, 2, 3])
        self.assertEqual(item["sample_id"], 0)

    def test_parser_defaults_keep_public_sampler_and_exact_compare(self):
        args = parser().parse_args(["compare", "--left", "full", "--right", "resumed"])
        self.assertEqual((args.atol, args.rtol), (0, 0))
        args = parser().parse_args(["run", "--runtime", "torch", "--task", "bert",
                                    "--device", "cpu", "--train-file", "train.jsonl",
                                    "--eval-file", "eval.jsonl", "--output-dir", "run"])
        self.assertFalse(args.fixed_order)
        self.assertEqual(args.epochs, 2)


if __name__ == "__main__":
    unittest.main()
