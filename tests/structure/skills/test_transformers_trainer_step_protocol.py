import unittest
from types import SimpleNamespace
from contextlib import ExitStack
import importlib.util
from pathlib import Path

from _helpers.child_process import PYTHON, child_env


def setUpModule():
    global protocol, ReadbackGuard, GateWindow
    protocol_path = (
        Path(__file__).resolve().parents[3]
        / "benchmarks/profiling/_trainer_step_protocol.py"
    )
    spec = importlib.util.spec_from_file_location("_trainer_step_protocol", protocol_path)
    protocol = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(protocol)
    ReadbackGuard, GateWindow = protocol.ReadbackGuard, protocol.GateWindow


class Tensor:
    def __init__(self, device):
        self.device = SimpleNamespace(type=device)
        self.original_calls = []

    def _call(self, name, result):
        self.original_calls.append(name)
        return result

    def cpu(self):
        return self._call("cpu", self)

    def numpy(self):
        return self._call("numpy", [])

    def item(self):
        return self._call("item", 1)

    def tolist(self):
        return self._call("tolist", [])

    def __bool__(self):
        return self._call("__bool__", True)

    def __float__(self):
        return self._call("__float__", 1.0)

    def __int__(self):
        return self._call("__int__", 1)


class GuardTests(unittest.TestCase):
    def test_all_accelerator_readbacks_rejected_before_original(self):
        for device in ["npu", "cuda"]:
            for method in ReadbackGuard.METHODS:
                with self.subTest(device=device, method=method), ExitStack() as stack:
                    value = Tensor(device)
                    guard = ReadbackGuard(GateWindow(), Tensor)
                    guard.install(stack)
                    with self.assertRaisesRegex(
                        RuntimeError, "accelerator readback " + device + "." + method
                    ):
                        getattr(value, method)()
                    self.assertEqual(value.original_calls, [])
                    self.assertEqual(guard.inside_accelerator, {method: 1})

    def test_cpu_optimizer_item_allowed_by_actual_device(self):
        with ExitStack() as stack:
            value = Tensor("cpu")
            guard = ReadbackGuard(GateWindow(), Tensor)
            guard.install(stack)
            self.assertEqual(value.item(), 1)
            self.assertEqual(guard.inside_cpu, {"item": 1})
            self.assertEqual(guard.inside_accelerator, {})

    def test_outside_window_archival_allowed(self):
        with ExitStack() as stack:
            window = GateWindow()
            window.in_timed_update = False
            value = Tensor("npu")
            guard = ReadbackGuard(window, Tensor)
            guard.install(stack)
            self.assertIs(value.cpu(), value)
            self.assertEqual(guard.outside_counts, {"cpu": 1})


class PersistentControlTests(unittest.TestCase):
    shapes = [str(index) for index in range(8)]

    def ready_row(self):
        return {
            "shadow_complete": True,
            "initial_reset_exact": True,
            "phase_label": "synchronized training phases",
            "attention": "eager",
            "shape_inventory": self.shapes,
            "initial_sha256": "a" * 64,
        }

    def control(self):
        result = protocol.PersistentStepControl()
        for runtime in ("torch", "jittor"):
            result.ready(runtime, self.ready_row())
        return result

    def row(self, step):
        index = (step - 1) % 4 * 2
        return {
            "shapes": self.shapes[index : index + 2],
            "sample_ids": [[index * 2, index * 2 + 1], [index * 2 + 2, index * 2 + 3]],
            "seconds": None if step <= 4 else 0.1,
            "synchronized": True,
        }

    def complete(self):
        control = self.control()
        for step in range(1, 15):
            for runtime in ("torch", "jittor"):
                control.grant(runtime, step)
                control.done(runtime, step, self.row(step))
        return control

    def test_four_warm_ten_steady_pairs(self):
        control = self.complete()
        control.finalize("torch")
        control.finalize("jittor")
        result = control.finish()
        self.assertEqual(result["steady_samples"], 20)
        self.assertEqual(result["updates_per_runtime"], 14)
        self.assertFalse(result["single_model_hbm_claim"])
        self.assertFalse(result["performance_claim"])

    def test_no_overlap_or_wrong_order(self):
        control = self.control()
        with self.assertRaises(ValueError):
            control.grant("jittor", 1)
        control.grant("torch", 1)
        with self.assertRaises(ValueError):
            control.grant("jittor", 1)

    def test_no_grant_before_both_ready(self):
        with self.assertRaises(ValueError):
            protocol.PersistentStepControl().grant("torch", 1)

    def test_initial_reset_shadow_attention_and_weight_guards(self):
        for patch in (
            {"initial_reset_exact": False},
            {"shadow_complete": False},
            {"attention": "sdpa"},
            {"phase_label": ""},
            {"shape_inventory": ["x"]},
        ):
            control = protocol.PersistentStepControl()
            row = self.ready_row()
            row.update(patch)
            with self.assertRaises(ValueError):
                control.ready("torch", row)
        control = protocol.PersistentStepControl()
        control.ready("torch", self.ready_row())
        row = self.ready_row()
        row["initial_sha256"] = "b" * 64
        with self.assertRaises(ValueError):
            control.ready("jittor", row)

    def test_unsynced_invalid_samples_and_input_mismatch(self):
        for patch in ({"synchronized": False}, {"seconds": 0.1}, {"shapes": ["invalid", "0"]}):
            control = self.control()
            control.grant("torch", 1)
            row = self.row(1)
            row.update(patch)
            with self.assertRaises(ValueError):
                control.done("torch", 1, row)
        control = self.control()
        control.grant("torch", 1)
        control.done("torch", 1, self.row(1))
        control.grant("jittor", 1)
        row = self.row(1)
        row["sample_ids"] = [[999]]
        with self.assertRaises(ValueError):
            control.done("jittor", 1, row)

    def test_no_early_or_unserialized_finalization(self):
        with self.assertRaises(ValueError):
            self.control().finalize("torch")
        control = self.complete()
        with self.assertRaises(ValueError):
            control.finalize("jittor")
        with self.assertRaises(ValueError):
            control.finish()

    def test_timing_guard_full_budget_shape_and_compile(self):
        clock = [0.0]
        timer = protocol.UpdateTimer(
            lambda: None, required_shapes=self.shapes, clock=lambda: clock[0]
        )
        for step in range(1, 15):
            timer.begin(step)
            for shape in self.row(step)["shapes"]:
                timer.microbatch(shape)
            clock[0] += 0.1
            timer.end(step)
        self.assertEqual(timer.finish("stable", "stable")["updates"], 10)
        with self.assertRaises(RuntimeError):
            timer.finish("old", "new")

    def test_missing_shapes_and_observation_inside_rejected(self):
        timer = protocol.UpdateTimer(lambda: None, required_shapes=self.shapes)
        for step in range(1, 5):
            timer.begin(step)
            timer.microbatch("0")
            timer.end(step)
        with self.assertRaises(RuntimeError):
            timer.begin(5)


class CompleteStateTests(unittest.TestCase):
    def test_schema_unknown_key_hyperparameter_and_step_are_exact(self):
        import numpy as np
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            (left, right) = (Path(directory) / "a.npz", Path(directory) / "b.npz")
            payload = {
                "model/weight": np.ones(2, dtype=np.float32),
                "optimizer/param_groups/0/lr": np.asarray(0.003),
                "optimizer/state/0/step": np.asarray(14.0),
                "rng/torch": np.array([1], dtype=np.uint8),
            }
            np.savez(left, **payload)
            for key, value in [
                ("scheduler/verbose", np.asarray(False)),
                ("optimizer/param_groups/0/lr", np.asarray(0.003001)),
                ("optimizer/state/0/step", np.asarray(14.001)),
            ]:
                changed = dict(payload)
                changed[key] = value
                np.savez(right, **changed)
                self.assertEqual(protocol.compare_arrays(left, right)["status"], "FAIL")
            changed = dict(payload)
            changed["rng/torch"] = np.array([1, 2], dtype=np.uint8)
            np.savez(right, **changed)
            self.assertEqual(protocol.compare_arrays(left, right)["status"], "PASS")
            self.assertEqual(
                protocol.compare_arrays(left, right, same_runtime=True)["status"], "FAIL"
            )

    def test_all_expected_gradients_and_state_inventory_required(self):
        import numpy as np
        import tempfile
        import json

        with tempfile.TemporaryDirectory() as directory:
            dirs = [Path(directory) / name for name in ("left", "right")]
            for root in dirs:
                root.mkdir()
                payload = {
                    "model/weight": np.ones(2, dtype=np.float32),
                    "rng/torch": np.array([1], dtype=np.uint8),
                }
                for phase in ("functional", "plain"):
                    np.savez(root / (phase + "-state.npz"), **payload)
                np.savez(
                    root / "functional-gradients.npz",
                    **{
                        "micro/00/weight": np.ones(2),
                        "micro/01/weight": np.ones(2),
                        "postclip/00/weight": np.ones(2),
                    },
                )
                np.save(root / "functional-losses.npy", np.array([1.0, 1.0]))
                (root / "functional-summary.json").write_text(
                    json.dumps(
                        {
                            "microbatches": 2,
                            "global_step": 1,
                            "trainable_names": ["weight"],
                            "state_array_count": 2,
                        }
                    )
                )
            self.assertEqual(protocol.compare_training_outputs(*dirs)["status"], "PASS")
            for root in dirs:
                np.savez(root / "functional-gradients.npz", **{"micro/00/weight": np.ones(2)})
            self.assertEqual(protocol.compare_training_outputs(*dirs)["status"], "FAIL")

    def test_artifacts_stay_inside_lab_outside_checkout(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lab = root / "lab"
            repo = root / "repo"
            self.assertEqual(protocol.checked_lab_path(lab / "result", lab, repo), lab / "result")
            with self.assertRaises(ValueError):
                protocol.checked_lab_path(repo / "result", lab, repo)


class RetainedCohortLedgerTests(unittest.TestCase):
    def test_retained_cohort_contracts(self):
        p = {
            "sha256": "a" * 64,
            "runtime_sources": {"torch": "b" * 64, "jittor": "c" * 64},
            "lease_id": "EXTERNAL-HOST-TEST-ONLY",
            "physical_mask": "EXTERNAL-MASK",
            "affinity": ["assigned-a", "assigned-b"],
            "shape_inventory": [str(x) for x in [22, 25, 29, 32, 37, 41, 45, 47]],
        }

        def row(index=0):
            runtime = ["torch", "jittor"][index % 2]
            return {
                "cohort": "HOST-CONTROL-" + str(index),
                "runtime": runtime,
                "protocol_sha256": p["sha256"],
                "runtime_source_sha256": p["runtime_sources"][runtime],
                "functional_status": "PASS",
                "functional_all_gradients_status": "PASS",
                "functional_complete_state_status": "PASS",
                "state_schema_complete": True,
                "functional_artifact_sha256": "d" * 64,
                "lease_id": p["lease_id"],
                "physical_mask": "EXTERNAL-MASK",
                "logical_device": 0,
                "visible_device_count": 1,
                "actual_pid_on_physical_device": True,
                "placement_tensor_count": 100,
                "fallback_count": 0,
                "threads": 2,
                "affinity": p["affinity"],
                "warmup_updates": 4,
                "measured_updates": 10,
                "warmed_shapes": p["shape_inventory"],
                "measured_update_steps": list(range(5, 15)),
                "seconds_per_update": [2.0] * 10,
                "optimizer_skipped": 0,
                "synchronized_begin_end": True,
                "device_readbacks": 0,
                "observation_save_eval_inside": 0,
                "jit_before": "stable-native",
                "jit_after": "stable-native",
                "vendor_compile_before": "stable-vendor",
                "vendor_compile_after": "stable-vendor",
                "logging_nan_inf_filter": False,
                "logging_eval_save_disabled": True,
                "inputs_preplaced": True,
                "sample_count": 40,
                "unmasked_token_count": 1000,
                "attention_implementation": "eager",
                "functional_attention_implementation": "eager",
                "phase_label": "synchronized training phases",
                "phase_barrier_position": "postclip-preoptimizer",
                "phase_barrier_steps": list(range(1, 15)),
                "actual_npu_sdpa_rejection_status": "ACTUAL_NPU_SDPA_READBACK_REJECTED",
                "actual_npu_sdpa_rejection_artifact_sha256": "e" * 64,
            }

        cases = []
        mutations = {
            "wrong_order": {"runtime": "jittor"},
            "changed_protocol": {"protocol_sha256": "x"},
            "changed_runtime_source": {"runtime_source_sha256": "x"},
            "failed_functional": {"functional_status": "FAIL"},
            "missing_functional_artifact": {"functional_artifact_sha256": ""},
            "missing_external_lease": {"lease_id": ""},
            "wrong_physical_mask": {"physical_mask": 3},
            "missing_actual_pid": {"actual_pid_on_physical_device": False},
            "wrong_placement": {"logical_device": 1},
            "fallback": {"fallback_count": 1},
            "wrong_threads": {"threads": 1},
            "short_warmup": {"warmup_updates": 2},
            "partial_shape_inventory": {"warmed_shapes": ["22"]},
            "short_updates": {"measured_updates": 9},
            "incomplete_update": {"measured_update_steps": list(range(5, 14))},
            "invalid_duration": {"seconds_per_update": [float("nan")] * 10},
            "optimizer_skip": {"optimizer_skipped": 1},
            "unsynchronized": {"synchronized_begin_end": False},
            "device_readback": {"device_readbacks": 1},
            "observer_inside": {"observation_save_eval_inside": 1},
            "new_jit": {"jit_after": "new"},
            "new_vendor_compile": {"vendor_compile_after": "new"},
            "missing_vendor_evidence": {
                "vendor_compile_before": None,
                "vendor_compile_after": None,
            },
            "nan_filter_readback": {"logging_nan_inf_filter": True},
            "logging_enabled": {"logging_eval_save_disabled": False},
            "inputs_not_preplaced": {"inputs_preplaced": False},
            "missing_token_count": {"unmasked_token_count": 0},
            "wrong_attention": {"attention_implementation": "sdpa"},
            "functional_attention_mismatch": {"functional_attention_implementation": "sdpa"},
            "phase_label_missing": {"phase_label": ""},
            "phase_position_changed": {"phase_barrier_position": "backward"},
            "incomplete_phase_boundaries": {"phase_barrier_steps": list(range(2, 15))},
            "missing_real_sdpa_rejection": {"actual_npu_sdpa_rejection_status": "HOST_ONLY"},
            "missing_real_sdpa_artifact": {"actual_npu_sdpa_rejection_artifact_sha256": ""},
            "incomplete_state": {"functional_complete_state_status": "FAIL"},
            "incomplete_gradients": {"functional_all_gradients_status": "FAIL"},
            "incomplete_schema": {"state_schema_complete": False},
        }
        for label, values in mutations.items():
            sequence = protocol.AdmissionSequence(p)
            candidate = row()
            candidate.update(values)
            try:
                sequence.append(candidate)
            except ValueError:
                assert not sequence.admissions and sequence.attempts[-1]["status"] == "FAIL"
                cases.append({"case": label, "status": "PASS"})
            else:
                raise AssertionError(label)
        sequence = protocol.AdmissionSequence(p)
        for i in range(20):
            sequence.append(row(i))
        result = sequence.finish()
        assert (
            result["admitted_cohorts"] == 20
            and result["admitted_pairs"] == 10
            and (result["runtime_summaries"]["torch"]["cohorts"] == 10)
            and (result["runtime_summaries"]["jittor"]["updates"] == 100)
            and (not result["performance_claim"])
        )
        cases.append(
            {
                "case": "ten_pairs_twenty_alternating_complete_caller_gate_admissions",
                "status": "PASS",
            }
        )
        sequence = protocol.AdmissionSequence(p)
        sequence.append(row())
        try:
            sequence.finish()
        except ValueError:
            cases.append({"case": "partial_series_cannot_finish", "status": "PASS"})
        else:
            raise AssertionError("partial finish")
        try:
            sequence.append(row())
        except ValueError:
            cases.append({"case": "duplicate_cohort_identity_rejected", "status": "PASS"})
        else:
            raise AssertionError("duplicate")


class ProcessOwnershipTests(unittest.TestCase):
    def _start(self, code):
        import subprocess

        return subprocess.Popen([PYTHON, "-B", "-c", code], env=child_env(), start_new_session=True)

    def test_no_connection_has_bounded_deadline_and_owned_cleanup(self):
        from multiprocessing.connection import Listener
        import time

        process = self._start("import time;time.sleep(20)")
        identity = protocol.capture_process_identity(process)
        listener = Listener(("127.0.0.1", 0), authkey=b"host-only")
        started = time.monotonic()
        try:
            with self.assertRaises(TimeoutError):
                protocol.accept_worker_connection(listener, process, 0.15)
        finally:
            listener.close()
            rows = protocol.cleanup_owned_workers({"owned": process}, {"owned": identity}, grace=1)
        self.assertLess(time.monotonic() - started, 3)
        self.assertIsNotNone(process.returncode)
        self.assertEqual(rows[0]["signals"], ["SIGTERM"])

    def test_crash_before_connect_is_detected_and_reaped(self):
        from multiprocessing.connection import Listener

        process = self._start("raise SystemExit(7)")
        identity = protocol.capture_process_identity(process)
        listener = Listener(("127.0.0.1", 0), authkey=b"host-only")
        try:
            with self.assertRaisesRegex(RuntimeError, "exited before connecting"):
                protocol.accept_worker_connection(listener, process, 2)
        finally:
            listener.close()
            rows = protocol.cleanup_owned_workers({"owned": process}, {"owned": identity}, grace=1)
        self.assertEqual(process.returncode, 7)
        self.assertEqual(rows[0]["signals"], [])

    def test_identity_mismatch_refuses_signal(self):
        process = self._start("import time;time.sleep(20)")
        identity = protocol.capture_process_identity(process)
        wrong = dict(identity, start_ticks=identity["start_ticks"] + 1)
        try:
            rows = protocol.cleanup_owned_workers({"owned": process}, {"owned": wrong}, grace=0.1)
            self.assertTrue(rows[0]["identity_mismatch_refused_signal"])
            self.assertIsNone(process.poll())
        finally:
            protocol.cleanup_owned_workers({"owned": process}, {"owned": identity}, grace=1)


class DtypeAndNpuRngTests(unittest.TestCase):
    def test_equal_values_with_different_dtype_are_rejected(self):
        import tempfile
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ["left.npz", "right.npz"])
            np.savez(left, **{"model/weight": np.array([1], np.float16)})
            np.savez(right, **{"model/weight": np.array([1], np.float32)})
            for same_runtime in [False, True]:
                result = protocol.compare_arrays(left, right, same_runtime=same_runtime)
                self.assertEqual(result["status"], "FAIL")
                self.assertEqual(result["failures"][0]["reason"], "dtype")

    def test_opaque_npu_bytes_and_list_schema_have_separate_contracts(self):
        import tempfile
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            left, right = (Path(directory) / name for name in ["left.npz", "right.npz"])
            np.savez(
                left,
                **{"rng/npu/@kind": np.asarray("list"), "rng/npu/0": np.arange(8, dtype=np.uint8)},
            )
            np.savez(
                right,
                **{"rng/npu/@kind": np.asarray("list"), "rng/npu/0": np.arange(4, dtype=np.uint8)},
            )
            result = protocol.compare_arrays(left, right)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual([r["name"] for r in result["opaque_rng"]], ["rng/npu/0"])
            np.savez(
                right,
                **{"rng/npu/@kind": np.asarray("tuple"), "rng/npu/0": np.arange(4, dtype=np.uint8)},
            )
            self.assertEqual(protocol.compare_arrays(left, right)["status"], "FAIL")
            np.savez(
                right,
                **{"rng/npu/@kind": np.asarray("list"), "rng/npu/0": np.arange(4, dtype=np.int64)},
            )
            self.assertEqual(protocol.compare_arrays(left, right)["status"], "FAIL")


if __name__ == "__main__":
    unittest.main(verbosity=2)
