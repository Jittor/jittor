"""Synthetic host-only trace comparator controls; never execution evidence."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

import actor_worker_fixture as actor
from compare_worker_forward_trace import SCHEMA, TRACE_SCRIPTS, compare, controlled_events
from test_compare_training_worker import fixture as worker_fixture
from test_worker_buffer_contract import add_buffer_fixture


def fixture(framework, role="critic", wrapped=False):
    report, previous = worker_fixture(framework, wrapped=framework == "torch")
    report.update(schema=SCHEMA, trace_role=role, requested_stage="infer", status="diagnostic_completed",
                  diagnostic_only=True, hook_removed=True, verl_device_name="npu",
                  trace_script_sha256={name: "d" * 64 for name in TRACE_SCRIPTS})
    report["stages"] = report["stages"][:-1]
    report["training_config"]["gradient_capture"] = False
    arrays = {key: value for key, value in previous.items() if key.startswith(("initial/", "initial_buffer/")) or key in ("bootstrap/all_reduce", "infer/values")}
    if role == "actor":
        report.update(model_spec=deepcopy(actor.MODEL_SPEC), parameter_names=list(actor.parameter_shapes()),
                      parameter_count=3648, training_config=actor.training_config(3),
                      script_sha256={name: "c" * 64 for name in actor.SCRIPT_NAMES},
                      engine_class="verl.workers.engine.fsdp.transformer_impl.FSDPEngineWithLMHead",
                      actual_attention={"<root>": "eager"})
        arrays = {"initial/" + name: np.zeros(shape, np.float32) for name, shape in actor.parameter_shapes().items()}
        arrays.update({"bootstrap/all_reduce": np.array([1, 3], np.float32),
                       "infer/log_probs": np.zeros((2, 4), np.float32),
                       "fixture/old_log_probs": np.zeros((2, 4), np.float32),
                       "fixture/advantages": np.asarray(actor.ADVANTAGES, np.float32),
                       "fixture/input_ids": np.asarray(actor.INPUT_IDS, np.int64),
                       "fixture/response_mask": np.ones((2, 4), np.float32)})
        add_buffer_fixture(report, arrays)
    trace = {"hook_api": "torch.nn.Module.register_forward_hook", "active_stage": "infer",
             "order": "forward-hook completion", "output_mutation": False,
             "host_snapshots_synchronize": True, "max_calls": 128, "max_leaf_elements": 4096, "events": []}
    report["forward_trace"] = trace
    calls = {}
    for index, (cls, path, leaves) in enumerate(controlled_events(role)):
        call = calls.get(cls, 0)
        calls[cls] = call + 1
        event = {"call_index": index, "class_name": cls, "module_path": ("module." + path if path else "module") if wrapped else path,
                 "source_class": "synthetic." + cls, "class_call_index": call, "module_instance_index": call, "leaves": []}
        trace["events"].append(event)
        for leaf_index, (leaf_path, shape) in enumerate(leaves):
            key = "trace/%03d/%s/class_call_%03d/tensor_%03d" % (index, cls, call, leaf_index)
            event["leaves"].append({"path": leaf_path, "array_key": key, "shape": shape,
                                    "numpy_dtype": "float32", "tensor_dtype": "torch.float32",
                                    "device": "npu:0", "requires_grad": False})
            arrays[key] = np.full(shape, .25, np.float32)
    return report, arrays


class TestForwardTraceComparison(unittest.TestCase):
    def setUp(self):
        self.oracle, self.oa = fixture("torch")
        self.candidate, self.ca = fixture("jittor", wrapped=True)

    def compare(self, atol=1e-6, rtol=1e-6):
        return compare(self.oracle, self.oa, self.candidate, self.ca, atol, rtol)

    def test_both_roles_and_controlled_wrapper(self):
        for role in ("critic", "actor"):
            for wrapped in (False, True):
                with self.subTest(role=role, wrapped=wrapped):
                    result = compare(*fixture("torch", role), *fixture("jittor", role, wrapped))
                    self.assertEqual(result["status"], "diagnostic_equal")
                    self.assertIsNone(result["first_difference"])
                    self.assertTrue(result["diagnostic_only"])
                    self.assertEqual(len(result["comparisons"]), 21)
                    self.assertIsNone(result["oracle_fallback_count"])

    def test_first_difference_is_completion_order_and_relative_tolerance(self):
        key = self.candidate["forward_trace"]["events"][1]["leaves"][1]["array_key"]
        self.ca[key].flat[0] += .125
        result = self.compare()
        self.assertEqual(result["status"], "diagnostic_difference")
        self.assertEqual(result["first_difference"]["class_name"], "Qwen3RotaryEmbedding")
        self.assertEqual(result["first_difference"]["leaf_path"], [["index", 1]])
        self.assertEqual(result["first_difference"]["max_abs_error"], .125)
        self.assertEqual(self.compare(atol=0, rtol=.5)["status"], "diagnostic_equal")

    def test_initial_parameters_require_exact_equality(self):
        key = next(key for key in self.ca if key.startswith("initial/"))
        self.ca[key].flat[0] = 1e-9
        with self.assertRaisesRegex(ValueError, "initial parameters/buffers not exactly equal"):
            self.compare(atol=100, rtol=100)

    def test_new_buffer_schema_and_loading_provenance_required(self):
        for kind in ("old_schema", "both_missing", "wrong_formula", "nonpersistent", "provenance"):
            with self.subTest(kind=kind):
                self.setUp()
                name = "model.rotary_emb.inv_freq"
                if kind == "old_schema":
                    self.candidate["schema"] = "verl-worker-forward-trace-v1"
                elif kind == "both_missing":
                    for report, arrays in ((self.oracle, self.oa), (self.candidate, self.ca)):
                        raw = next(key for key in report["initial_buffers"] if key.endswith(name))
                        del report["initial_buffers"][raw]
                        del arrays["initial_buffer/" + raw]
                elif kind == "wrong_formula":
                    self.ca["initial_buffer/" + name][:] = 0
                elif kind == "nonpersistent":
                    self.candidate["initial_buffers"][name]["persistent"] = True
                else:
                    del self.candidate["loading_provenance"]
                with self.assertRaises(ValueError):
                    self.compare(atol=100, rtol=100)

    def test_both_sides_missing_same_parameter_or_leaf_rejected(self):
        for missing in ("parameter", "leaf", "event"):
            with self.subTest(missing=missing):
                self.setUp()
                for report, arrays in ((self.oracle, self.oa), (self.candidate, self.ca)):
                    if missing == "parameter":
                        name = report["parameter_names"].pop()
                        del arrays["initial/" + name]
                    elif missing == "leaf":
                        leaf = report["forward_trace"]["events"][1]["leaves"].pop()
                        del arrays[leaf["array_key"]]
                    else:
                        report["forward_trace"]["events"].pop()
                with self.assertRaises(ValueError):
                    self.compare()

    def test_manifest_mismatches_rejected(self):
        for key, value in (("status", "passed_requested_scope"), ("trace_role", "actor"),
                           ("model_device", "cpu"), ("actual_backend", "gloo"),
                           ("world_size", 2), ("fallback_count", 1), ("fallback_count", False),
                           ("hook_removed", False), ("diagnostic_only", False)):
            with self.subTest(key=key, value=value):
                self.setUp()
                self.candidate[key] = value
                with self.assertRaises(ValueError):
                    self.compare()
        for key in ("checkpoint_sha256", "script_sha256", "trace_script_sha256"):
            self.setUp()
            name = next(iter(self.candidate[key]))
            self.candidate[key][name] = "f" * 64
            with self.assertRaisesRegex(ValueError, "protocol differs"):
                self.compare()

    def test_event_identity_and_non_generic_wrapper_rejected(self):
        for key, value in (("class_name", "Linear"), ("call_index", 9), ("class_call_index", 1),
                           ("module_instance_index", 2), ("module_path", "module.module.model.embed_tokens")):
            self.setUp()
            self.candidate["forward_trace"]["events"][0][key] = value
            with self.assertRaises(ValueError):
                self.compare()
        self.setUp()
        for event in self.candidate["forward_trace"]["events"]:
            event["module_path"] = "arbitrary." + event["module_path"]
        with self.assertRaises(ValueError):
            self.compare()

    def test_leaf_metadata_and_arrays_fail_closed(self):
        for key, value in (("device", "cpu"), ("device", "npu:1"), ("path", [["index", 0]]),
                           ("shape", [256]), ("numpy_dtype", "float64"), ("array_key", "unknown"),
                           ("requires_grad", True)):
            self.setUp()
            self.candidate["forward_trace"]["events"][0]["leaves"][0][key] = value
            with self.assertRaises(ValueError):
                self.compare()
        for kind in ("missing", "extra", "nan", "inf", "shape", "dtype", "collision"):
            self.setUp()
            key = self.candidate["forward_trace"]["events"][0]["leaves"][0]["array_key"]
            if kind == "missing":
                del self.ca[key]
            elif kind == "extra":
                self.ca["extra"] = np.zeros(1)
            elif kind in ("nan", "inf"):
                self.ca[key].flat[0] = float(kind)
            elif kind == "shape":
                self.ca[key] = self.ca[key].ravel()
            elif kind == "dtype":
                self.ca[key] = self.ca[key].astype(np.float64)
            else:
                param = self.candidate["parameter_names"][0]
                self.ca["initial/_fsdp_wrapped_module." + param] = self.ca["initial/" + param]
            with self.assertRaises(ValueError):
                self.compare()

    def test_no_unknown_oracle_fallback_claim_and_bad_tolerances(self):
        self.oracle["oracle_fallback_count"] = 0
        with self.assertRaises(ValueError):
            self.compare()
        self.setUp()
        for invalid in (-1, float("nan"), float("inf")):
            for atol, rtol in ((invalid, 0), (0, invalid)):
                with self.assertRaises(ValueError):
                    self.compare(atol, rtol)

    def test_endpoint_difference_is_not_invented_module_difference(self):
        self.ca["infer/values"][0, 0] = .1
        result = self.compare()
        self.assertEqual(result["status"], "diagnostic_difference")
        self.assertIsNone(result["first_difference"])
        self.assertFalse(result["infer_equal"])

    def test_cli_writes_separate_json_and_rejects_invalid_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, report, arrays in (("torch", self.oracle, self.oa), ("jittor", self.candidate, self.ca)):
                (root / (name + ".json")).write_text(json.dumps(report))
                np.savez(root / (name + ".npz"), **arrays)
            command = [sys.executable, str(Path(__file__).with_name("compare_worker_forward_trace.py")),
                       "--oracle", str(root / "torch.json"), "--candidate", str(root / "jittor.json"),
                       "--output", str(root / "comparison.json")]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            self.candidate["status"] = "failed"
            (root / "jittor.json").write_text(json.dumps(self.candidate))
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 2)
            self.assertEqual(json.loads((root / "comparison.json").read_text())["status"], "invalid_diagnostic")


if __name__ == "__main__":
    unittest.main()
