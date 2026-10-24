"""Host-only diagnostic contracts; fake tensors here are not hardware evidence."""

from contextlib import ExitStack, redirect_stderr
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import types
import unittest

import numpy as np

from trace_worker_forward import (
    ACTOR_MODEL_SPEC, ForwardTrace, MAX_CALLS, MODEL_SPEC, SCHEMA, TraceRecorder,
    install_trace_hook, parse_args, require_tiny_manifest, tensor_leaves, validate_checkpoint_config,
)


class Tensor:
    def __init__(self, values, device="npu"):
        self.array = np.asarray(values, dtype=np.float32)
        self.device = types.SimpleNamespace(type=device, index=0)
        self.dtype = "torch.float32"
        self.requires_grad = False

    def numel(self):
        return self.array.size


class Embedding:
    pass


class Linear:
    pass


class Qwen3Attention:
    pass


def manifest():
    return {"model_spec": deepcopy(MODEL_SPEC), "parameter_count": 3153,
            "parameter_names": ["parameter_%d" % i for i in range(15)],
            "stages": [{"name": "infer", "status": "running"}],
            "forward_trace": {"events": []}}


class Record:
    def __init__(self):
        self.report, self.arrays = manifest(), {}

    def tensor(self, key, value):
        self.arrays[key] = value.array.copy()
        return self.arrays[key]

    def flush(self):
        pass


def frontend(register=None):
    return types.SimpleNamespace(Tensor=Tensor, nn=types.SimpleNamespace(
        Embedding=Embedding, Linear=Linear,
        modules=types.SimpleNamespace(module=types.SimpleNamespace(register_module_forward_hook=register))))


class TestTraceWorkerForward(unittest.TestCase):
    def test_cli_fixed_infer_scope_and_separate_diagnostic_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--framework", "torch", "--checkpoint", directory,
                               "--output", str(Path(directory) / "trace.json")])
            self.assertEqual((args.stop_after, args.steps, args.capture_gradients, args.prepare_checkpoint),
                             ("infer", 3, False, False))
            record = TraceRecorder(args)
            self.assertEqual(record.report["schema"], SCHEMA)
            self.assertTrue(record.report["diagnostic_only"])
            self.assertFalse(record.report["forward_trace"]["output_mutation"])
            self.assertEqual(record.report["oracle_fallback_status"], "not_instrumented")
            self.assertIsNone(record.report["oracle_fallback_count"])
            self.assertIn("trace_worker_forward.py", record.report["trace_script_sha256"])
            self.assertEqual(record.report["forward_trace"]["hook_api"], "torch.nn.Module.register_forward_hook")
            args = parse_args(["--framework", "jittor", "--role", "actor", "--checkpoint", directory,
                               "--output", str(Path(directory) / "actor-trace.json")])
            actor = TraceRecorder(args)
            self.assertTrue(args.capture_gradients)
            self.assertEqual(actor.report["trace_role"], "actor")
            self.assertIn("probe_actor_worker.py", actor.report["script_sha256"])
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--framework", "torch", "--checkpoint", directory,
                            "--output", str(Path(directory) / "trace.json"), "--stop-after", "train"])

    def test_tiny_reset_gate_rejects_larger_or_partial_model(self):
        require_tiny_manifest(manifest())
        actor = manifest()
        actor.update(trace_role="actor", model_spec=ACTOR_MODEL_SPEC, parameter_count=3648,
                     parameter_names=["parameter_%d" % i for i in range(14)])
        require_tiny_manifest(actor)
        for key, value in (("parameter_count", 3154), ("parameter_names", []), ("model_spec", {})):
            bad = manifest()
            bad[key] = value
            with self.assertRaises(ValueError):
                require_tiny_manifest(bad)

    def test_checkpoint_scope_checked_without_loading_model(self):
        for role, spec, architecture in (("critic", MODEL_SPEC, "Qwen3ForTokenClassification"),
                                         ("actor", ACTOR_MODEL_SPEC, "Qwen3ForCausalLM")):
            config = dict(spec, architectures=[architecture])
            validate_checkpoint_config(config, role)
            if role == "critic":
                config.pop("num_labels")
                config["id2label"] = {"0": "LABEL_0"}
                validate_checkpoint_config(config, role)
            for key, bad_value in (("hidden_size", 4096), ("num_hidden_layers", 32), ("architectures", ["OtherModel"])):
                bad = dict(config)
                bad[key] = bad_value
                with self.assertRaises(ValueError):
                    validate_checkpoint_config(bad, role)

    def test_recursive_outputs_keep_paths_and_tensor_identity(self):
        first, second = Tensor([1, 2]), Tensor([[3]])
        output = ({"hidden": first, "optional": None}, [second, 7])
        leaves = list(tensor_leaves(output, Tensor))
        self.assertEqual([item[0] for item in leaves], [(("index", 0), ("key", "hidden")),
                                                       (("index", 1), ("index", 0))])
        self.assertIs(leaves[0][1], first)
        self.assertIs(leaves[1][1], second)
        with self.assertRaises(TypeError):
            list(tensor_leaves({3: first}, Tensor))
        with self.assertRaises(TypeError):
            list(tensor_leaves(object(), Tensor))

    def test_public_hook_observes_only_infer_without_replacing_output(self):
        record = Record()
        trace = ForwardTrace(frontend(), None, record)
        module = Linear()
        output = {"x": (Tensor([1, 2]), Tensor([3]))}
        record.report["stages"][-1]["name"] = "reset"
        self.assertIsNone(trace(module, (), output))
        self.assertEqual(record.report["forward_trace"]["events"], [])
        record.report["stages"][-1]["name"] = "infer"
        self.assertIsNone(trace(object(), (), output))
        self.assertIsNone(trace(module, (), output))
        self.assertIsNone(trace(module, (), output))
        self.assertIsNone(trace(Qwen3Attention(), (), output))
        events = record.report["forward_trace"]["events"]
        self.assertEqual([event["call_index"] for event in events], [0, 1, 2])
        self.assertEqual([event["class_call_index"] for event in events], [0, 1, 0])
        self.assertEqual([event["module_instance_index"] for event in events], [0, 0, 0])
        self.assertEqual(len(record.arrays), 6)
        self.assertEqual(events[0]["leaves"][0]["path"], [["key", "x"], ["index", 0]])
        np.testing.assert_array_equal(output["x"][0].array, [1, 2])

    def test_wrong_device_nonfinite_and_trace_bound_fail(self):
        for output in (Tensor([1], device="cpu"), Tensor([float("nan")]), Tensor(np.ones(4097))):
            record = Record()
            with self.assertRaises((AssertionError, RuntimeError)):
                ForwardTrace(frontend(), None, record)(Linear(), (), output)
        record = Record()
        record.report["forward_trace"]["events"] = [{}] * MAX_CALLS
        with self.assertRaises(RuntimeError):
            ForwardTrace(frontend(), None, record)(Linear(), (), Tensor([1]))

    def test_hook_cleanup_success_failure_and_missing_api(self):
        for fail in (False, True):
            removed, registered = [], []

            def register(callback):
                registered.append(callback)
                return types.SimpleNamespace(remove=lambda: removed.append(True))

            callback = ForwardTrace(frontend(), None, Record())
            module = Linear()
            module.register_forward_hook = register
            root = types.SimpleNamespace(named_modules=lambda: [("_fsdp_wrapped_module.model.proj", module)])
            try:
                with ExitStack() as stack:
                    install_trace_hook(callback, stack, root)
                    self.assertEqual(registered, [callback])
                    self.assertEqual(callback.module_paths[id(module)], "model.proj")
                    self.assertEqual(removed, [])
                    if fail:
                        raise ValueError("synthetic body failure")
            except ValueError:
                self.assertTrue(fail)
            self.assertEqual(removed, [True])
        with ExitStack() as stack, self.assertRaises(NotImplementedError):
            install_trace_hook(ForwardTrace(frontend(), None, Record()), stack,
                               types.SimpleNamespace(named_modules=lambda: [("proj", Linear())]))

    def test_recorder_installs_only_after_real_init_and_reset_stages(self):
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            args = parse_args(["--framework", "torch", "--checkpoint", directory,
                               "--output", str(Path(directory) / "trace.json")])
            record = TraceRecorder(args)
            installed, removed = [], []
            module = Linear()
            module.register_forward_hook = lambda callback: (
                installed.append(callback) or types.SimpleNamespace(remove=lambda: removed.append(True)))
            root = types.SimpleNamespace(named_modules=lambda: [("proj", module)])
            worker = types.SimpleNamespace(engine=types.SimpleNamespace(module=root))
            record.trace, record.hook_stack = ForwardTrace(frontend(), None, record), stack
            record.stage("init", lambda: (worker, object()))
            self.assertEqual(installed, [])

            def reset():
                for key in ("model_spec", "parameter_count", "parameter_names"):
                    record.report[key] = manifest()[key]
                self.assertEqual(installed, [])
                return {"parameter": "reset result"}

            self.assertEqual(record.stage("reset", reset), {"parameter": "reset result"})
            self.assertEqual(installed, [record.trace])
            self.assertEqual(removed, [])
        self.assertEqual(removed, [True])


if __name__ == "__main__":
    unittest.main()
