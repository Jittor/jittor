"""Diagnostic public-forward-hook trace of real tiny critic/actor worker infer.

Uses the unchanged probe runtime/run_worker and the same checkpoint/configuration.
No forward/math replacement or private upstream patch is installed here. The
candidate still uses its normal installed Jittor compat/adapters. Hooks observe
every selected module output recursively and return None (never replace output).
Snapshots synchronize and copy observations to the host, so this is NOT a
performance measurement or an acceptance artifact. The oracle's CPU fallback
instrumentation remains unknown; candidate fallback remains forbidden.

Only the controlled 3153-parameter critic / 3648-parameter actor are allowed,
with stop_after=infer. Our Recorder subclass registers public instance hooks
after reset, using named_modules; it does not replace any upstream function.
Events are ordered by hook completion, with a global and per-class call index;
NPZ keys use class/call indices and tensor leaf indices, not framework namespaces.
Compare matching event/leaf manifests before comparing their complete arrays.
"""

import argparse
from collections.abc import Mapping
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import traceback

import numpy as np

from compare_training_worker import MODEL_SPEC
from actor_worker_fixture import MODEL_SPEC as ACTOR_MODEL_SPEC, SCRIPT_NAMES as ACTOR_SCRIPTS, training_config
from probe_training_worker import Recorder, checkpoint_manifest, runtime, run_worker


SCHEMA = "verl-worker-forward-trace-v2"
MAX_CALLS = 128
MAX_LEAF_ELEMENTS = 4096


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", required=True, choices=("torch", "jittor"))
    parser.add_argument("--role", choices=("critic", "actor"), default="critic")
    parser.add_argument("--backend", choices=("hccl",), default="hccl")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--stop-after", choices=("infer",), default="infer")
    parser.set_defaults(prepare_checkpoint=False, capture_gradients=False, steps=3)
    args = parser.parse_args(argv)
    args.capture_gradients = args.role == "actor"  # Match the unchanged actor probe's configuration.
    if not args.checkpoint.is_dir():
        parser.error("--checkpoint must be the existing shared tiny critic directory")
    if args.output.suffix != ".json" or args.timeout < 1:
        parser.error("--output must end in .json and timeout must be positive")
    return args


def require_tiny_manifest(report):
    role = report.get("trace_role", "critic")
    if role not in ("critic", "actor"):
        raise ValueError("unknown trace role")
    spec, count, tensors = (MODEL_SPEC, 3153, 15) if role == "critic" else (ACTOR_MODEL_SPEC, 3648, 14)
    if report.get("model_spec") != spec or report.get("parameter_count") != count:
        raise ValueError("forward trace is restricted to the controlled tiny Qwen3 worker")
    if len(report.get("parameter_names", ())) != tensors:
        raise ValueError("forward trace requires every controlled parameter")


def validate_checkpoint_config(config, role):
    spec = MODEL_SPEC if role == "critic" else ACTOR_MODEL_SPEC
    observed = {key: config.get(key) for key in spec}
    if role == "critic" and observed["num_labels"] is None:
        observed["num_labels"] = len(config.get("id2label", {}))
    architecture = "Qwen3ForTokenClassification" if role == "critic" else "Qwen3ForCausalLM"
    if observed != spec or config.get("architectures") != [architecture]:
        raise ValueError("checkpoint config is outside the controlled tiny worker scope")


def tensor_leaves(value, tensor_type, path=()):
    """Yield complete tensor leaves; retain tuple/list indices and mapping keys."""
    if isinstance(value, tensor_type):
        yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError("diagnostic output mapping keys must be strings")
            yield from tensor_leaves(child, tensor_type, path + (("key", key),))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            yield from tensor_leaves(child, tensor_type, path + (("index", index),))
    elif value is not None and not isinstance(value, (bool, int, float, str)):
        raise TypeError("unsupported diagnostic output container: " + type(value).__name__)


class ForwardTrace:
    def __init__(self, torch, jt, record):
        self.torch, self.jt, self.record = torch, jt, record
        self.class_calls = {}
        self.instances = {}
        self.module_paths = {}

    def selected(self, module):
        return (type(module).__name__.startswith("Qwen3")
                or isinstance(module, (self.torch.nn.Embedding, self.torch.nn.Linear)))

    def __call__(self, module, inputs, output):
        stages = self.record.report["stages"]
        if not stages or stages[-1]["name"] != "infer" or stages[-1]["status"] != "running":
            return None
        class_name = type(module).__name__
        if not self.selected(module):
            return None
        # Check the actual reset result, not only a command-line model claim.
        require_tiny_manifest(self.record.report)
        events = self.record.report["forward_trace"]["events"]
        if len(events) >= MAX_CALLS:
            raise RuntimeError("tiny forward trace call bound exceeded")
        call_index = len(events)
        class_call = self.class_calls.get(class_name, 0)
        self.class_calls[class_name] = class_call + 1
        instances = self.instances.setdefault(class_name, {})
        instance_index = instances.setdefault(id(module), len(instances))
        event = {"call_index": call_index, "class_name": class_name,
                 "module_path": self.module_paths.get(id(module)),
                 "source_class": type(module).__module__ + "." + class_name,
                 "class_call_index": class_call, "module_instance_index": instance_index,
                 "leaves": []}
        events.append(event)
        for index, (path, tensor) in enumerate(tensor_leaves(output, self.torch.Tensor)):
            if int(tensor.numel()) > MAX_LEAF_ELEMENTS:
                raise RuntimeError("tiny forward trace tensor bound exceeded")
            expected_index = int(os.environ.get("LOCAL_RANK", "0"))
            if self.jt is not None:
                native = getattr(tensor, "_var", tensor)
                native.sync()
                if native.location() != "device" or native.device_id != expected_index:
                    raise AssertionError("trace tensor is not on the requested NPU")
            elif tensor.device.type != "npu" or tensor.device.index != expected_index:
                raise AssertionError("oracle trace tensor is not on the requested NPU")
            key = "trace/%03d/%s/class_call_%03d/tensor_%03d" % (call_index, class_name, class_call, index)
            array = self.record.tensor(key, tensor)
            if not np.isfinite(array).all():
                raise AssertionError("nonfinite diagnostic output: " + key)
            event["leaves"].append({"path": [list(part) for part in path], "array_key": key,
                                    "shape": list(array.shape), "numpy_dtype": str(array.dtype),
                                    "tensor_dtype": str(tensor.dtype), "device": str(tensor.device),
                                    "requires_grad": bool(tensor.requires_grad)})
        if not event["leaves"]:
            raise AssertionError("selected forward module produced no observable tensor: " + class_name)
        self.record.flush()
        return None


class TraceRecorder(Recorder):
    def __init__(self, args):
        super().__init__(args)
        self.trace = self.hook_stack = self.worker = None
        self.report["trace_role"] = args.role
        if args.role == "actor":
            self.report["training_config"] = training_config(args.steps)
            self.report["script_sha256"] = {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ACTOR_SCRIPTS
            }
        self.report["schema"] = SCHEMA
        self.report["scope"] = "diagnostic tiny %s module outputs during real worker infer only" % args.role
        self.report["diagnostic_only"] = True
        self.report["forward_trace"] = {
            "hook_api": "torch.nn.Module.register_forward_hook",
            "active_stage": "infer", "order": "forward-hook completion",
            "selection": "Qwen3 class names and nn.Embedding/nn.Linear instances",
            "output_mutation": False, "host_snapshots_synchronize": True,
            "max_calls": MAX_CALLS, "max_leaf_elements": MAX_LEAF_ELEMENTS, "events": [],
        }
        self.report["trace_script_sha256"] = {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("trace_worker_forward.py", "compare_training_worker.py", "actor_worker_fixture.py")
        }
        self.report["limitations"].extend([
            "diagnostic hooks synchronize and alter timing", "not an acceptance artifact",
            "class/call sequence must match before numerical trace comparison",
        ])

    def stage(self, name, function):
        if name == "infer":
            require_tiny_manifest(self.report)
        result = super().stage(name, function)
        if name == "init":
            self.worker, _ = result
        elif name == "reset":
            require_tiny_manifest(self.report)
            if self.trace is None or self.hook_stack is None or self.worker is None:
                raise RuntimeError("diagnostic hook lifecycle not initialized")
            install_trace_hook(self.trace, self.hook_stack, self.worker.engine.module)
        return result


def install_trace_hook(trace, stack, root):
    paths = set()
    for path, module in root.named_modules():
        if not trace.selected(module):
            continue
        normalized = ".".join(part for part in path.split(".") if part != "_fsdp_wrapped_module")
        if normalized in paths:
            raise ValueError("module-path normalization collision")
        paths.add(normalized)
        register = getattr(module, "register_forward_hook", None)
        if not callable(register):
            raise NotImplementedError("public instance register_forward_hook is required")
        trace.module_paths[id(module)] = normalized
        handle = register(trace)
        stack.callback(handle.remove)
    if not paths:
        raise ValueError("no selected Qwen3/Embedding/Linear modules to trace")


def main(argv=None):
    args = parse_args(argv)
    record = TraceRecorder(args)
    try:
        validate_checkpoint_config(json.loads((args.checkpoint / "config.json").read_text()), args.role)
        with ExitStack() as stack:
            torch, jt = record.stage("runtime", lambda: runtime(args, stack, record))
            record.trace, record.hook_stack = ForwardTrace(torch, jt, record), stack
            record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
            if args.role == "critic":
                run_worker(args, stack, record, torch, jt)
            else:
                from probe_actor_worker import run_worker as run_actor
                run_actor(args, record, torch, jt)
            classes = {event["class_name"] for event in record.report["forward_trace"]["events"]}
            model_class = "Qwen3ForTokenClassification" if args.role == "critic" else "Qwen3ForCausalLM"
            if not {"Qwen3Attention", "Qwen3Model", model_class, "Embedding", "Linear"} <= classes:
                raise AssertionError("incomplete public module hook trace: " + repr(sorted(classes)))
            if jt is not None:
                jt.sync_all(True)
                count = int(jt.core.backend_fallback_count()) - record.report["fallback_before"]
                record.report["fallback_count"] = count
                assert count == 0, "backend fallback occurred"
        record.report["hook_removed"] = True
        record.report["status"] = "diagnostic_completed"
        record.flush()
    except Exception as error:
        record.report["status"] = "failed"
        record.report["error_type"] = type(error).__name__
        record.report["error"] = str(error)
        record.report["traceback"] = traceback.format_exc()
        if record.report["stages"] and record.report["stages"][-1]["status"] == "running":
            record.report["stages"][-1]["status"] = "failed"
        record.flush()
        raise


if __name__ == "__main__":
    main()
