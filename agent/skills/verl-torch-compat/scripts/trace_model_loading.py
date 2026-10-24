"""CPU-only tiny Qwen3 loading-lifecycle diagnostic, never training acceptance.

Observe constructor -> strict load_state_dict and independent AutoModel
from_pretrained using the same local config and safetensors. Snapshot every
named parameter/buffer (including nonpersistent RoPE buffers), actual placement,
and read-only _is_hf_initialized markers. No forward, RoPE reconstruction,
private mutation, monkeypatch, worker, NPU, or model download is performed.
Installed Jittor compatibility/adapters remain enabled for the candidate.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import traceback

import numpy as np

import actor_worker_fixture as actor
from compare_training_worker import parameter_shapes
from probe_training_worker import checkpoint_manifest
from trace_worker_forward import validate_checkpoint_config
from worker_loading_provenance import observe_loading


def marker(value):
    """Observe, never set, the HF initialization marker on a module or tensor."""
    flag = getattr(value, "_is_hf_initialized", None)
    return {"present": hasattr(value, "_is_hf_initialized"), "value": flag if flag is None or type(flag) is bool else repr(flag)}


def cpu_runtime(framework, stack, report):
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DEACTIVATE_ASYNC_LOAD="1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    jt = None
    if framework == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
        import jittor as jt
        import torch
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner
        from jittor._runtime.fallback import forbid_backend_fallbacks

        assert namespace_owner(torch) is jt and activation_status(jt).active
        assert torch.Tensor is not jt.Var
        stack.enter_context(jt.runtime.scope(use_cuda=0, backend_fallback="error"))
        stack.enter_context(forbid_backend_fallbacks())
        report["fallback_before"] = int(jt.core.backend_fallback_count())
    else:
        os.environ.pop("JITTOR_TORCH_SHIM", None)
        import torch

        assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
    report["torch_api_version"] = str(getattr(torch, "__torch_version__", torch.__version__))
    return torch, jt


class Diagnostic:
    def __init__(self, args):
        self.path, self.role = args.output, args.role
        self.arrays = {}
        self.report = {
            "schema": "tiny-qwen3-loading-lifecycle-v1", "diagnostic_only": True,
            "scope": "CPU constructor versus from_pretrained initialization only; not training or NPU acceptance",
            "framework": args.framework, "role": args.role, "device": "cpu", "attention": "eager",
            "seed": 20260924, "status": "running", "stages": [], "snapshots": {},
            "harness_private_patches": False, "async_weight_loading": False,
            "framework_layer": "native_torch" if args.framework == "torch" else "jittor.compat.torch and installed adapters",
            "script_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in (Path(__file__).name, "trace_worker_forward.py", "probe_training_worker.py",
                                           "compare_training_worker.py", "actor_worker_fixture.py",
                                           "worker_loading_provenance.py")},
        }

    def flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        if self.arrays:
            np.savez(self.path.with_suffix(".npz"), **self.arrays)

    def stage(self, name, function):
        stage = {"name": name, "status": "running"}
        self.report["stages"].append(stage)
        self.flush()
        result = function()
        stage["status"] = "passed"
        self.flush()
        return result

    def snapshot(self, tag, model, jt):
        expected = parameter_shapes() if self.role == "critic" else actor.parameter_shapes()
        parameters, buffers = dict(model.named_parameters()), dict(model.named_buffers())
        if set(parameters) != set(expected) or any(tuple(p.shape) != expected[name] for name, p in parameters.items()):
            raise ValueError("actual model parameters outside controlled tiny architecture")
        if "model.rotary_emb.inv_freq" not in buffers:
            raise ValueError("required real nonpersistent rotary inv_freq buffer is missing")
        state_names = set(model.state_dict())
        snapshot = {"parameters": {}, "buffers": {}, "modules": {}}
        self.report["snapshots"][tag] = snapshot
        for name, module in model.named_modules():
            snapshot["modules"][name] = {"class": type(module).__name__, "initialization": marker(module)}
        for kind, tensors in (("parameters", parameters), ("buffers", buffers)):
            for name, tensor in tensors.items():
                if int(tensor.numel()) > 4096:
                    raise ValueError("diagnostic tensor exceeds tiny bound")
                info = {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                        "device": str(tensor.device), "in_state_dict": name in state_names,
                        "initialization": marker(tensor)}
                snapshot[kind][name] = info
                if info["device"] != "cpu":
                    raise AssertionError("loading diagnostic tensor is not CPU: " + name)
                if jt is not None:
                    native = getattr(tensor, "_var", tensor)
                    native.sync()
                    info["actual_location"] = native.location()
                    if info["actual_location"] != "cpu":
                        raise AssertionError("candidate actual location is not CPU: " + name)
                else:
                    info["actual_location"] = "cpu"
                key = tag + "/" + kind + "/" + name
                value = np.array(tensor.detach().cpu().numpy(), copy=True)
                info.update(array_key=key, numpy_dtype=str(value.dtype), finite=bool(np.isfinite(value).all()))
                self.arrays[key] = value
                # Nonfinite values are retained as diagnostic evidence, not silently
                # converted to success or discarded before the other path is observed.
        self.flush()

    def compare_paths(self):
        result = {}
        before, after = "constructor_loaded", "from_pretrained"
        for kind in ("parameters", "buffers"):
            left = self.report["snapshots"][before][kind]
            right = self.report["snapshots"][after][kind]
            entry = {"same_names": set(left) == set(right), "missing_from_pretrained": sorted(set(left) - set(right)),
                     "unexpected_from_pretrained": sorted(set(right) - set(left)), "tensors": {}}
            for name in sorted(set(left) & set(right)):
                a, b = self.arrays[left[name]["array_key"]], self.arrays[right[name]["array_key"]]
                compatible = a.shape == b.shape and a.dtype == b.dtype
                finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
                entry["tensors"][name] = {"shape_dtype_equal": compatible, "finite": finite,
                    "exact_equal": compatible and bool(np.array_equal(a, b)),
                    "max_abs_error": float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if compatible and finite else None}
            result[kind] = entry
        self.report["loading_path_comparison"] = result


def run(args, stack, record):
    config_json = json.loads((args.checkpoint / "config.json").read_text())
    validate_checkpoint_config(config_json, args.role)
    record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
    weights = sorted(args.checkpoint.glob("*.safetensors"))
    if len(weights) != 1 or weights[0].stat().st_size > 1024 * 1024:
        raise ValueError("diagnostic requires one tiny safetensors checkpoint under 1 MiB")
    torch, jt = record.stage("runtime", lambda: cpu_runtime(args.framework, stack, record.report))
    from safetensors.torch import load_file
    from transformers import (AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification,
                              Qwen3ForCausalLM, Qwen3ForTokenClassification)

    record.report["loading_provenance"] = observe_loading(args.framework)
    record.report["versions"] = {name: importlib.metadata.version(name) for name in ("transformers", "safetensors")}
    constructor = Qwen3ForTokenClassification if args.role == "critic" else Qwen3ForCausalLM
    automatic = AutoModelForTokenClassification if args.role == "critic" else AutoModelForCausalLM
    def config():
        return AutoConfig.from_pretrained(args.checkpoint, local_files_only=True, attn_implementation="eager")
    with torch.device("cpu"):
        torch.manual_seed(record.report["seed"])
        model_config = config()
        record.report["config"] = model_config.to_dict()
        direct = record.stage("constructor", lambda: constructor(model_config))
        record.stage("snapshot_constructor", lambda: record.snapshot("constructor_initial", direct, jt))
        state = record.stage("load_safetensors", lambda: load_file(str(weights[0]), device="cpu"))
        loaded = record.stage("load_state_dict", lambda: direct.load_state_dict(state, strict=True))
        record.report["load_state_dict"] = {"missing_keys": list(loaded.missing_keys), "unexpected_keys": list(loaded.unexpected_keys)}
        if loaded.missing_keys or loaded.unexpected_keys:
            raise AssertionError("strict load unexpectedly returned unmatched keys")
        record.stage("snapshot_constructor_loaded", lambda: record.snapshot("constructor_loaded", direct, jt))
        torch.manual_seed(record.report["seed"])
        pretrained = record.stage("from_pretrained", lambda: automatic.from_pretrained(
            args.checkpoint, config=config(), local_files_only=True, dtype=torch.float32,
            device_map=None, attn_implementation="eager"))
        record.stage("snapshot_from_pretrained", lambda: record.snapshot("from_pretrained", pretrained, jt))
    record.compare_paths()
    if jt is not None:
        jt.sync_all(True)
        record.report["fallback_count"] = int(jt.core.backend_fallback_count()) - record.report["fallback_before"]
        if record.report["fallback_count"] != 0:
            raise AssertionError("candidate fallback was observed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--role", choices=("critic", "actor"), default="critic")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.suffix != ".json" or not args.checkpoint.is_dir():
        parser.error("existing checkpoint directory and .json output required")
    if args.checkpoint.resolve() in args.output.resolve().parents:
        parser.error("diagnostic output must not modify the checkpoint directory")
    record = Diagnostic(args)
    try:
        with ExitStack() as stack:
            run(args, stack, record)
        record.report["status"] = "diagnostic_completed"
        record.flush()
    except Exception as exc:
        record.report.update(status="failed", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        for stage in record.report["stages"]:
            if stage["status"] == "running":
                stage["status"] = "failed"
        record.flush()
        raise


if __name__ == "__main__":
    main()
