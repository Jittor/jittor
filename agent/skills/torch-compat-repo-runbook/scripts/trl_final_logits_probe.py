"""Compare full-vocabulary logits from each runtime's actually trained weights.

This measures combined training and inference differences. It is neither a
same-weight operator comparison nor sampled-generation quality evaluation.
Every final parameter is verified against its completed training trace before
and after loading; original checkpoint parameters cannot silently remain.
"""

import argparse
from contextlib import ExitStack
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from trl_rl_probe import array_hash, compare_arrays
from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


TRAINING_SHARED = ("protocol", "device", "versions", "trl_source", "tool_sources",
                   "model_artifacts", "initial", "parameter_count", "parameter_names")


def manifest_path(path):
    path = Path(path).resolve()
    return path / "manifest.json" if path.is_dir() else path


def read_training(path, runtime, device):
    path = manifest_path(path)
    trace = json.loads(path.read_text())
    if trace.get("status") != "complete" or trace.get("fallback_count") != 0:
        raise AssertionError("Training trace must be complete with zero fallback")
    if trace.get("runtime") != runtime or trace.get("device") != device:
        raise AssertionError("Training trace runtime/device mismatch")
    if trace.get("updates") != trace["protocol"]["steps"]:
        raise AssertionError("Training trace has incomplete optimizer updates")
    records = trace["protocol"]["records"]
    if len(records) != 10 or len({row["prompt"] for row in records}) != 10:
        raise AssertionError("Require ten distinct fixed training prompts")
    names = trace["parameter_names"]
    if not names or len(names) != len(set(names)):
        raise AssertionError("Invalid training parameter inventory")
    expected = {"final_parameter/" + name for name in names}
    if {name for name in trace["arrays"] if name.startswith("final_parameter/")} != expected:
        raise AssertionError("Incomplete trained parameter artifact inventory")
    return trace


def validate_training_pair(candidate, reference):
    if candidate["runtime"] != "jittor" or reference["runtime"] != "torch":
        raise AssertionError("Require independently trained Jittor and Torch models")
    for key in TRAINING_SHARED:
        if candidate[key] != reference[key]:
            raise AssertionError("Training protocol/provenance mismatch: " + key)


def checked_array(directory, info):
    import numpy as np
    directory = Path(directory).resolve()
    path = (directory / info["path"]).resolve()
    if directory not in path.parents:
        raise AssertionError("Array artifact escapes its trace directory")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if list(array.shape) != info["shape"] or str(array.dtype) != info["dtype"]:
        raise AssertionError("Array artifact shape/dtype mismatch")
    if array_hash(array) != info["sha256"]:
        raise AssertionError("Array artifact hash mismatch")
    if not np.isfinite(array).all():
        raise AssertionError("Non-finite artifact")
    return array


def load_trained_parameters(torch, model, trace, directory, physical_check):
    """Load every unique named parameter and verify bytes after device copying."""
    parameters = dict(model.named_parameters())
    if list(parameters) != trace["parameter_names"]:
        raise AssertionError("Model and trained parameter inventories differ")
    if sum(value.numel() for value in parameters.values()) != trace["parameter_count"]:
        raise AssertionError("Model parameter count differs from training")
    hashes = {}
    with torch.no_grad():
        for name, parameter in parameters.items():
            info = trace["arrays"]["final_parameter/" + name]
            array = checked_array(directory, info)
            if tuple(parameter.shape) != tuple(array.shape) or str(parameter.dtype).split(".")[-1] != str(array.dtype):
                raise AssertionError("Trained parameter shape/dtype mismatch: " + name)
            if info["dtype"] != "float32":
                raise AssertionError("Require all trained parameters in FP32")
            # Writable copy avoids aliasing the read-only mmap. Only one tensor
            # is copied at a time, including the large tied embedding matrix.
            parameter.copy_(torch.from_numpy(array.copy()).to(parameter.device))
            physical_check(parameter, name)
            actual = array_hash(parameter.detach().cpu().numpy())
            if actual != info["sha256"]:
                raise AssertionError("Loaded trained weight bytes differ: " + name)
            hashes[name] = actual
    return hashes


def logit_metrics(reference, candidate):
    """Full-element errors and stable distribution diagnostics for each row."""
    import numpy as np
    if reference.ndim < 1 or reference.shape[-1] < 2 or not reference.size:
        raise AssertionError("Require a nonempty vocabulary axis")
    result = compare_arrays(reference, candidate)
    result["rmse"] = result["difference_l2"] / math.sqrt(result["elements"])
    left = reference.reshape(-1, reference.shape[-1])
    right = candidate.reshape(-1, candidate.shape[-1])
    matches, divergences = 0, []
    for a, b in zip(left, right):
        matches += int(int(a.argmax()) == int(b.argmax()))
        a, b = a.astype(np.float64), b.astype(np.float64)
        a, b = a - a.max(), b - b.max()
        log_a, log_b = a - np.log(np.exp(a).sum()), b - np.log(np.exp(b).sum())
        # This is KL(P_Torch || P_Jittor). Values near zero may include a tiny
        # negative FP64 rounding residual, retained rather than hidden.
        divergences.append(float(np.sum(np.exp(log_a) * (log_a - log_b))))
    result.update(positions=len(left), top1_matches=matches,
                  top1_agreement=matches / len(left),
                  mean_softmax_kl_torch_to_jittor=math.fsum(divergences) / len(divergences),
                  max_softmax_kl_torch_to_jittor=max(divergences))
    return result


def aggregate_logits(rows):
    rows = list(rows)
    difference = math.hypot(*(row["difference_l2"] for row in rows))
    reference = math.hypot(*(row["reference_l2"] for row in rows))
    elements = sum(row["elements"] for row in rows)
    positions = sum(row["positions"] for row in rows)
    if not elements or not positions:
        raise AssertionError("No logit observations")
    return {
        "elements": elements, "positions": positions,
        "max_abs": max(row["max_abs"] for row in rows),
        "rmse": difference / math.sqrt(elements),
        "global_relative_l2": difference / reference if reference else (0. if difference == 0. else None),
        "difference_l2": difference, "reference_l2": reference,
        "failed_fields": sum(not row["passed"] for row in rows),
        "failed_elements": sum(row["mismatched_elements"] for row in rows),
        "top1_matches": sum(row["top1_matches"] for row in rows),
        "top1_agreement": sum(row["top1_matches"] for row in rows) / positions,
        "mean_softmax_kl_torch_to_jittor": math.fsum(
            row["mean_softmax_kl_torch_to_jittor"] * row["positions"] for row in rows) / positions,
        "max_softmax_kl_torch_to_jittor": max(row["max_softmax_kl_torch_to_jittor"] for row in rows),
    }


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "model", "training-trace", "initial-buffers", "upstream", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for name in ("root", "model", "training_trace", "initial_buffers", "upstream", "output", "reference"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.threads < 1 or (args.runtime == "jittor") != (args.reference is not None):
        parser.error("Positive threads; only Jittor requires an independent --reference logit run")
    for path in (args.model, args.training_trace, args.initial_buffers, args.output):
        if path == args.root or args.root in path.parents:
            parser.error("Model and mutable artifacts must be outside the checkout")
    args.training_trace = manifest_path(args.training_trace)
    if args.reference is not None:
        args.reference = manifest_path(args.reference)
    return args


def run(args):
    import numpy as np
    trace = read_training(args.training_trace, args.runtime, args.device)
    trace_hash = digest(args.training_trace)
    if digest(args.initial_buffers) != trace["protocol"]["initial_buffers"]:
        raise AssertionError("Initial buffer artifact differs from actual training")
    if args.device == "cpu":
        os.environ.update(CUDA_VISIBLE_DEVICES="", use_cuda="0", JT_USE_CUDA="0",
                          nvcc_path="", JT_BUILD_NVCC_PATH="")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Offline model loading is required")
    os.environ["JITTOR_ECOSYSTEM_TF32"] = "0"
    sys.path.insert(0, str(args.root / "compat/tests/torch"))
    import _ecosystem_runner as runner
    sources = [Path(__file__), Path(__file__).with_name("trl_sft_probe.py"),
               Path(__file__).with_name("trl_rl_probe.py"), Path(runner.__file__)]
    frozen = {str(path): digest(path) for path in sources}
    reference = None
    if args.reference is not None:
        reference = json.loads(args.reference.read_text())
        if reference.get("status") != "complete" or reference.get("runtime") != "torch" or reference.get("fallback_count") != 0:
            raise AssertionError("Require completed independent Torch logit reference")
        oracle_trace_path = Path(reference["training_trace"]["path"])
        if digest(oracle_trace_path) != reference["training_trace"]["sha256"]:
            raise AssertionError("Oracle training manifest changed")
        oracle_trace = read_training(oracle_trace_path, "torch", args.device)
        validate_training_pair(trace, oracle_trace)
        expected_hashes = {name: oracle_trace["arrays"]["final_parameter/" + name]["sha256"]
                           for name in oracle_trace["parameter_names"]}
        if reference["loaded_parameter_hashes"] != expected_hashes:
            raise AssertionError("Oracle logits did not load its own complete trained weights")
        if reference["tool_sources"] != frozen:
            raise AssertionError("Oracle logit probe source mismatch")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arrays").mkdir()
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        precision = runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        import trl
        from transformers import AutoModelForCausalLM, AutoTokenizer
        versions = {name: importlib.metadata.version(name) for name in PINS}
        if versions != PINS or versions != trace["versions"]:
            raise AssertionError("Downstream version mismatch")
        source = bind_trl_source(trl, args.upstream)
        if source != trace["trl_source"]:
            raise AssertionError("TRL source differs from training")
        artifacts = {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}
        if artifacts != trace["model_artifacts"]:
            raise AssertionError("Original model/tokenizer artifacts differ from training")
        binary = getattr(getattr(torch, "_C", None), "__file__", None)
        if args.runtime == "torch" and (not binary or not Path(binary).is_file()):
            raise AssertionError("Oracle must load genuine Torch binary")
        jt, fallback_before = None, 0
        if args.runtime == "jittor":
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            if Path(jt.__file__).resolve() != args.root / "python/jittor/__init__.py":
                raise AssertionError("Incorrect Jittor checkout")
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            stack.enter_context(forbid_backend_fallbacks())
            fallback_before = jt.core.backend_fallback_count()
        protocol = {"prompts": [row["prompt"] for row in trace["protocol"]["records"]],
                    "dtype": "float32", "attention": "eager", "eval": True,
                    "sampling": False, "use_cache": False, "add_special_tokens": False,
                    "threads": args.threads, "precision": precision,
                    "initial_buffers": digest(args.initial_buffers)}
        if reference is not None and (reference["protocol"] != protocol or reference["device"] != args.device):
            raise AssertionError("Oracle logit protocol/device differs")
        manifest = {"status": "running", "runtime": args.runtime, "device": args.device,
                    "protocol": protocol, "versions": versions, "trl_source": source,
                    "model_artifacts": artifacts, "tool_sources": frozen,
                    "training_trace": {"path": str(args.training_trace), "sha256": trace_hash,
                                       "algorithm": trace["protocol"]["algorithm"], "updates": trace["updates"]},
                    "torch_origin": str(torch.__file__), "torch_version": str(torch.__version__),
                    "oracle_binary": binary if args.runtime == "torch" else None,
                    "jittor_commit": subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
                    "command": [sys.executable, *sys.argv], "results": [],
                    "scope": "Each runtime's trained parameters: combined training and inference differences; ten fixed prompts, no sampling",
                    "scorecard_eligible": False}
        target = args.output / "manifest.json"
        write_json(target, manifest)

        def physical(value, name):
            if value.device.type != args.device:
                raise AssertionError("Wrong tensor device: " + name)
            if jt is not None:
                value.sync()
                if value.location() != ("device" if args.device == "cuda" else "cpu"):
                    raise AssertionError("Wrong physical residency: " + name)
                if jt.core.dispatch_context([value])[0] != args.device:
                    raise AssertionError("Wrong physical dispatch: " + name)

        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                                                    torch_dtype=torch.float32, attn_implementation="eager")
        if model.config.model_type != "qwen3":
            raise AssertionError("Require the complete Qwen3 model")
        model.to(args.device)
        model.eval()
        model.config.use_cache = False
        synchronize_initial_buffers(torch, model, args.initial_buffers, args.runtime)
        manifest["loaded_parameter_hashes"] = load_trained_parameters(
            torch, model, trace, args.training_trace.parent, physical)
        for name, value in model.named_buffers():
            expected = trace["initial"]["policy/buffer::" + name]
            if array_hash(value.detach().cpu().numpy()) != expected["sha256"]:
                raise AssertionError("Loaded buffer differs from training: " + name)
        write_json(target, manifest)

        def save(name, array):
            relative = "arrays/" + name + ".npy"
            with (args.output / relative).open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
            return {"path": relative, "shape": list(array.shape), "dtype": str(array.dtype),
                    "sha256": array_hash(array)}

        with torch.no_grad():
            for index, prompt in enumerate(protocol["prompts"]):
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {name: value.to(args.device) for name, value in inputs.items()}
                for name, value in inputs.items():
                    physical(value, name)
                output = model(**inputs, use_cache=False)
                physical(output.logits, "full_vocabulary_logits")
                logits = output.logits.detach().cpu().numpy()
                ids = inputs["input_ids"].detach().cpu().numpy()
                if logits.shape != (1, ids.shape[1], model.config.vocab_size) or logits.dtype != np.float32 or not np.isfinite(logits).all():
                    raise AssertionError("Unexpected full-vocabulary logits")
                next_logits = logits[0, -1, :]
                top = int(next_logits.argmax())
                row = {"prompt": prompt, "input_ids": save("%02d-input_ids" % index, ids),
                       "logits": save("%02d-logits" % index, logits),
                       "next_logits": save("%02d-next_logits" % index, next_logits),
                       "next_token_id": top, "next_token": tokenizer.convert_ids_to_tokens(top),
                       "next_text": tokenizer.decode([top], skip_special_tokens=False)}
                if reference is not None:
                    oracle = reference["results"][index]
                    if oracle["prompt"] != prompt:
                        raise AssertionError("Oracle prompt order differs")
                    oracle_ids = checked_array(args.reference.parent, oracle["input_ids"])
                    if ids.dtype != oracle_ids.dtype or not np.array_equal(ids, oracle_ids):
                        raise AssertionError("Oracle prompt tokens differ")
                    row["comparison"] = {kind: logit_metrics(
                        checked_array(args.reference.parent, oracle[kind]), value)
                        for kind, value in (("logits", logits), ("next_logits", next_logits))}
                    row["oracle_next_token_id"] = oracle["next_token_id"]
                    row["oracle_next_token"] = oracle["next_token"]
                    row["oracle_next_text"] = oracle["next_text"]
                manifest["results"].append(row)
                write_json(target, manifest)
                print("FINAL_LOGITS_PROMPT", args.runtime, index + 1, "next_token", top, flush=True)
        runner._synchronize(torch, args.runtime, args.device)
        fallback = jt.core.backend_fallback_count() - fallback_before if jt is not None else 0
        if fallback or len(manifest["results"]) != 10:
            raise AssertionError("Incomplete logit coverage or backend fallback")
        if digest(args.training_trace) != trace_hash or any(digest(path) != expected for path, expected in frozen.items()):
            raise AssertionError("Training manifest or probe source changed during measurement")
        if bind_trl_source(trl, args.upstream) != source:
            raise AssertionError("TRL source changed during measurement")
        if reference is not None:
            manifest["comparison"] = {kind: aggregate_logits(row["comparison"][kind] for row in manifest["results"])
                                      for kind in ("logits", "next_logits")}
        manifest.update(status="complete", fallback_count=fallback,
                        fallback_policy="error" if jt is not None else None)
        write_json(target, manifest)
        print("FINAL_LOGITS_COMPLETE", args.runtime, args.device, flush=True)


if __name__ == "__main__":
    run(options())
