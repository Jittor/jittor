"""Observe full-checkpoint TRL DPO/GRPO and compare with an independent oracle.

Oracle runs retain every step's parameter gradients and final parameters.
Candidates compare every element while streaming and retain final parameters,
small observations, hashes and error statistics rather than a second gradient
trajectory. This is a numerical experiment, not a performance scorecard.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


def array_hash(array):
    result = hashlib.sha256()
    flat = array.reshape(-1)
    for start in range(0, flat.size, 1 << 20):
        result.update(flat[start:start + (1 << 20)].tobytes())
    return result.hexdigest()


def compare_arrays(reference, candidate, exact=False, initial=None):
    """Bound temporary memory and report both element budgets and L2 error."""
    import numpy as np
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError("Array shape/dtype mismatch")
    if initial is not None:
        if initial.shape != reference.shape or initial.dtype != reference.dtype:
            raise AssertionError("Initial baseline shape/dtype mismatch")
        if not np.isfinite(initial).all():
            raise AssertionError("Non-finite initial baseline")
    left, right = reference.reshape(-1), candidate.reshape(-1)
    maximum = 0.
    for start in range(0, left.size, 1 << 20):
        part = left[start:start + (1 << 20)]
        if part.size:
            maximum = max(maximum, float(np.max(np.abs(part))))
    mismatches, max_abs = 0, 0.
    ref_sq = candidate_sq = diff_sq = dot = update_sq = 0.
    baseline = initial.reshape(-1) if initial is not None else None
    for start in range(0, left.size, 1 << 20):
        a, b = (value[start:start + (1 << 20)].astype(np.float64) for value in (left, right))
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise AssertionError("Non-finite observation")
        delta = b - a
        integer_exact = exact and reference.dtype.kind in "biu"
        if integer_exact:
            raw_a, raw_b = (value[start:start + (1 << 20)] for value in (left, right))
            unequal = raw_a != raw_b
            # Preserve distinctions above 2**53 and avoid signed/unsigned
            # subtraction overflow before converting diagnostic magnitudes.
            delta[unequal] = [int(y) - int(x) for x, y in zip(raw_a[unequal], raw_b[unequal])]
        scale = np.maximum(np.abs(a), max(maximum * 1e-6, 1e-30))
        allowed = 0. if exact else 1e-5 + 1e-4 * scale
        mismatches += int(np.count_nonzero(unequal if integer_exact else np.abs(delta) > allowed))
        max_abs = max(max_abs, float(np.max(np.abs(delta))) if delta.size else 0.)
        ref_sq += float(np.sum(a * a))
        candidate_sq += float(np.sum(b * b))
        diff_sq += float(np.sum(delta * delta))
        dot += float(np.sum(a * b))
        if baseline is not None:
            update = a - baseline[start:start + (1 << 20)].astype(np.float64)
            update_sq += float(np.sum(update * update))
    result = {
        "passed": mismatches == 0, "elements": int(left.size),
        "mismatched_elements": mismatches, "max_abs": max_abs,
        "reference_l2": ref_sq ** .5, "candidate_l2": candidate_sq ** .5,
        "difference_l2": diff_sq ** .5,
        "relative_l2": (diff_sq / ref_sq) ** .5 if ref_sq else (0. if not diff_sq else None),
        "cosine": dot / (ref_sq * candidate_sq) ** .5 if ref_sq and candidate_sq else None,
        "exact": exact,
    }
    if baseline is not None:
        result["reference_update_l2"] = update_sq ** .5
        result["relative_to_update_l2"] = (diff_sq / update_sq) ** .5 if update_sq else None
    return result


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--initial-buffers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--algorithm", choices=("dpo", "grpo"), required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    for key in ("root", "model", "upstream", "initial_buffers", "output", "reference"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    if not 1 <= args.steps <= 10 or args.threads < 1:
        parser.error("Use 1..10 steps and positive threads; ten is the full experiment")
    if (args.runtime == "jittor") != (args.reference is not None):
        parser.error("Only the Jittor candidate requires --reference")
    for path in (args.model, args.output, args.initial_buffers):
        if path == args.root or args.root in path.parents:
            parser.error("Models and mutable artifacts must be outside the checkout")
    return args


def run(args):
    import numpy as np
    if args.device == "cpu":
        os.environ.update(CUDA_VISIBLE_DEVICES="", use_cuda="0", JT_USE_CUDA="0",
                          nvcc_path="", JT_BUILD_NVCC_PATH="")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Offline model loading is required")
    os.environ["JITTOR_ECOSYSTEM_TF32"] = "0"
    sys.path.insert(0, str(args.root / "compat/tests/torch"))
    import _ecosystem_runner as runner
    case = importlib.import_module("trl_" + args.algorithm + "_case")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arrays").mkdir()
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        precision = runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        import trl
        from transformers import AutoTokenizer, TrainerCallback
        versions = {name: importlib.metadata.version(name) for name in PINS}
        if versions != PINS:
            raise ValueError("Unexpected downstream versions")
        source = bind_trl_source(trl, args.upstream)
        binary = getattr(getattr(torch, "_C", None), "__file__", None)
        jt, fallback_before = None, 0
        if args.runtime == "torch" and (not binary or not Path(binary).is_file()):
            raise ValueError("Oracle must load the genuine Torch binary")
        if args.runtime == "jittor":
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            if Path(jt.__file__).resolve() != args.root / "python/jittor/__init__.py":
                raise ValueError("Incorrect Jittor checkout")
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            stack.enter_context(forbid_backend_fallbacks())
            fallback_before = jt.core.backend_fallback_count()
        protocol = {
            "algorithm": args.algorithm, "steps": args.steps, "sample_count": 10,
            "full_experiment": args.steps == 10, "records": case.records(),
            "precision": precision, "dtype": "float32", "optimizer": "SGD",
            "learning_rate": args.learning_rate, "seed": 0, "threads": args.threads,
            "initial_buffers": digest(args.initial_buffers),
            "atol": 1e-5, "rtol": 1e-4, "scale_floor_fraction": 1e-6,
        }
        manifest = {
            "status": "running", "runtime": args.runtime, "device": args.device,
            "versions": versions, "protocol": protocol, "trl_source": source,
            "jittor_commit": subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
            "source_dirty": subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain"], text=True),
            "tool_sources": {Path(path).name: digest(path) for path in
                             (__file__, case.__file__, str(Path(__file__).with_name("trl_sft_probe.py")))},
            "model_artifacts": {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()},
            "torch_origin": str(torch.__file__), "torch_version": str(torch.__version__),
            "oracle_binary": binary if args.runtime == "torch" else None,
            "initial": {}, "arrays": {}, "comparison": {}, "gradient_steps": [],
            "scorecard_eligible": False, "command": [sys.executable, *sys.argv],
            "retention": "oracle full gradients; candidate gradient hashes/statistics; both full final parameters and small observations",
        }
        manifest_path = args.output / "manifest.json"
        reference = None
        if args.reference is not None:
            reference = json.loads((args.reference / "manifest.json").read_text())
            if reference["status"] != "complete" or reference["runtime"] != "torch":
                raise AssertionError("Require a complete independent oracle run")
            for key in ("protocol", "device", "versions", "trl_source", "tool_sources", "model_artifacts"):
                if reference[key] != manifest[key]:
                    raise AssertionError("Oracle protocol mismatch: " + key)
        write_json(manifest_path, manifest)

        def to_array(value, physical=False):
            if torch.is_tensor(value):
                if physical and value.device.type != args.device:
                    raise AssertionError("Wrong observed tensor device")
                if physical and jt is not None:
                    value.sync()
                    if value.location() != ("device" if args.device == "cuda" else "cpu"):
                        raise AssertionError("Wrong physical tensor residency")
                    if jt.core.dispatch_context([value])[0] != args.device:
                        raise AssertionError("Wrong native tensor dispatch")
                value = value.detach().cpu().numpy()
            data = np.asarray(value)
            if data.dtype.kind not in "biuf":
                raise ValueError("Observation must be numeric")
            if not np.isfinite(data).all():
                raise AssertionError("Non-finite observation")
            return data

        def observe(name, value, exact=False, save_candidate=True, initial=None, physical=False):
            if name in manifest["arrays"]:
                raise AssertionError("Duplicate observation: " + name)
            data = to_array(value, physical=physical)
            info = {"shape": list(data.shape), "dtype": str(data.dtype), "sha256": array_hash(data),
                    "exact": exact, "physical_device_checked": physical}
            if reference is None or save_candidate:
                relative = "arrays/" + hashlib.sha256(name.encode()).hexdigest() + ".npy"
                with (args.output / relative).open("xb") as stream:
                    np.save(stream, data, allow_pickle=False)
                info["path"] = relative
            manifest["arrays"][name] = info
            if reference is not None:
                if name not in reference["arrays"]:
                    raise AssertionError("Unexpected candidate observation: " + name)
                expected = reference["arrays"][name]
                if any(info[k] != expected[k] for k in ("shape", "dtype", "exact", "physical_device_checked")):
                    raise AssertionError("Observation metadata mismatch: " + name)
                target = np.load(args.reference / expected["path"], mmap_mode="r", allow_pickle=False)
                if array_hash(target) != expected["sha256"]:
                    raise AssertionError("Oracle observation changed: " + name)
                manifest["comparison"][name] = compare_arrays(target, data, exact, initial)
            return data

        phase, calls = "setup", {}
        trainer = None

        def record(name, value, exact=False):
            prefix = "train/step/%d" % trainer.state.global_step if phase == "train" else phase
            key = prefix + "/" + name
            index = calls.get(key, 0)
            calls[key] = index + 1
            observe(key + "/%d" % index, value, exact=exact)

        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        config = dict(
            output_dir=str(args.output / "trainer"), use_cpu=args.device == "cpu",
            bf16=False, fp16=False, tf32=False, max_steps=args.steps,
            gradient_accumulation_steps=1, learning_rate=args.learning_rate,
            weight_decay=0., optim="sgd", lr_scheduler_type="constant",
            warmup_steps=0, max_grad_norm=0., gradient_checkpointing=False,
            dataloader_num_workers=0, dataloader_pin_memory=False,
            logging_steps=1, disable_tqdm=True, report_to=[], save_strategy="no",
            eval_strategy="no", seed=0, data_seed=0,
        )
        trainer, metadata = case.build(args, torch, tokenizer, config, record)
        manifest["case"] = metadata
        model = trainer.model
        synchronize_initial_buffers(torch, model, args.initial_buffers, args.runtime)
        ref_model = getattr(trainer, "ref_model", None)
        if ref_model is not None:
            synchronize_initial_buffers(torch, ref_model, args.initial_buffers, args.runtime)
        parameters = dict(model.named_parameters())
        if not all(value.requires_grad and value.dtype == torch.float32 for value in parameters.values()):
            raise AssertionError("All policy parameters must be trainable FP32")
        manifest["parameter_count"] = sum(value.numel() for value in parameters.values())
        manifest["parameter_names"] = list(parameters)
        initial_arrays = {}
        for role, current in (("policy", model), ("reference", ref_model)):
            if current is None:
                continue
            for kind, values in (("parameter", current.named_parameters()), ("buffer", current.named_buffers())):
                for name, value in values:
                    data = to_array(value, physical=True)
                    manifest["initial"][role + "/" + kind + "::" + name] = {
                        "shape": list(data.shape), "dtype": str(data.dtype), "sha256": array_hash(data)}
                    if role == "policy" and kind == "parameter":
                        initial_arrays[name] = data.copy()
        if reference is not None:
            for key in ("initial", "parameter_count", "parameter_names"):
                if manifest[key] != reference[key]:
                    raise AssertionError("Different initial policy/reference: " + key)
        write_json(manifest_path, manifest)

        if args.algorithm == "dpo":
            trainer.get_train_dataloader()
            trainer.get_eval_dataloader()
            phase = "eval_before"
            manifest["eval_before"] = trainer.evaluate()

        class Observer(TrainerCallback):
            def on_pre_optimizer_step(self, training_args, state, control, **kwargs):
                count, squared = 0, 0.
                for name, value in parameters.items():
                    if value.grad is None:
                        raise AssertionError("Missing parameter gradient: " + name)
                    data = observe("gradient/step/%d/%s" % (state.global_step, name), value.grad,
                                   save_candidate=False, physical=True)
                    for start in range(0, data.size, 1 << 20):
                        part = data.reshape(-1)[start:start + (1 << 20)].astype(np.float64)
                        squared += float(np.sum(part * part))
                    count += 1
                manifest["gradient_steps"].append({"step": int(state.global_step), "tensors": count,
                                                   "l2": squared ** .5, "nonzero": squared > 0})

            def on_step_end(self, training_args, state, control, **kwargs):
                write_json(manifest_path, manifest)
                print("TRL_RL_STEP", args.algorithm, args.runtime, int(state.global_step), flush=True)

        trainer.add_callback(Observer())
        phase = "train"
        started = time.monotonic()
        trainer.train()
        runner._synchronize(torch, args.runtime, args.device)
        manifest["diagnostic_training_seconds"] = time.monotonic() - started
        manifest["trainer_log_history"] = trainer.state.log_history
        if trainer.state.global_step != args.steps or len(manifest["gradient_steps"]) != args.steps:
            raise AssertionError("Incorrect training update count")
        if not any(row["nonzero"] for row in manifest["gradient_steps"]):
            raise AssertionError("No effective parameter gradient in the entire experiment")
        for name, value in parameters.items():
            observe("final_parameter/" + name, value, initial=initial_arrays[name], physical=True)
        del initial_arrays
        if ref_model is not None:
            for kind, values in (("parameter", ref_model.named_parameters()), ("buffer", ref_model.named_buffers())):
                for name, value in values:
                    if array_hash(to_array(value, physical=True)) != manifest["initial"]["reference/" + kind + "::" + name]["sha256"]:
                        raise AssertionError("Frozen reference changed during training")
        phase = "eval_after"
        if args.algorithm == "dpo":
            manifest["eval_after"] = trainer.evaluate()
        else:
            if metadata["rollout_calls"] != args.steps or len(metadata["reward_groups"]) != args.steps:
                raise AssertionError("Incomplete GRPO rollout/reward coverage")
            if not metadata["nonzero_reward_groups"]:
                raise AssertionError("All GRPO reward groups have zero variance")
            manifest["eval_after"] = case.evaluate_rollouts(trainer, args, torch, record)
        fallback = jt.core.backend_fallback_count() - fallback_before if jt is not None else 0
        if fallback:
            raise AssertionError("Backend fallback occurred")
        if bind_trl_source(trl, args.upstream) != source:
            raise AssertionError("TRL source changed during measurement")
        for name, checksum in manifest["tool_sources"].items():
            if digest(Path(__file__).with_name(name)) != checksum:
                raise AssertionError("Probe source changed during measurement: " + name)
        if reference is not None and set(reference["arrays"]) != set(manifest["arrays"]):
            raise AssertionError("Missing candidate observations")
        manifest.update(status="complete", updates=int(trainer.state.global_step), fallback_count=fallback,
                        numerical_passed=all(row["passed"] for row in manifest["comparison"].values()) if reference is not None else None)
        write_json(manifest_path, manifest)
        print("TRL_RL_RESULT", json.dumps({k: manifest[k] for k in
              ("status", "runtime", "device", "updates", "fallback_count", "numerical_passed")}), flush=True)


if __name__ == "__main__":
    run(options())
