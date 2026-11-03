"""Observe official TRL SFTTrainer on a local, full pretrained Qwen3 checkpoint.

Run this file in independent Torch and Jittor interpreters, then use ``compare``.
All gradients and updated parameters are saved one tensor at a time, avoiding
an in-memory copy of the whole training trajectory. Observations add host I/O;
these timings are diagnostic and are not a throughput benchmark. This probe
does not itself grant registry coverage or upstream source/wheel provenance.
"""

import argparse
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


TEXTS = (
    "The capital of France is Paris. Paris is a city in France.",
    "Two plus three equals five. Five minus two equals three.",
    "Water freezes at zero degrees Celsius under ordinary atmospheric pressure.",
)
PINS = {"trl": "0.21.0", "transformers": "4.56.2", "accelerate": "1.15.0"}
TRL_REVISION = "46d09bd2408f17605409fb3ee8ba12705add7faa"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def bind_trl_source(module, upstream):
    """Bind installed package bytes to the pinned clean official checkout.

    Every installed payload file must be an identical source file, and every
    tracked Python source must actually be installed. This binds package bytes,
    not a wheel archive; no wheel hash or complete scorecard is implied.
    """
    package = Path(module.__file__).resolve().parent
    files = {path.relative_to(package).as_posix(): digest(path)
             for path in sorted(package.rglob("*"))
             if path.is_file() and "__pycache__" not in path.relative_to(package).parts
             and path.suffix not in (".pyc", ".pyo")}
    result = {"version": str(module.__version__), "installed_files_sha256": files,
              "source_verified": False}
    if upstream is None:
        return result
    upstream = Path(upstream).resolve()
    def git(*arguments):
        return subprocess.check_output(["git", "-C", str(upstream), *arguments], text=True).strip()
    revision, remote = git("rev-parse", "HEAD"), git("remote", "get-url", "origin")
    if revision != TRL_REVISION:
        raise ValueError("TRL upstream must be pinned at " + TRL_REVISION)
    if remote not in ("https://github.com/huggingface/trl", "https://github.com/huggingface/trl.git",
                      "git@github.com:huggingface/trl.git"):
        raise ValueError("TRL upstream origin is not the official repository")
    if git("status", "--porcelain"):
        raise ValueError("TRL upstream checkout is dirty")
    for relative, expected in files.items():
        original = upstream / "trl" / relative
        if not original.is_file() or digest(original) != expected:
            raise ValueError("Installed TRL differs from pinned source: " + relative)
    source_python = {name[len("trl/"):] for name in git("ls-files", "trl").splitlines()
                     if name.endswith(".py")}
    if source_python != {name for name in files if name.endswith(".py")}:
        raise ValueError("Installed TRL Python file inventory differs from pinned source")
    result.update(source_verified=True, upstream={"remote": remote, "revision": revision})
    return result


def synchronize_initial_buffers(torch, model, path, runtime):
    """Load one oracle-created buffer artifact into both complete models.

    Checkpoints may omit regenerated nonpersistent buffers, such as RoPE
    frequencies. Their framework-dependent initialization is not a training
    comparison: give both runtimes identical bytes before observing training.
    """
    import numpy as np
    path = Path(path)
    buffers = dict(model.named_buffers())
    if not path.exists():
        if runtime != "torch":
            raise ValueError("Jittor requires an existing oracle initial-buffer artifact")
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: value.detach().cpu().numpy() for name, value in buffers.items()}
        with path.open("xb") as stream:
            np.savez(stream, **arrays)
        del arrays
    artifact_hash = digest(path)
    with np.load(path, allow_pickle=False) as arrays:
        if len(arrays.files) != len(set(arrays.files)) or set(arrays.files) != set(buffers):
            raise ValueError("Initial-buffer artifact names differ from model buffers")
        # Validate the entire artifact before changing any buffer.
        for name, value in buffers.items():
            array = arrays[name]
            if tuple(array.shape) != tuple(value.shape):
                raise ValueError("Initial-buffer shape mismatch: " + name)
            if str(array.dtype) != str(value.dtype).split(".")[-1]:
                raise ValueError("Initial-buffer dtype mismatch: " + name)
            if not np.isfinite(array).all():
                raise ValueError("Non-finite initial buffer: " + name)
        with torch.no_grad():
            for name, value in buffers.items():
                value.copy_(torch.from_numpy(arrays[name]).to(value.device))
    if digest(path) != artifact_hash:
        raise ValueError("Initial-buffer artifact changed while loading")
    return artifact_hash


def options(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--model", type=Path, required=True)
    run.add_argument("--initial-buffers", type=Path, required=True,
                     help="Shared external NPZ: created by the oracle if absent, then loaded by both runtimes")
    run.add_argument("--upstream", type=Path, help="Clean official TRL checkout at the pinned revision")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--steps", type=int, default=3)
    run.add_argument("--max-length", type=int, default=32)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    compare = sub.add_parser("compare")
    compare.add_argument("--reference", type=Path, required=True)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--atol", type=float, default=1e-5)
    compare.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args(argv)
    if args.action == "run":
        args.root, args.model, args.output = (p.resolve() for p in (args.root, args.model, args.output))
        args.initial_buffers = args.initial_buffers.resolve()
        if args.steps < 1 or args.max_length < 2 or args.threads < 1:
            parser.error("steps/threads must be positive and max-length must be >= 2")
        if not (args.root / "compat/tests/torch/_ecosystem_runner.py").is_file():
            parser.error("--root must be the Jittor source checkout")
        if not (args.model / "config.json").is_file():
            parser.error("--model must be a downloaded local checkpoint")
        for path in (args.model, args.output, args.initial_buffers):
            if path == args.root or args.root in path.parents:
                parser.error("model and output artifacts must be outside the checkout")
        if args.initial_buffers == args.model or args.model in args.initial_buffers.parents:
            parser.error("initial-buffer artifact must be outside the immutable checkpoint directory")
    return args


def run(args):
    if args.device == "cpu":
        os.environ.update(CUDA_VISIBLE_DEVICES="", use_cuda="0", JT_USE_CUDA="0",
                          nvcc_path="", JT_BUILD_NVCC_PATH="")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 are required")
    sys.path.insert(0, str(args.root / "compat/tests/torch"))
    import _ecosystem_runner as runner
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        precision = runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        import numpy as np
        from datasets import Dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
        from trl import SFTConfig, SFTTrainer
        import trl

        versions = {name: importlib.metadata.version(name) for name in PINS}
        if versions != PINS:
            raise ValueError("Unexpected dependency versions: " + repr(versions))
        source_binding = bind_trl_source(trl, args.upstream)
        before = None
        jt = None
        if args.runtime == "jittor":
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            if Path(jt.__file__).resolve() != args.root / "python/jittor/__init__.py":
                raise ValueError("Loaded Jittor is not the requested checkout")
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            stack.enter_context(forbid_backend_fallbacks())
            before = jt.core.backend_fallback_count()
        binary = getattr(getattr(torch, "_C", None), "__file__", None)
        if args.runtime == "torch" and (not binary or not Path(binary).is_file()):
            raise ValueError("Reference must load the genuine Torch binary extension")
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "arrays").mkdir()
        baseline = subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain"], text=True)
        model_artifacts = {p.relative_to(args.model).as_posix(): digest(p)
                           for p in sorted(args.model.rglob("*"))
                           if p.is_file() and ".cache" not in p.relative_to(args.model).parts
                           and ".git" not in p.relative_to(args.model).parts}
        # A one-record map-style dataset repeats deterministically across
        # epochs, without relying on identical RNGs for the two samplers.
        records = [{"text": TEXTS[args.seed % len(TEXTS)]}]
        protocol = {"steps": args.steps, "max_length": args.max_length,
                    "seed": args.seed, "threads": args.threads, "batch_size": 1,
                    "gradient_accumulation_steps": 1, "dtype": "float32",
                    "optimizer": "SGD", "learning_rate": 1e-4, "momentum": 0,
                    "weight_decay": 0, "scheduler": "constant", "max_grad_norm": 0,
                    "attention": "eager", "gradient_checkpointing": False,
                    "packing": False, "dataset_order": "single_record", "records": records}
        manifest = {"status": "running", "runtime": args.runtime, "device": args.device,
                    "versions": versions, "protocol": protocol, "precision": precision,
                    "trl_source": source_binding,
                    "model_artifacts": model_artifacts, "jittor_commit": baseline,
                    "source_dirty": dirty, "probe_sha256": digest(__file__),
                    "runner_sha256": digest(args.root / "compat/tests/torch/_ecosystem_runner.py"),
                    "interpreter": sys.executable, "torch_origin": str(torch.__file__),
                    "torch_version": str(torch.__version__), "oracle_binary": binary if args.runtime == "torch" else None,
                    "command": [sys.executable, *sys.argv], "arrays": {}, "initial": {},
                    "scorecard_eligible": False}
        manifest_path = args.output / "manifest.json"
        write_json(manifest_path, manifest)

        def snapshot(value, name, initial=False):
            if value is None:
                raise AssertionError("Missing tensor: " + name)
            if value.device.type != args.device:
                raise AssertionError("Wrong tensor device: " + name + " " + str(value.device))
            if jt is not None:
                value.sync()
                if value.location() != ("device" if args.device == "cuda" else "cpu"):
                    raise AssertionError("Wrong physical residency: " + name)
                if jt.core.dispatch_context([value])[0] != args.device:
                    raise AssertionError("Wrong native dispatch: " + name)
            data = value.detach().cpu().numpy()
            flat = data.reshape(-1)
            # Bound temporary memory, including for the large embedding matrix.
            for start in range(0, flat.size, 1 << 20):
                if not np.isfinite(flat[start:start + (1 << 20)]).all():
                    raise AssertionError("Non-finite tensor: " + name)
            info = {"shape": list(data.shape), "dtype": str(data.dtype), "device": args.device}
            if initial:
                checksum = hashlib.sha256()
                for start in range(0, flat.size, 1 << 20):
                    checksum.update(flat[start:start + (1 << 20)].tobytes())
                info["sha256"] = checksum.hexdigest()
                manifest["initial"][name] = info
            else:
                relative = "arrays/" + hashlib.sha256(name.encode()).hexdigest() + ".npy"
                target = args.output / relative
                with target.open("xb") as stream:
                    np.save(stream, data, allow_pickle=False)
                info.update(path=relative, sha256=digest(target))
                manifest["arrays"][name] = info

        torch.manual_seed(args.seed)
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(args.model), local_files_only=True,
                    torch_dtype=torch.float32, attn_implementation="eager")
        if model.config.model_type != "qwen3":
            raise ValueError("This evaluation requires an official Qwen3 checkpoint")
        model.config.use_cache = False
        model.train()
        dataset = Dataset.from_list(records)
        config = SFTConfig(output_dir=str(args.output / "trainer"), use_cpu=args.device == "cpu",
            bf16=False, fp16=False, tf32=False, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, max_steps=args.steps, max_length=args.max_length,
            learning_rate=1e-4, weight_decay=0., optim="sgd", lr_scheduler_type="constant",
            warmup_steps=0, max_grad_norm=0., gradient_checkpointing=False, packing=False,
            dataset_text_field="text", dataloader_num_workers=0, dataloader_pin_memory=False,
            logging_steps=1, disable_tqdm=True, report_to=[], save_strategy="no",
            eval_strategy="no", seed=args.seed, data_seed=args.seed)
        trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)
        if type(trainer) is not SFTTrainer:
            raise AssertionError("Training must use the official SFTTrainer")
        protocol["initial_buffers_sha256"] = synchronize_initial_buffers(
            torch, model, args.initial_buffers, args.runtime)
        parameters = dict(model.named_parameters())
        if not all(p.requires_grad and p.dtype == torch.float32 for p in parameters.values()):
            raise AssertionError("Every parameter must be trainable FP32")
        manifest.update(model_class=type(model).__module__ + "." + type(model).__name__,
                        trainer_class=type(trainer).__module__ + "." + type(trainer).__name__,
                        parameter_count=sum(p.numel() for p in parameters.values()),
                        parameter_names=list(parameters), input_gradient_names=[],
                        intermediate_gradient_names=["embedding_output"])
        for name, value in parameters.items():
            snapshot(value, "parameter::" + name, initial=True)
        for name, value in model.named_buffers():
            snapshot(value, "buffer::" + name, initial=True)
        write_json(manifest_path, manifest)

        class Observer(TrainerCallback):
            def __init__(self):
                self.steps = self.updates = 0
                self.outputs, self.embeddings = [], []
                def embedding_hook(module, inputs, output):
                    output.retain_grad()
                    self.embeddings.append(output)
                def model_hook(module, inputs, kwargs, output):
                    if inputs:
                        raise AssertionError("Unexpected positional model arguments")
                    prefix = "step/%d/" % self.steps
                    for name, value in kwargs.items():
                        if torch.is_tensor(value):
                            snapshot(value, prefix + "input::" + name)
                    self.outputs.append(output)
                self.hooks = [model.get_input_embeddings().register_forward_hook(embedding_hook),
                              model.register_forward_hook(model_hook, with_kwargs=True)]

            def on_pre_optimizer_step(self, training_args, state, control, **kwargs):
                if int(state.global_step) != self.steps or len(self.outputs) != 1 or len(self.embeddings) != 1:
                    raise AssertionError("Unexpected Trainer forward/backward order")
                prefix = "step/%d/" % self.steps
                snapshot(self.outputs[0].logits, prefix + "logits")
                snapshot(self.outputs[0].loss, prefix + "loss")
                snapshot(self.embeddings[0].grad, prefix + "intermediate_grad::embedding_output")
                for name, value in parameters.items():
                    snapshot(value.grad, prefix + "gradient::" + name)
                self.steps += 1
                write_json(manifest_path, manifest)

            def on_step_end(self, training_args, state, control, **kwargs):
                if int(state.global_step) != self.updates + 1 or self.steps != self.updates + 1:
                    raise AssertionError("Unexpected Trainer optimizer order")
                prefix = "step/%d/" % self.updates
                for name, value in parameters.items():
                    snapshot(value, prefix + "parameter::" + name)
                self.outputs.clear()
                self.embeddings.clear()
                self.updates += 1
                write_json(manifest_path, manifest)
                print("TRL_SFT_STEP " + json.dumps({"updates": self.updates}), flush=True)

        observer = Observer()
        trainer.add_callback(observer)
        start = time.monotonic()
        try:
            trainer.train()
        finally:
            for handle in observer.hooks:
                handle.remove()
        runner._synchronize(torch, args.runtime, args.device)
        if observer.steps != args.steps or observer.updates != args.steps or trainer.state.global_step != args.steps:
            raise AssertionError("Trainer did not complete the requested updates")
        fallbacks = jt.core.backend_fallback_count() - before if jt is not None else 0
        if fallbacks:
            raise AssertionError("Backend fallback occurred")
        if bind_trl_source(trl, args.upstream) != source_binding:
            raise AssertionError("TRL installed package/source changed during training")
        manifest.update(status="complete", updates=observer.updates, fallback_count=fallbacks,
                        fallback_policy="error" if jt is not None else None,
                        observed_training_seconds=time.monotonic() - start,
                        trainer_log_history=trainer.state.log_history,
                        runtime_conditions=runner._runtime_conditions(torch, precision),
                        backend=runner._backend_report(args.runtime))
        write_json(manifest_path, manifest)
        print("TRL_SFT_RESULT " + json.dumps({key: manifest[key] for key in
              ("status", "runtime", "device", "updates", "parameter_count", "fallback_count")}), flush=True)


def compare(args):
    import numpy as np
    ref = json.loads((args.reference / "manifest.json").read_text())
    candidate = json.loads((args.candidate / "manifest.json").read_text())
    if ref["runtime"] != "torch" or candidate["runtime"] != "jittor":
        raise AssertionError("Comparison requires independent Torch and Jittor runs")
    for metadata in (ref, candidate):
        if metadata["status"] != "complete" or metadata["fallback_count"] != 0:
            raise AssertionError("Both runs must complete with zero fallback")
    for key in ("device", "versions", "trl_source", "protocol", "precision", "model_artifacts", "initial",
                "parameter_names", "parameter_count", "probe_sha256", "runner_sha256"):
        if ref[key] != candidate[key]:
            raise AssertionError("Runtime inputs/protocol mismatch: " + key)
    for metadata in (ref, candidate):
        required = set()
        for step in range(metadata["protocol"]["steps"]):
            prefix = "step/%d/" % step
            required.update(prefix + name for name in
                            ("logits", "loss", "intermediate_grad::embedding_output", "input::input_ids"))
            for parameter in metadata["parameter_names"]:
                required.update((prefix + "gradient::" + parameter, prefix + "parameter::" + parameter))
        missing = required - set(metadata["arrays"])
        if missing:
            raise AssertionError("Incomplete observations or missing gradients: " + repr(sorted(missing)))
    if set(ref["arrays"]) != set(candidate["arrays"]):
        raise AssertionError("Different observed tensors or missing gradients")
    results = {}
    for name, info in ref["arrays"].items():
        other = candidate["arrays"][name]
        if any(info[key] != other[key] for key in ("shape", "dtype", "device")):
            raise AssertionError("Tensor metadata mismatch: " + name)
        paths = (args.reference / info["path"], args.candidate / other["path"])
        if digest(paths[0]) != info["sha256"] or digest(paths[1]) != other["sha256"]:
            raise AssertionError("Observation changed after measurement: " + name)
        arrays = [np.load(path, mmap_mode="r", allow_pickle=False).reshape(-1) for path in paths]
        field_max = 0.
        for start in range(0, arrays[0].size, 1 << 20):
            part = arrays[0][start:start + (1 << 20)]
            field_max = max(field_max, float(np.max(np.abs(part))) if part.size else 0.)
        max_abs, max_scaled, failures = 0., 0., 0
        exact = info["dtype"].startswith(("int", "uint", "bool")) or "/input::" in name
        for start in range(0, arrays[0].size, 1 << 20):
            left, right = (array[start:start + (1 << 20)].astype(np.float64) for array in arrays)
            difference = np.abs(left - right)
            scale = np.maximum(np.abs(left), max(field_max * 1e-6, 1e-30))
            if difference.size:
                max_abs = max(max_abs, float(difference.max()))
                max_scaled = max(max_scaled, float((difference / scale).max()))
            allowed = 0. if exact else args.atol + args.rtol * scale
            failures += int(np.count_nonzero((difference > allowed) | ~np.isfinite(difference)))
        results[name] = {"passed": failures == 0, "mismatched_elements": failures,
                         "max_abs": max_abs, "max_scaled": max_scaled, "exact": exact}
    report = {"passed": all(row["passed"] for row in results.values()),
              "reference": str(args.reference.resolve()), "candidate": str(args.candidate.resolve()),
              "device": ref["device"], "steps": ref["protocol"]["steps"],
              "atol": args.atol, "rtol": args.rtol, "scale_floor_fraction": 1e-6,
              "tensor_count": len(results), "results": results, "scorecard_eligible": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print("TRL_SFT_COMPARISON " + json.dumps({key: report[key] for key in ("passed", "device", "steps", "tensor_count")}))
    if not report["passed"]:
        raise SystemExit(1)


def main(argv=None):
    args = options(argv)
    if args.action == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
