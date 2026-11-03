"""Official full Qwen3 PPO with policy/value LoRA and adapter-disabled reference.

This separate FP32 CUDA profile retains every trained gradient, parameter and
Adam state at each actual optimizer call (one CPU prerequisite or ten CUDA calls). It does not replace the
full-parameter PPO capacity result or claim quality/convergence equivalence.
"""

import argparse
import ctypes
from contextlib import ExitStack
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import types

from trl_rl_probe import array_hash, compare_arrays
from trl_sft_probe import PINS, bind_trl_source, digest, synchronize_initial_buffers, write_json


ALGORITHMS = ("ppo",)


def validate_manifest(manifest):
    """Reject partial traces even when every present comparison passes."""
    steps = manifest['protocol']['steps']
    if steps not in (1, 10) or manifest['optimizer_steps'] != steps or len(manifest['trainable_parameters']) != 225:
        raise AssertionError('Require complete calls and all224 LoRA tensors plus value head')
    names = set(manifest['arrays'])
    for step in range(steps):
        required = {'step/%d/loss/0' % step}
        for parameter in manifest['trainable_parameters']:
            for category in ('gradient', 'updated_parameter', 'adam_step', 'exp_avg', 'exp_avg_sq'):
                required.add('step/%d/%s/%s/0' % (step, category, parameter))
        if not required <= names:
            raise AssertionError('Missing full per-step loss/gradient/parameter/Adam inventory')
    if len(manifest['generation_calls']) != (1 if steps == 1 else 5):
        raise AssertionError('Incomplete official PPO rollout batches')
    if any(len(call['draws']) != 4 for call in manifest['generation_calls']):
        raise AssertionError('Require four sampled tokens per rollout')
    for index in range(10):
        if 'step/%d/final_eval/%d/next_logits/0' % (steps, index) not in names:
            raise AssertionError('Missing final fixed-prompt logits')


def options():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "upstream", "output", "model", "initial-buffers"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--runtime", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    for name in ("root", "upstream", "output", "reference", "model", "initial_buffers"):
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.steps not in (1, 10) or args.threads < 1 or (args.device == "cuda" and args.steps != 10):
        parser.error("Use ten updates, or one CPU prerequisite update, and positive threads")
    if (args.runtime == "jittor") != (args.reference is not None):
        parser.error("Only Jittor requires a completed independent --reference")
    if args.output == args.root or args.root in args.output.parents:
        parser.error("Output must be outside the source checkout")
    return args


def run(args):
    import numpy as np
    if args.device == "cpu":
        os.environ.update(CUDA_VISIBLE_DEVICES="", use_cuda="0", JT_USE_CUDA="0",
                          nvcc_path="", JT_BUILD_NVCC_PATH="")
    if any(os.environ.get(name) != "1" for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")):
        raise ValueError("Require offline environments")
    os.environ["JITTOR_ECOSYSTEM_TF32"] = "0"
    sys.path.insert(0, str(args.root / "compat/tests/torch"))
    import _ecosystem_runner as runner
    import trl_ppo_lora_case as case
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "arrays").mkdir()
    (args.output / "initial").mkdir()
    with ExitStack() as stack:
        torch = runner._import_torch(args.runtime)
        runner._select_device(torch, args.runtime, args.device, policy_stack=stack)
        precision = runner._configure_tf32(torch, args.device)
        torch.set_num_threads(args.threads)
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        import trl
        from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM, Qwen3ForSequenceClassification
        import peft
        from peft import LoraConfig, get_peft_model, inject_adapter_in_model
        from trl_lora_probe import package_binding
        versions = {name: importlib.metadata.version(name) for name in (*PINS, "peft", "numpy")}
        if versions != dict(PINS, peft="0.17.1", numpy="1.26.4"):
            raise ValueError("Downstream versions differ from the fixed stack")
        source = bind_trl_source(trl, args.upstream)
        binary = getattr(getattr(torch, "_C", None), "__file__", None)
        jt, fallback_before = None, 0
        if args.runtime == "torch" and (not binary or not Path(binary).is_file()):
            raise AssertionError("Oracle must load genuine Torch binary")
        if args.runtime == "jittor":
            import jittor as jt
            from jittor._runtime.fallback import forbid_backend_fallbacks
            if Path(jt.__file__).resolve() != args.root / "python/jittor/__init__.py":
                raise AssertionError("Incorrect Jittor source checkout")
            stack.enter_context(jt.runtime.scope(backend_fallback="error"))
            stack.enter_context(forbid_backend_fallbacks())
            fallback_before = jt.core.backend_fallback_count()
        sources = [Path(__file__), Path(case.__file__), Path(runner.__file__),
                   Path(__file__).with_name("trl_dpo_case.py"),
                   Path(__file__).with_name("trl_rl_probe.py"), Path(__file__).with_name("trl_sft_probe.py"),
                   Path(__file__).with_name("trl_lora_probe.py")]
        hashes = {path.name: digest(path) for path in sources}
        model_artifacts = {p.name: digest(p) for p in sorted(args.model.iterdir()) if p.is_file()}
        config_dict = json.loads(AutoConfig.from_pretrained(str(args.model), local_files_only=True).to_json_string(use_diff=False))
        protocol = {"algorithm": args.algorithm, "steps": args.steps, "threads": args.threads,
                    "learning_rate": args.learning_rate, "dtype": "float32", "attention": "eager",
                    "config": config_dict, "precision": precision, "seed": 0,
                    "model_artifacts": model_artifacts, "initial_buffers_sha256": digest(args.initial_buffers),
                    "retention": "complete gradients, parameters and Adam state at every call in both runtimes",
                    "optimizer": "AdamW", "weight_decay": .01, "betas": [.9, .999], "epsilon": 1e-8,
                    "lora": {"r": 8, "alpha": 16, "dropout": 0., "targets": ["q_proj", "v_proj"]},
                    "tokenizer": "official local checkpoint tokenizer",
                    "scope": "full pretrained Qwen3-0.6B; policy/value LoRA and trained value head, frozen reward; separate from full-parameter PPO",
                    "generation": "native forward/sampling; candidate sampled token results controlled by oracle"}
        from trl_dpo_case import records as preference_records
        protocol["final_eval_prompts"] = [row["prompt"] for row in preference_records()]
        manifest = {"status": "running", "runtime": args.runtime, "device": args.device,
                    "protocol": protocol, "versions": versions, "trl_source": source,
                    "tool_sources": hashes, "peft_source": package_binding(peft), "initial": {}, "arrays": {}, "comparison": {},
                    "generation_calls": [], "optimizer_steps": 0, "gradient_norms": [],
                    "parameter_updates": [],
                    "scorecard_eligible": False, "torch_origin": str(torch.__file__),
                    "torch_version": str(torch.__version__), "oracle_binary": binary,
                    "jittor_commit": subprocess.check_output(["git", "-C", str(args.root), "rev-parse", "HEAD"], text=True).strip(),
                    "source_dirty": subprocess.check_output(["git", "-C", str(args.root), "status", "--porcelain"], text=True),
                    "command": [sys.executable, *sys.argv]}
        reference = None
        if args.reference is not None:
            reference = json.loads((args.reference / "manifest.json").read_text())
            if reference["status"] != "complete" or reference["runtime"] != "torch" or reference["fallback_count"]:
                raise AssertionError("Require complete zero-fallback genuine Torch oracle")
            for name in ("protocol", "versions", "trl_source", "tool_sources", "device", "peft_source"):
                if manifest[name] != reference[name]:
                    raise AssertionError("Oracle configuration/source mismatch: " + name)
        target = args.output / "manifest.json"
        write_json(target, manifest)

        driver = ctypes.CDLL("libcuda.so.1") if args.device == "cuda" else None

        def physical(value):
            if value.device.type != args.device:
                raise AssertionError("Wrong tensor device: " + str(value.device))
            if args.device == 'cuda':
                pointer = value.device_raw_ptr if jt is not None else value.data_ptr()
                ordinal = ctypes.c_int(-1)
                if driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9, ctypes.c_void_p(pointer)) != 0 or ordinal.value != 0:
                    raise AssertionError("Wrong physical CUDA ordinal")
            if jt is not None:
                value.sync()
                if value.location() != ("device" if args.device == "cuda" else "cpu"):
                    raise AssertionError("Wrong physical tensor residency")
                if jt.core.dispatch_context([value])[0] != args.device:
                    raise AssertionError("Wrong native dispatch")

        def array(value, check=True):
            if torch.is_tensor(value):
                if check:
                    physical(value)
                value = value.detach().cpu().numpy()
            result = np.asarray(value)
            if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
                raise AssertionError("Non-finite or unsupported observed tensor")
            return result

        counters = {}

        def record(name, value, exact=False):
            prefix = "step/%d/%s" % (manifest["optimizer_steps"], name)
            count = counters.get(prefix, 0)
            counters[prefix] = count + 1
            key = prefix + "/%d" % count
            result = array(value)
            relative = "arrays/" + hashlib.sha256(key.encode()).hexdigest() + ".npy"
            retained = True
            if retained:
                np.save(args.output / relative, result, allow_pickle=False)
            info = {"shape": list(result.shape), "dtype": str(result.dtype),
                    "sha256": array_hash(result), "exact": exact, "retained": retained}
            if retained:
                info["path"] = relative
            manifest["arrays"][key] = info
            if reference is not None:
                expected = reference["arrays"].get(key)
                if expected is None or any(expected[k] != info[k] for k in ("shape", "dtype", "exact")):
                    raise AssertionError("Observation inventory/type mismatch: " + key)
                other = np.load(args.reference / expected["path"], mmap_mode="r", allow_pickle=False)
                if array_hash(other) != expected["sha256"]:
                    raise AssertionError("Oracle array changed: " + key)
                manifest["comparison"][key] = compare_arrays(other, result, exact)

        def wrap_generation(model, role):
            original = model.generate

            def generate(*positional, **kwargs):
                if positional:
                    raise AssertionError("Expected keyword generation input")
                call_index = len(manifest["generation_calls"])
                inputs = {name: array(value).tolist() for name, value in kwargs.items() if torch.is_tensor(value)}
                config, _ = model._prepare_generation_config(
                    kwargs.get("generation_config"), use_model_defaults=False,
                    **{name: value for name, value in kwargs.items() if name != "generation_config"})
                config.use_cache = False
                current = {"role": role, "inputs": inputs, "config": config.to_dict(), "draws": []}
                expected = None
                if reference is not None:
                    if call_index >= len(reference["generation_calls"]):
                        raise AssertionError("Extra generation call")
                    expected = reference["generation_calls"][call_index]
                    if any(current[name] != expected[name] for name in ("role", "inputs", "config")):
                        raise AssertionError("Generation prompt/configuration mismatch")
                native_multinomial = torch.multinomial

                def multinomial(probabilities, *sample_args, **sample_kwargs):
                    sampled = native_multinomial(probabilities, *sample_args, **sample_kwargs)
                    draw = len(current["draws"])
                    record("generation/%d/probabilities/%d" % (call_index, draw), probabilities)
                    if expected is not None:
                        if draw >= len(expected["draws"]):
                            raise AssertionError("Extra generation sampling call")
                        spec = expected["draws"][draw]
                        if list(probabilities.shape) != spec["distribution_shape"] or list(sampled.shape) != spec["shape"]:
                            raise AssertionError("Generation sampling shape mismatch")
                        sampled = torch.tensor(spec["tokens"], dtype=sampled.dtype, device=sampled.device)
                    current["draws"].append({"distribution_shape": list(probabilities.shape),
                                             "shape": list(sampled.shape), "tokens": array(sampled).tolist()})
                    return sampled

                torch.multinomial = multinomial
                try:
                    forwarded = dict(kwargs)
                    forwarded.update(generation_config=config, use_model_defaults=False, use_cache=False,
                                     disable_compile=True)
                    result = original(**forwarded)
                finally:
                    torch.multinomial = native_multinomial
                if expected is not None and len(current["draws"]) != len(expected["draws"]):
                    raise AssertionError("Missing generation sampling calls")
                sequences = result if torch.is_tensor(result) else result.sequences
                record("generation/%d/sequences" % call_index, sequences, exact=True)
                current["sequences"] = array(sequences).tolist()
                if expected is not None and current["sequences"] != expected["sequences"]:
                    raise AssertionError("Generated sequence differs after controlled replay")
                if not torch.is_tensor(result) and result.scores is not None:
                    for index, score in enumerate(result.scores):
                        record("generation/%d/scores/%d" % (call_index, index), score)
                manifest["generation_calls"].append(current)
                return result

            model.generate = generate

        class Models:
            def __init__(self):
                self.registry = {}

            def __call__(self, role, kind="causal", copy_from=None):
                cls = Qwen3ForCausalLM if kind == "causal" else Qwen3ForSequenceClassification
                kwargs = {"num_labels": 1} if kind == "sequence_classification" else {}
                value = cls.from_pretrained(str(args.model), local_files_only=True,
                                            torch_dtype=torch.float32, attn_implementation="eager", **kwargs)
                if value.config.num_hidden_layers != 28 or value.config.hidden_size != 1024:
                    raise AssertionError("Require the complete pretrained Qwen3-0.6B")
                synchronize_initial_buffers(torch, value, args.initial_buffers, args.runtime)
                configuration = dict(r=8, lora_alpha=16, lora_dropout=0.,
                                     target_modules=['q_proj', 'v_proj'], bias='none')
                if role == 'policy':
                    value = get_peft_model(value, LoraConfig(task_type='CAUSAL_LM', **configuration))
                elif role == 'value':
                    value = inject_adapter_in_model(LoraConfig(**configuration), value)
                    value.score.requires_grad_(True)
                elif role == 'reward':
                    value.requires_grad_(False)
                else:
                    raise AssertionError('Only three model roles are admitted')
                return self.adopt(role, value, copy_from)

            def adopt(self, role, value, copy_from=None):
                if role in self.registry:
                    raise AssertionError("Duplicate model role: " + role)
                if copy_from is not None and copy_from not in self.registry:
                    raise AssertionError("Unknown original policy role")
                value.config._attn_implementation = "eager"
                value.config.use_cache = False
                value.config.pad_token_id = tokenizer.pad_token_id
                named = {"parameter/" + k: v for k, v in value.named_parameters()}
                named.update({"buffer/" + k: v for k, v in value.named_buffers()})
                extras = {}
                if reference is None:
                    for name, tensor in named.items():
                        if name == "parameter/score.weight":
                            seed = int(hashlib.sha256(role.encode()).hexdigest()[:8], 16)
                            extras[name] = np.random.RandomState(seed).normal(0, .02, tuple(tensor.shape)).astype(np.float32)
                        elif tensor.requires_grad:
                            extras[name] = array(tensor, check=False).copy()
                    artifact = args.output / "initial" / (role + "-extras.npz")
                    np.savez(artifact, **extras)
                    artifact_hash = digest(artifact)
                else:
                    artifact = args.reference / "initial" / (role + "-extras.npz")
                    artifact_hash = digest(artifact)
                    if artifact_hash != reference["initial"][role]["extra_artifact_sha256"]:
                        raise AssertionError("Initial head/teacher artifact changed")
                    with np.load(artifact, allow_pickle=False) as payload:
                        extras = {name: payload[name] for name in payload.files}
                with torch.no_grad():
                    for name, source_value in extras.items():
                        tensor = named[name]
                        tensor.copy_(torch.from_numpy(source_value.copy()).to(tensor.device))
                value.to(args.device)
                named = {"parameter/" + k: v for k, v in value.named_parameters()}
                named.update({"buffer/" + k: v for k, v in value.named_buffers()})
                tensors = {}
                for name, tensor in named.items():
                    data = array(tensor)
                    tensors[name] = {"sha256": array_hash(data), "shape": list(data.shape), "dtype": str(data.dtype)}
                if copy_from is not None and tensors != manifest["initial"][copy_from]["tensors"]:
                    raise AssertionError("Reference policy does not equal original policy")
                manifest["initial"][role] = {"extra_artifact_sha256": artifact_hash, "tensors": tensors,
                                            "parameter_count": sum(p.numel() for p in value.parameters())}
                if reference is not None and manifest["initial"][role] != reference["initial"][role]:
                    raise AssertionError("Initial model bytes differ")
                self.registry[role] = value
                if hasattr(value, "generate"):
                    wrap_generation(value, role)
                return value

        models = Models()
        tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, padding_side="left")
        if tokenizer.pad_token_id is None:
            raise AssertionError("Official tokenizer must supply a padding token")
        args.training_kwargs = dict(output_dir=str(args.output / "trainer"), use_cpu=args.device == "cpu",
            bf16=False, fp16=False, tf32=False, max_steps=args.steps, gradient_accumulation_steps=1,
            learning_rate=args.learning_rate, weight_decay=.01, optim="adamw_torch",
            adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, lr_scheduler_type="constant",
            warmup_steps=0, max_grad_norm=0., gradient_checkpointing=False, dataloader_num_workers=0,
            dataloader_pin_memory=False, logging_steps=1, disable_tqdm=True, report_to=[], save_strategy="no",
            eval_strategy="no", seed=0, data_seed=0)
        trainer, metadata = case.build(args, torch, tokenizer, models, record)
        manifest["case"] = metadata
        trained = {name: value for name, value in trainer.model.named_parameters() if value.requires_grad}
        manifest["trainable_parameters"] = {name: list(value.shape) for name, value in trained.items()}
        if len(trained) != 225 or sum('lora_' in name for name in trained) != 224:
            raise AssertionError('Require224 LoRA tensors and one value score head')
        frozen = {}
        for role, model in models.registry.items():
            for name, value in model.named_parameters():
                if not value.requires_grad:
                    frozen[(role, name)] = array_hash(array(value))
        manifest["frozen_parameters"] = len(frozen)
        frozen_buffers = {(role, name): array_hash(array(value))
                          for role, model in models.registry.items()
                          if not any(parameter.requires_grad for parameter in model.parameters())
                          for name, value in model.named_buffers()}
        if reference is not None:
            for name in ("case", "trainable_parameters", "frozen_parameters"):
                if manifest[name] != reference[name]:
                    raise AssertionError("Training inventory/protocol mismatch: " + name)
            if set(manifest["initial"]) != set(reference["initial"]):
                raise AssertionError("Initial model role inventory mismatch")
        embeddings, hooks = [], []
        for role, model in models.registry.items():
            if not any(value.requires_grad for value in model.parameters()):
                continue

            def embedding_hook(module, inputs, output, role=role):
                del module, inputs
                if output.requires_grad:
                    output.retain_grad()
                    embeddings.append((role, output))

            hooks.append(model.get_input_embeddings().register_forward_hook(embedding_hook))

        def output_hook(module, inputs, kwargs, output):
            del module
            values = output if isinstance(output, tuple) else (output,)
            for index, item in enumerate(values):
                item = item.logits if hasattr(item, "logits") else item
                if torch.is_tensor(item) and item.requires_grad:
                    record("forward/output/%d" % index, item[:, -1, :] if item.ndim == 3 else item)
            for name, value in kwargs.items():
                if torch.is_tensor(value) and any(torch.is_tensor(v) and v.requires_grad or
                      hasattr(v, "logits") and v.logits.requires_grad for v in values):
                    record("forward/input/" + name, value, exact=not value.is_floating_point())
            for index, value in enumerate(inputs):
                if torch.is_tensor(value) and torch.is_grad_enabled():
                    record("forward/positional/%d" % index, value, exact=not value.is_floating_point())

        hooks.append(trainer.model.register_forward_hook(output_hook, with_kwargs=True))
        if trainer.optimizer is None:
            trainer.create_optimizer()
        optimizer = trainer.optimizer
        while hasattr(optimizer, "optimizer"):
            optimizer = optimizer.optimizer
        actual = {id(value) for group in optimizer.param_groups for value in group["params"]}
        if actual != {id(value) for value in trained.values()}:
            raise AssertionError("Optimizer parameters differ from complete trained inventory")
        original_step = optimizer.step

        def observed_step(self, *positional, **kwargs):
            del self
            before_parameters = {name: array(value).copy() for name, value in trained.items()}
            norm = 0.
            for name, value in trained.items():
                if value.grad is None:
                    raise AssertionError("Missing trained parameter gradient: " + name)
                grad = array(value.grad)
                record("gradient/" + name, grad)
                # Full-tensor norms supplement the complete elementwise snapshots.
                squared = float(np.sum(grad.astype(np.float64) ** 2))
                record("gradient_norm/" + name, np.asarray(squared ** .5, dtype=np.float64))
                norm += squared
            for index, (role, value) in enumerate(embeddings):
                if value.grad is None:
                    raise AssertionError("Missing requested embedding-output gradient: " + role)
                record("embedding_gradient/%s/%d" % (role, index), value.grad)
            result = original_step(*positional, **kwargs)
            update_squared, changed_elements = 0., 0
            for name, value in trained.items():
                after = array(value)
                record("updated_parameter/" + name, after)
                state = optimizer.state[value]
                step = state['step']
                if float(step) != manifest['optimizer_steps'] + 1:
                    raise AssertionError('Adam step count differs from actual optimizer calls')
                record('adam_step/' + name, np.asarray(float(step), dtype=np.float64), exact=True)
                record('exp_avg/' + name, state['exp_avg'])
                record('exp_avg_sq/' + name, state['exp_avg_sq'])
                before = before_parameters[name]
                changed_elements += int(np.count_nonzero(before != after))
                change = after.astype(np.float64) - before.astype(np.float64)
                update_squared += float(np.sum(change * change))
            manifest["gradient_norms"].append(norm ** .5)
            manifest["parameter_updates"].append({"changed_elements": changed_elements,
                                                   "l2": update_squared ** .5})
            manifest["optimizer_steps"] += 1
            embeddings.clear()
            write_json(target, manifest)
            print("PPO_LORA_STEP", args.algorithm, args.runtime, manifest["optimizer_steps"], flush=True)
            return result

        optimizer.step = types.MethodType(observed_step, optimizer)
        original_backward = trainer.accelerator.backward

        def backward(loss, *positional, **kwargs):
            record('loss', loss)
            return original_backward(loss, *positional, **kwargs)

        trainer.accelerator.backward = backward
        write_json(target, manifest)
        # Model initialization and constructors can use different runtime RNG
        # implementations. Reset the common Python/NumPy schedule immediately
        # before the official loop; sampled token choices are controlled above.
        np.random.seed(0)
        random.seed(0)
        try:
            trainer.train()
        finally:
            for hook in hooks:
                hook.remove()
            optimizer.step = original_step
            trainer.accelerator.backward = original_backward
        runner._synchronize(torch, args.runtime, args.device)
        if manifest["optimizer_steps"] != metadata["expected_optimizer_steps"] or not any(manifest["gradient_norms"]):
            raise AssertionError("Incomplete or entirely zero-gradient training")
        if not any(value["changed_elements"] for value in manifest["parameter_updates"]):
            raise AssertionError("Optimizer never changed any parameter")
        for (role, name), expected in frozen.items():
            value = dict(models.registry[role].named_parameters())[name]
            if array_hash(array(value)) != expected:
                raise AssertionError("Frozen model changed: " + role + "/" + name)
        for (role, name), expected in frozen_buffers.items():
            value = dict(models.registry[role].named_buffers())[name]
            if array_hash(array(value)) != expected:
                raise AssertionError("Frozen model buffer changed: " + role + "/" + name)
        policy = models.registry["policy"]
        policy.eval()
        with torch.no_grad():
            for index, prompt in enumerate(protocol["final_eval_prompts"]):
                inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                inputs = {name: value.to(args.device) for name, value in inputs.items()}
                record("final_eval/%d/input_ids" % index, inputs["input_ids"], exact=True)
                output = policy(**inputs, use_cache=False).logits
                if output.ndim == 3:
                    record("final_eval/%d/next_logits" % index, output[0, -1, :])
        if reference is not None:
            if set(manifest["arrays"]) != set(reference["arrays"]):
                raise AssertionError("Missing candidate observations")
            if len(manifest["generation_calls"]) != len(reference["generation_calls"]):
                raise AssertionError("Missing generation calls")
        validate_manifest(manifest)
        fallback = jt.core.backend_fallback_count() - fallback_before if jt is not None else 0
        if fallback or any(digest(path) != hashes[path.name] for path in sources) or bind_trl_source(trl, args.upstream) != source:
            raise AssertionError("Fallback or changed measurement sources")
        manifest.update(status="complete", fallback_count=fallback,
                        numerical_passed=all(value["passed"] for value in manifest["comparison"].values()) if reference else None,
                        trainer_log_history=trainer.state.log_history)
        write_json(target, manifest)
        print("PPO_LORA_COMPLETE", args.algorithm, args.runtime, manifest["numerical_passed"], flush=True)


if __name__ == "__main__":
    run(options())
