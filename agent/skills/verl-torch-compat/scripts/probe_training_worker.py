"""Stage-by-stage smoke of the real verl 0.9.1 in-process critic TrainingWorker.

First create a shared tiny Qwen3 token-classification checkpoint with an official
Torch interpreter and --prepare-checkpoint. Then run this exact file in official
Torch/torch_npu and Jittor environments against that same local checkpoint.

The worker, FSDP engine and value loss are upstream implementations. The probe
does not install private temporary framework/verl patches. The Jittor run DOES
use its Torch compatibility implementation and installed versioned adapters,
including the pinned verl/TensorDict NestedTensor type-dispatch adaptations;
this is not an unmodified-framework claim. The report records those observed
adapter sources and activation markers, separately from the native Torch run.
The probe uses public initialize_global_process_group_ray(backend="hccl") before
constructing TrainingWorker, so the worker reuses that real process group. It
does not claim the default CPU:gloo/NPU:hccl combination, Ray RPC/serialization,
multiple ranks, rollout servers, or a complete PPO trainer have been validated.

The JSON report is updated before and after each stage; failures are written and
then re-raised, never converted to success. Numerical tensors are saved beside
it as NPZ. --capture-gradients selects a normal SGD subclass through verl's
public optimizer_impl setting. It observes every parameter gradient AFTER
worker clipping, immediately BEFORE SGD updates and the worker clears gradients.
Missing gradients fail explicitly. Without that option ordinary SGD is used;
only value outputs, update trajectories and worker gradient norm are covered.
"""

import argparse
from contextlib import ExitStack
from functools import partial
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from worker_buffer_contract import capture_initial_buffers
from worker_loading_provenance import observe_loading


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--backend", choices=("hccl",), default="hccl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON stage report")
    parser.add_argument("--prepare-checkpoint", action="store_true")
    parser.add_argument("--stop-after", choices=("import", "init", "reset", "infer", "train"), default="train")
    parser.add_argument("--capture-gradients", action="store_true")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error("--output must be a .json path")
    if args.steps < 1 or args.timeout < 1:
        parser.error("steps and timeout must be positive")
    if args.prepare_checkpoint and args.framework != "torch":
        parser.error("checkpoint creation requires independent official Torch")
    if args.prepare_checkpoint and args.checkpoint.exists():
        parser.error("checkpoint creation refuses an existing target directory")
    if not args.prepare_checkpoint and not args.checkpoint.is_dir():
        parser.error("--checkpoint must name the shared existing model directory")
    return args


class Recorder:
    def __init__(self, args):
        self.path = args.output
        self.arrays = {}
        self.report = {
            "schema": 3, "scope": "real in-process single-rank critic TrainingWorker",
            "framework": args.framework, "requested_backend": args.backend,
            "requested_stage": args.stop_after, "status": "running", "stages": [],
            "model_device": "cpu" if args.prepare_checkpoint else "npu",
            "batch_preprocessing_device": "cpu",
            "gradient_scope": "all parameters post-clip/pre-SGD" if args.capture_gradients else "worker grad_norm only",
            "requested_attention": "eager",
            "actual_attention": None,
            "compatibility": None,  # Observed after upstream imports finish.
            "script_sha256": {
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("probe_training_worker.py", "worker_gradient_observer.py", "worker_buffer_contract.py", "worker_loading_provenance.py")
            },
            "training_config": {
                "optimizer": "SGD", "lr": 0.01, "weight_decay": 0.0,
                "clip_grad": 1.0, "cliprange_value": 0.5, "dtype": "float32",
                "steps": args.steps, "batch_size": 2, "sequence_length": 8,
                "response_length": 4, "micro_batch_size": 2,
                "engine_seed": 42, "mask_seed": None, "mask_rule": "all_valid",
                "input_ids": [[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 9, 10, 11, 12]],
                "returns": [[0.4, 0.3, 0.2, 0.1], [-0.1, 0.1, 0.3, 0.5]],
                "gradient_capture": args.capture_gradients,
                "wrap_policy": {"disable": True},
            },
            "oracle_fallback_count": None,
            "oracle_fallback_status": "not_instrumented" if args.framework == "torch" else "not_applicable",
            "limitations": ["not Ray RPC", "not multi-rank", "not CPU gloo",
                            "no actor/rollout/reward trainer", "not performance",
                            "native Torch CPU fallback warnings require preserved stderr review"],
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def flush(self):
        if self.arrays:
            np.savez(self.path.with_suffix(".npz"), **self.arrays)
        self.path.write_text(json.dumps(self.report, indent=2) + "\n")

    def stage(self, name, function):
        entry = {"name": name, "status": "running"}
        self.report["stages"].append(entry)
        self.flush()
        print("worker probe stage: " + name, flush=True)
        result = function()
        entry["status"] = "passed"
        self.flush()
        return result

    def tensor(self, name, value):
        array = np.array(value.detach().cpu().numpy(), copy=True)
        if not np.isfinite(array).all():
            raise AssertionError("nonfinite tensor: " + name)
        self.arrays[name] = array
        return array


def checkpoint_manifest(path):
    files = sorted(path.glob("*.json")) + sorted(path.glob("*.safetensors"))
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("shared checkpoint must contain safetensors")
    hashes = {}
    for file in files:
        digest = hashlib.sha256()
        with file.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[file.name] = digest.hexdigest()
    return hashes


def compatibility_metadata(framework):
    """Observe already-loaded adapters; never install/import/patch them here.

    This deliberately inventories only the two nested-integration adapters,
    not every optional adapter or every framework compatibility implementation.
    Distribution metadata may be absent in a source checkout; source SHA and
    observed third-party activation markers remain independent evidence.
    """
    result = {
        "harness_private_patches": False,
        "framework_layer": "native_torch" if framework == "torch" else "jittor.compat.torch",
        "status": "not_applicable" if framework == "torch" else "observed",
        "observation_scope": "verl/TensorDict nested integration adapters only; not exhaustive",
        "distribution_version": None, "distribution_version_status": "not_applicable",
        "adapters": {},
    }
    if framework == "torch":
        return result
    if framework != "jittor":
        raise ValueError("unknown framework for compatibility observation")
    try:
        result["distribution_version"] = importlib.metadata.version("jittor-torch-adapters")
        result["distribution_version_status"] = "installed_metadata"
    except importlib.metadata.PackageNotFoundError:
        result["distribution_version_status"] = "metadata_unavailable"
    frontend = sys.modules.get("jittor.compat.torch")
    getter = getattr(frontend, "nested_tensor_type", None)
    nested_type = getter() if callable(getter) else None
    for name, scope in (
        ("verl", "five pinned TensorDict helper type predicates; no mathematical replacement"),
        ("tensordict", "0.10.0 accepted Nested leaf registry and two private validators"),
    ):
        module = sys.modules.get("jittor_adapters." + name)
        item = {"loaded": module is not None, "source_sha256": None,
                "supported_versions": sorted(getattr(module, "SUPPORTED_VERSIONS", ())),
                "scope": scope, "active_targets": {}}
        if module is not None:
            source_path = getattr(module, "__file__", None)
            try:
                item["source_sha256"] = hashlib.sha256(Path(source_path).read_bytes()).hexdigest() if source_path else None
            except OSError:
                item["source_status"] = "unreadable"
        if name == "verl":
            item["upstream_source_sha256"] = getattr(module, "SOURCE_SHA256", None)
            helper = sys.modules.get("verl.utils.tensordict_utils")
            for target in ("get_tensordict", "get", "chunk_tensordict", "concat_tensordict", "index_select_tensor_dict"):
                function = getattr(helper, target, None)
                item["active_targets"][target] = (
                    isinstance(nested_type, type)
                    and getattr(function, "_jittor_verl_nested_type", None) is nested_type
                    and callable(getattr(function, "_jittor_verl_original", None)))
        else:
            base = sys.modules.get("tensordict.base")
            owner = getattr(base, "TensorDictBase", None)
            accepted = getattr(base, "_ACCEPTED_CLASSES", ())
            item["nested_leaf_registered"] = isinstance(nested_type, type) and nested_type in accepted
            for target in ("_validate_value_generic", "_validate_value_devicefree"):
                item["active_targets"][target] = (
                    isinstance(nested_type, type)
                    and getattr(getattr(owner, target, None), "_jittor_tensordict_nested", None) is nested_type)
        result["adapters"][name] = item
    return result


def runtime(args, stack, record):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # Transformers' public loader option: CANN device context is thread-local,
    # and the current Jittor safetensors path cannot initialize arbitrary HF
    # loader threads. Both runtimes use serial loading for this comparison.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    record.report["async_weight_loading"] = False
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    jt = None
    if args.framework == "jittor":
        os.environ["JITTOR_TORCH_SHIM"] = "1"
        import jittor as jt
        import torch
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner
        from jittor._runtime.fallback import forbid_backend_fallbacks

        assert namespace_owner(torch) is jt and activation_status(jt).active
        assert torch.Tensor is not jt.Var
        assert jt.compiler.has_acl, "ACL unavailable"
        stack.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1, backend_fallback="error"))
        stack.enter_context(forbid_backend_fallbacks())
        record.report["fallback_before"] = int(jt.core.backend_fallback_count())
    else:
        os.environ.pop("JITTOR_TORCH_SHIM", None)
        import torch

        assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
        if not args.prepare_checkpoint:
            import torch_npu  # noqa: F401

            assert torch.npu.is_available(), "native NPU unavailable"
    record.report["torch_api_version"] = str(getattr(torch, "__torch_version__", torch.__version__))
    return torch, jt


def prepare_checkpoint(args, torch, record):
    from transformers import Qwen3Config, Qwen3ForTokenClassification

    torch.manual_seed(20260923)
    config = Qwen3Config(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=32, attention_dropout=0.0, num_labels=1,
        classifier_dropout=0.0, tie_word_embeddings=False, pad_token_id=0,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForTokenClassification(config).float().cpu()
    model.save_pretrained(args.checkpoint, safe_serialization=True)
    record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
    record.report["checkpoint_parameter_count"] = sum(int(p.numel()) for p in model.parameters())


def run_worker(args, stack, record, torch, jt):
    # Device allocation, rendezvous address/port and isolation are launcher-owned.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if jt is None:
        torch.npu.set_device(local_rank)

    def synchronize():
        jt.sync_all(True) if jt is not None else torch.npu.synchronize()

    def device_assert(value):
        expected_index = int(os.environ.get("LOCAL_RANK", "0"))
        if jt is None:
            assert value.device.type == "npu", str(value.device)
            assert value.device.index == expected_index, str(value.device)
        else:
            native = getattr(value, "_var", value)
            # Clipping builds a lazy gradient update. Observe residency only
            # after executing it; sync must not be replaced by a device move.
            native.sync()
            assert native.location() == "device", native.location()
            assert native.device_id == expected_index, native.device_id

    def imports():
        # Resolve Transformers against the already verified interpreter before
        # a deliberately supplied shared pure-Python dependency site is added.
        from transformers import Qwen3ForTokenClassification  # noqa: F401

        extra = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
        if extra and extra not in sys.path:
            sys.path.append(extra)
        import verl
        from verl.workers.engine_workers import TrainingWorker
        from verl.utils.distributed import initialize_global_process_group_ray
        from verl.utils.device import get_device_name

        assert verl.__version__ == "0.9.1", verl.__version__
        record.report["verl_device_name"] = get_device_name()
        assert get_device_name() == "npu", "verl did not select its NPU platform"
        record.report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("verl", "transformers", "tensordict", "ray", "peft", "accelerate")
        }
        record.report["compatibility"] = compatibility_metadata(args.framework)
        return TrainingWorker, initialize_global_process_group_ray

    TrainingWorker, initialize = record.stage("import", imports)
    if args.stop_after == "import":
        return

    def init_group():
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("this initial worker probe is deliberately single-rank")
        for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
            if key not in os.environ:
                raise ValueError("launcher must supply " + key)
        if any(key.startswith("RAY_EXPERIMENTAL_NOSET_") and value for key, value in os.environ.items()):
            raise ValueError("in-process worker must not inherit Ray NOSET flags (they require a Ray actor context)")
        initialize(timeout_second=args.timeout, backend=args.backend)
        actual = str(torch.distributed.get_backend())
        record.report["actual_backend"] = actual
        record.report["world_size"] = torch.distributed.get_world_size()
        record.report["rank"] = torch.distributed.get_rank()
        assert actual == args.backend, (actual, args.backend)
        # Successful init alone is insufficient: execute a real device collective.
        value = torch.tensor([1.0, 3.0], device="npu")
        torch.distributed.all_reduce(value)
        synchronize()
        device_assert(value)
        np.testing.assert_array_equal(record.tensor("bootstrap/all_reduce", value), [1.0, 3.0])

    record.stage("distributed_init", init_group)

    def init_worker():
        from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig, TrainingWorkerConfig

        model_config = HFModelConfig(
            path=str(args.checkpoint), model_type="value_model", load_tokenizer=False,
            use_remove_padding=False, enable_gradient_checkpointing=False,
            enable_activation_offload=False, use_fused_kernels=False, use_liger=False,
            override_config={"attn_implementation": "eager"},
        )
        config = TrainingWorkerConfig(
            model_type="value_model", model_config=model_config,
            engine_config=FSDPEngineConfig(
                strategy="fsdp", fsdp_size=1, use_orig_params=True,
                wrap_policy={"disable": True},
                ulysses_sequence_parallel_size=1, model_dtype="fp32",
                mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
                use_torch_compile=False, use_dynamic_bsz=False,
                micro_batch_size_per_gpu=2, infer_micro_batch_size_per_gpu=2,
                param_offload=False, optimizer_offload=False, offload_policy=False,
                use_no_sync_for_gradient_accumulation=False,
            ),
            optimizer_config=FSDPOptimizerConfig(
                optimizer="ObservedSGD" if args.capture_gradients else "SGD",
                optimizer_impl="worker_gradient_observer" if args.capture_gradients else "torch.optim",
                lr=0.01, weight_decay=0.0, clip_grad=1.0,
                total_training_steps=args.steps, lr_warmup_steps=0,
                lr_warmup_steps_ratio=0.0,
            ),
            checkpoint_config=None,
        )
        worker = TrainingWorker(config)
        if type(worker) is not TrainingWorker:
            raise AssertionError("expected the upstream TrainingWorker itself")
        record.report["worker_class"] = type(worker).__module__ + "." + type(worker).__name__
        record.report["engine_class"] = type(worker.engine).__module__ + "." + type(worker.engine).__name__
        return worker, model_config

    worker, model_config = record.stage("init", init_worker)
    if args.stop_after == "init":
        return

    def reset():
        worker.reset()
        synchronize()
        record.report["loading_provenance"] = observe_loading(args.framework)
        attention_configs = {}
        for name, module in worker.engine.module.named_modules():
            config = getattr(module, "config", None)
            if config is not None:
                actual = getattr(config, "_attn_implementation", None)
                attention_configs[name or "<root>"] = actual
        record.report["actual_attention"] = attention_configs
        model_hf_config = worker.engine.module.config
        record.report["model_spec"] = {
            key: getattr(model_hf_config, key)
            for key in ("model_type", "vocab_size", "hidden_size", "intermediate_size",
                        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                        "head_dim", "num_labels", "attention_bias", "tie_word_embeddings")
        }
        record.report["optimizer_class"] = (
            type(worker.engine.optimizer).__module__ + "." + type(worker.engine.optimizer).__name__
        )
        parameters = dict(worker.engine.module.named_parameters())
        for name, parameter in parameters.items():
            device_assert(parameter)
            record.tensor("initial/" + name, parameter)
        record.report["parameter_names"] = sorted(parameters)
        record.report["parameter_count"] = sum(int(p.numel()) for p in parameters.values())
        capture_initial_buffers(worker.engine.module, record, device_assert)
        return parameters

    parameters = record.stage("reset", reset)
    if args.stop_after == "reset":
        return
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    def batch(values=None, training=False):
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8], [1, 2, 3, 5, 9, 10, 11, 12]], dtype=torch.long, device="cpu")
        data = {
            "input_ids": ids, "prompts": ids[:, :4], "responses": ids[:, 4:],
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(8, dtype=torch.long, device="cpu").unsqueeze(0).repeat(2, 1),
            "response_mask": torch.ones((2, 4), dtype=torch.float32, device="cpu"),
        }
        if values is not None:
            data["values"] = values.detach().cpu().clone()
            data["returns"] = torch.tensor([[0.4, 0.3, 0.2, 0.1], [-0.1, 0.1, 0.3, 0.5]], device="cpu")
        proto = DataProto.from_single_dict(
            data, meta_info={"temperature": 1.0, "global_token_num": [8, 8], "compute_loss": training},
        )
        packed = left_right_2_no_padding(proto.to_tensordict())
        if training:
            tu.assign_non_tensor(packed, global_batch_size=2)
        return packed

    def infer():
        packed = batch()
        # Direct in-process calls return TensorDict, not a Ray future.
        result = worker.infer_batch(packed)
        synchronize()
        values = no_padding_2_padding(tu.get(result, "values"), packed)
        assert tuple(values.shape) == (2, 4), tuple(values.shape)
        record.tensor("infer/values", values)
        return values

    old_values = record.stage("infer", infer)
    if args.stop_after == "infer":
        return

    def train():
        from verl.workers.config import CriticConfig
        from verl.workers.utils.losses import value_loss

        critic = CriticConfig(
            strategy="fsdp", model=model_config, ppo_micro_batch_size_per_gpu=2,
            ppo_mini_batch_size=2, rollout_n=1, cliprange_value=0.5,
        )
        worker.set_loss_fn(partial(value_loss, config=critic))
        captured = {}
        observed_steps = []
        if args.capture_gradients:
            from worker_gradient_observer import ObservedSGD

            if not isinstance(worker.engine.optimizer, ObservedSGD):
                raise AssertionError("verl did not instantiate the requested ObservedSGD")

            def capture(optimizer):
                active = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                expected = {id(p): name for name, p in parameters.items() if p.requires_grad}
                if len(active) != len(expected) or {id(p) for p in active} != set(expected):
                    raise AssertionError("optimizer parameters differ from named trainable parameters")
                missing = [expected[id(p)] for p in active if p.grad is None]
                if missing:
                    raise AssertionError("missing pre-SGD parameter gradients: " + repr(missing))
                for parameter in active:
                    name = expected[id(parameter)]
                    gradient = parameter.grad
                    device_assert(gradient)
                    array = np.array(gradient.detach().cpu().numpy(), copy=True)
                    if array.shape != tuple(parameter.shape) or not np.isfinite(array).all():
                        raise AssertionError("invalid pre-SGD parameter gradient: " + name)
                    captured[name] = array
                observed_steps.append(True)

            worker.engine.optimizer.set_gradient_observer(capture)
        summaries = []
        for step in range(args.steps):
            captured.clear()
            observed_steps.clear()
            before = {name: np.array(p.detach().cpu().numpy(), copy=True) for name, p in parameters.items()}
            output = worker.train_batch(batch(old_values, training=True))
            synchronize()
            prefix = "step_{:03d}/".format(step)
            metrics = tu.get(output, "metrics")
            # These two entries are scalar numeric values in the real worker's
            # output; the other entries may be verl Metric objects.
            loss = float(metrics["loss"])
            grad_norm = float(metrics["grad_norm"])
            if not np.isfinite([loss, grad_norm]).all() or grad_norm <= 0:
                raise AssertionError("nonfinite loss or absent/nonpositive gradient norm")
            largest_change = 0.0
            for name, parameter in parameters.items():
                device_assert(parameter)
                updated = record.tensor(prefix + "post_weight/" + name, parameter)
                largest_change = max(largest_change, float(np.max(np.abs(updated - before[name]))))
            if largest_change <= 0:
                raise AssertionError("real worker optimizer changed no parameter")
            if args.capture_gradients:
                if len(observed_steps) != 1:
                    raise AssertionError("expected exactly one observed optimizer step")
                missing = [name for name, p in parameters.items() if p.requires_grad and name not in captured]
                if missing:
                    raise AssertionError("pre-SGD gradients not observed: " + repr(missing))
                for name, array in captured.items():
                    if not np.isfinite(array).all():
                        raise AssertionError("nonfinite parameter gradient: " + name)
                    record.arrays[prefix + "param_grad/" + name] = array
            record.arrays[prefix + "loss"] = np.asarray(loss)
            record.arrays[prefix + "grad_norm"] = np.asarray(grad_norm)
            summaries.append({"step": step, "loss": loss, "grad_norm": grad_norm,
                              "largest_parameter_change": largest_change,
                              "captured_gradient_count": len(captured)})
            record.report["training_steps"] = summaries
            record.flush()
        # Fresh infer through the worker establishes that updated weights affect
        # the same real inference path, not only its state_dict.
        packed = batch()
        final_output = worker.infer_batch(packed)
        synchronize()
        final_values = no_padding_2_padding(tu.get(final_output, "values"), packed)
        record.tensor("post_train/values", final_values)

    record.stage("train", train)


def main():
    args = parse_args()
    record = Recorder(args)
    try:
        with ExitStack() as stack:
            torch, jt = record.stage("runtime", lambda: runtime(args, stack, record))
            if args.prepare_checkpoint:
                record.stage("prepare_checkpoint", lambda: prepare_checkpoint(args, torch, record))
            else:
                record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
                run_worker(args, stack, record, torch, jt)
            if jt is not None:
                jt.sync_all(True)
                count = int(jt.core.backend_fallback_count()) - record.report["fallback_before"]
                record.report["fallback_count"] = count
                assert count == 0, "backend fallback occurred"
            record.report["status"] = "passed_requested_scope"
            record.flush()
    except Exception as error:
        # Diagnostic boundary only: persist the failure then propagate it so
        # shell/CI cannot mistake a written report for a successful worker.
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
