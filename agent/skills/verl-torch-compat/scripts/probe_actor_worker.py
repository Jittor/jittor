"""Real verl 0.9.1 language-model TrainingWorker and PPO-loss gradient probe.

Prepare one independent official-Torch tiny Qwen3ForCausalLM checkpoint, then
run this file in separate Torch/torch_npu and Jittor interpreters. The default
three updates use upstream FSDP TrainingWorker, upstream ppo_loss and ordinary
SGD observed after clipping. Advantages are an explicitly synthetic fixture;
responses are fixed inputs, NOT generated rollout. No reward model, verifier,
reference KL, critic coupling, Ray RPC, multi-rank or full RLHF claim is made.
No private temporary patches are installed by this probe. The Jittor candidate
uses its Torch compatibility layer and installed versioned adapters; observed
verl/TensorDict adapter versions, source hashes and activation are reported.
"""
import argparse
from contextlib import ExitStack
from functools import partial
import hashlib
import importlib.metadata
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from actor_worker_fixture import (
    ADVANTAGES, INPUT_IDS, MODEL_SPEC, SCOPE, SCRIPT_NAMES, training_config,
)
from probe_training_worker import Recorder, checkpoint_manifest, compatibility_metadata, runtime


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--backend", choices=("hccl",), default="hccl")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-checkpoint", action="store_true")
    parser.add_argument("--stop-after", choices=("import", "init", "reset", "infer", "train"), default="train")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    args.capture_gradients = True
    if args.output.suffix != ".json":
        parser.error("--output must end with .json")
    if not 1 <= args.steps <= 999 or args.timeout < 1:
        parser.error("steps must be 1..999 and timeout positive; acceptance requires >=3 steps")
    if args.prepare_checkpoint:
        if args.framework != "torch" or args.checkpoint.exists():
            parser.error("prepare needs official Torch and a new checkpoint directory")
    elif not args.checkpoint.is_dir():
        parser.error("shared checkpoint directory does not exist")
    return args


def prepare_checkpoint(args, torch, record):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(20260924)
    config = Qwen3Config(
        **{key: value for key, value in MODEL_SPEC.items() if key != "model_type"},
        max_position_embeddings=32, attention_dropout=0.0, pad_token_id=0,
        eos_token_id=31, bos_token_id=1,
    )
    config._attn_implementation = "eager"
    model = Qwen3ForCausalLM(config).float().cpu()
    model.save_pretrained(args.checkpoint, safe_serialization=True)
    record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
    record.report["checkpoint_parameter_count"] = sum(int(p.numel()) for p in model.parameters())


def run_worker(args, record, torch, jt):
    if jt is None:
        torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))

    def synchronize():
        jt.sync_all(True) if jt is not None else torch.npu.synchronize()

    def device_assert(value):
        expected_index = int(os.environ.get("LOCAL_RANK", "0"))
        if jt is None:
            assert value.device.type == "npu", str(value.device)
            assert value.device.index == expected_index, str(value.device)
        else:
            native = getattr(value, "_var", value)
            # Match the critic observer: execute a lazy gradient update before
            # asserting residency, without migrating it or changing its value.
            native.sync()
            assert native.location() == "device", native.location()
            assert native.device_id == expected_index, native.device_id

    def imports():
        from transformers import Qwen3ForCausalLM  # noqa: F401
        extra = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
        if extra and extra not in sys.path:
            sys.path.append(extra)
        import verl
        from verl.workers.engine_workers import TrainingWorker
        from verl.utils.distributed import initialize_global_process_group_ray
        from verl.utils.device import get_device_name

        assert verl.__version__ == "0.9.1"
        assert get_device_name() == "npu", "verl selected a non-NPU backend"
        record.report["verl_device_name"] = get_device_name()
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
            raise ValueError("this actor-worker probe only covers one rank")
        for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"):
            if key not in os.environ:
                raise ValueError("launcher must supply " + key)
        if any(key.startswith("RAY_EXPERIMENTAL_NOSET_") and value for key, value in os.environ.items()):
            raise ValueError("in-process probe must not inherit Ray NOSET flags")
        initialize(timeout_second=args.timeout, backend=args.backend)
        record.report.update(
            actual_backend=str(torch.distributed.get_backend()),
            world_size=torch.distributed.get_world_size(), rank=torch.distributed.get_rank(),
        )
        assert record.report["actual_backend"] == "hccl"
        assert record.report["world_size"] == 1 and record.report["rank"] == 0
        value = torch.tensor([1.0, 3.0], device="npu")
        torch.distributed.all_reduce(value)
        synchronize()
        device_assert(value)
        np.testing.assert_array_equal(record.tensor("bootstrap/all_reduce", value), [1.0, 3.0])

    record.stage("distributed_init", init_group)

    def init_worker():
        from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig, TrainingWorkerConfig

        model = HFModelConfig(
            path=str(args.checkpoint), model_type="language_model", load_tokenizer=False,
            use_remove_padding=False, enable_gradient_checkpointing=False,
            enable_activation_offload=False, use_fused_kernels=False, use_liger=False,
            override_config={"attn_implementation": "eager"},
        )
        config = TrainingWorkerConfig(
            model_type="language_model", model_config=model,
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
                optimizer="ObservedSGD", optimizer_impl="worker_gradient_observer",
                lr=0.01, weight_decay=0.0, clip_grad=1.0, total_training_steps=args.steps,
                lr_warmup_steps=0, lr_warmup_steps_ratio=0.0,
            ),
            checkpoint_config=None,
        )
        worker = TrainingWorker(config)
        assert type(worker) is TrainingWorker
        record.report["worker_class"] = type(worker).__module__ + "." + type(worker).__name__
        record.report["engine_class"] = type(worker.engine).__module__ + "." + type(worker.engine).__name__
        return worker, model

    worker, model_config = record.stage("init", init_worker)
    if args.stop_after == "init":
        return

    def reset():
        worker.reset()
        synchronize()
        from worker_loading_provenance import observe_loading
        record.report["loading_provenance"] = observe_loading(args.framework)
        attention = {}
        for name, module in worker.engine.module.named_modules():
            if getattr(module, "config", None) is not None:
                attention[name or "<root>"] = getattr(module.config, "_attn_implementation", None)
        record.report["actual_attention"] = attention
        record.report["model_spec"] = {key: getattr(worker.engine.module.config, key) for key in MODEL_SPEC}
        optimizer = worker.engine.optimizer
        record.report["optimizer_class"] = type(optimizer).__module__ + "." + type(optimizer).__name__
        parameters = dict(worker.engine.module.named_parameters())
        for name, parameter in parameters.items():
            assert parameter.requires_grad, "unexpected frozen actor parameter: " + name
            device_assert(parameter)
            record.tensor("initial/" + name, parameter)
        record.report["parameter_names"] = sorted(parameters)
        record.report["parameter_count"] = sum(int(p.numel()) for p in parameters.values())
        from worker_buffer_contract import capture_initial_buffers
        capture_initial_buffers(worker.engine.module, record, device_assert)
        return parameters

    parameters = record.stage("reset", reset)
    if args.stop_after == "reset":
        return
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    def batch(old_log_probs=None):
        ids = torch.tensor(INPUT_IDS, dtype=torch.long, device="cpu")
        data = {
            "input_ids": ids, "prompts": ids[:, :4], "responses": ids[:, 4:],
            "attention_mask": torch.ones_like(ids),
            "position_ids": torch.arange(8, dtype=torch.long, device="cpu").unsqueeze(0).repeat(2, 1),
            "response_mask": torch.ones((2, 4), dtype=torch.float32, device="cpu"),
        }
        training = old_log_probs is not None
        if training:
            data["old_log_probs"] = old_log_probs.detach().cpu().clone()
            data["advantages"] = torch.tensor(ADVANTAGES, dtype=torch.float32, device="cpu")
        proto = DataProto.from_single_dict(data, meta_info={
            "temperature": 1.0, "global_token_num": [8, 8], "compute_loss": training,
            "calculate_entropy": False,
        })
        packed = left_right_2_no_padding(proto.to_tensordict())
        if training:
            tu.assign_non_tensor(packed, global_batch_size=2)
        return packed

    def infer(name):
        packed = batch()
        output = worker.infer_batch(packed)
        synchronize()
        log_probs = no_padding_2_padding(tu.get(output, "log_probs"), packed)
        assert tuple(log_probs.shape) == (2, 4), {
            "padded_log_probs_shape": tuple(log_probs.shape),
            "packed_log_probs_values_shape": tuple(tu.get(output, "log_probs").values().shape),
        }
        record.tensor(name, log_probs)
        return log_probs.detach().cpu().clone()

    old_log_probs = record.stage("infer", lambda: infer("infer/log_probs"))
    record.tensor("fixture/old_log_probs", old_log_probs)
    record.arrays["fixture/advantages"] = np.asarray(ADVANTAGES, dtype=np.float32)
    record.arrays["fixture/input_ids"] = np.asarray(INPUT_IDS, dtype=np.int64)
    record.arrays["fixture/response_mask"] = np.ones((2, 4), dtype=np.float32)
    if args.stop_after == "infer":
        return

    def train():
        from verl.workers.config import ActorConfig
        from verl.workers.utils.losses import ppo_loss
        from worker_gradient_observer import ObservedSGD

        actor = ActorConfig(
            strategy="fsdp", model_config=model_config, rollout_n=1,
            ppo_mini_batch_size=2, ppo_micro_batch_size_per_gpu=2,
            use_torch_compile=False, use_dynamic_bsz=False, ppo_epochs=1,
            clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2,
            clip_ratio_c=3.0, loss_agg_mode="token-mean", entropy_coeff=0.0,
            use_kl_loss=False,
        )
        worker.set_loss_fn(partial(ppo_loss, config=actor))
        assert isinstance(worker.engine.optimizer, ObservedSGD)
        captured, observed_steps = {}, []

        def capture(optimizer):
            expected = {id(p): name for name, p in parameters.items()}
            active = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
            if len(active) != len(expected) or {id(p) for p in active} != set(expected):
                raise AssertionError("optimizer and named actor parameters differ")
            for group in optimizer.param_groups:
                assert group["lr"] == 0.01 and group.get("momentum", 0) == 0
                assert group.get("weight_decay", 0) == 0
            for parameter in active:
                name = expected[id(parameter)]
                gradient = parameter.grad
                if gradient is None:
                    raise AssertionError("missing actor gradient: " + name)
                device_assert(gradient)
                array = np.array(gradient.detach().cpu().numpy(), copy=True)
                if array.shape != tuple(parameter.shape) or not np.isfinite(array).all():
                    raise AssertionError("invalid actor gradient: " + name)
                captured[name] = array
            if not any(np.any(array != 0) for array in captured.values()):
                raise AssertionError("all observed actor gradients are zero")
            observed_steps.append(True)

        worker.engine.optimizer.set_gradient_observer(capture)
        record.report["training_steps"] = []
        for step in range(args.steps):
            captured.clear()
            observed_steps.clear()
            before = {name: np.array(p.detach().cpu().numpy(), copy=True) for name, p in parameters.items()}
            output = worker.train_batch(batch(old_log_probs))
            synchronize()
            metrics = tu.get(output, "metrics")
            loss, grad_norm = float(metrics["loss"]), float(metrics["grad_norm"])
            if not np.isfinite([loss, grad_norm]).all() or grad_norm <= 0:
                raise AssertionError("nonfinite loss or invalid actor grad_norm")
            if len(observed_steps) != 1 or set(captured) != set(parameters):
                raise AssertionError("incomplete pre-SGD gradient capture")
            prefix = "step_{:03d}/".format(step)
            largest = 0.0
            for name, parameter in parameters.items():
                device_assert(parameter)
                updated = record.tensor(prefix + "post_weight/" + name, parameter)
                largest = max(largest, float(np.max(np.abs(updated - before[name]))))
                record.arrays[prefix + "param_grad/" + name] = captured[name]
            if largest <= 0:
                raise AssertionError("actor optimizer changed no parameter")
            record.arrays[prefix + "loss"] = np.asarray(loss)
            record.arrays[prefix + "grad_norm"] = np.asarray(grad_norm)
            infer(prefix + "post_log_probs")
            record.report["training_steps"].append({
                "step": step, "loss": loss, "grad_norm": grad_norm,
                "largest_parameter_change": largest, "captured_gradient_count": len(captured),
            })
            record.flush()

    record.stage("train", train)


def main():
    args = parse_args()
    record = Recorder(args)
    record.report.update(
        schema=2, artifact_type="verl_actor_worker", scope=SCOPE,
        training_config=training_config(args.steps),
        script_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in SCRIPT_NAMES},
        limitations=["fixed response and advantage fixture, not rollout", "no reward/critic/reference KL",
                     "not full RLHF/RLVR or GRPO", "not Ray RPC or multi-rank", "not performance",
                     "native Torch fallback warnings require preserved stderr review"],
    )
    try:
        with ExitStack() as stack:
            torch, jt = record.stage("runtime", lambda: runtime(args, stack, record))
            if args.prepare_checkpoint:
                record.stage("prepare_checkpoint", lambda: prepare_checkpoint(args, torch, record))
            else:
                record.report["checkpoint_sha256"] = checkpoint_manifest(args.checkpoint)
                run_worker(args, record, torch, jt)
            if jt is not None:
                jt.sync_all(True)
                count = int(jt.core.backend_fallback_count()) - record.report["fallback_before"]
                record.report["fallback_count"] = count
                assert count == 0, "backend fallback occurred"
            record.report["status"] = "passed_requested_scope"
            record.flush()
    except Exception as error:
        record.report.update(status="failed", error_type=type(error).__name__, error=str(error),
                             traceback=traceback.format_exc())
        if record.report["stages"] and record.report["stages"][-1]["status"] == "running":
            record.report["stages"][-1]["status"] = "failed"
        record.flush()
        raise


if __name__ == "__main__":
    main()
