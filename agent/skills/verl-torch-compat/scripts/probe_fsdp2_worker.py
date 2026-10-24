"""Three real verl FSDP2 TrainingWorker updates on two NPU ranks.

Use an independent official-Torch --prepare-checkpoint first. Launch this same
script separately in the two runtimes with WORLD_SIZE=2 and ranks 0/1. Reports
are written to <output stem>.rank<RANK>.json/.npz. The fixed responses and
advantages/returns are fixtures, not rollout, a reward model, GRPO or RLHF.
No framework/model/loss monkeypatch is installed by this script.
The selected upstream NVTX marker functions are observed, not replaced. This
protocol does not validate MSTX or NPU profiler/trace collection.
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

from fsdp2_worker_fixture import (
    INPUTS, MODEL_SPEC, SCRIPT_NAMES, TARGETS, aliases, parameter_shapes, protocol, scope,
)
from probe_training_worker import Recorder, checkpoint_manifest, compatibility_metadata, runtime
from worker_buffer_contract import capture_initial_buffers, expected_inv_freq
from worker_loading_provenance import observe_loading
from worker_npu_provenance import observe_npu_policy


def observe_profiler_markers():
    """Read the already-imported upstream marker bindings, without invoking them."""
    profiler = sys.modules.get("verl.utils.profiler")
    implementation = "verl.utils.profiler.nvtx_profile"
    owner = sys.modules.get(implementation)
    if profiler is None or owner is None:
        raise RuntimeError("upstream NVTX marker implementation was not loaded")
    functions = {}
    for name in ("mark_annotate", "mark_end_range", "mark_start_range", "marked_timer"):
        selected = getattr(profiler, name, None)
        source = getattr(selected, "__module__", None)
        if (not callable(selected) or source != implementation
                or selected is not getattr(owner, name, None)):
            raise RuntimeError("unexpected upstream profiler marker binding: " + name)
        functions[name] = source
    return {"implementation": owner.__name__, "functions": functions,
            "scope": "NVTX markers only; MSTX/NPU profiling not validated"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--role", choices=("actor", "critic"), required=True)
    parser.add_argument("--algorithm", choices=("fixed", "ppo", "grpo", "rlhf", "rlvr"), default="fixed")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-checkpoint", action="store_true")
    parser.add_argument("--stop-after", choices=("import", "init", "reset", "infer", "train"), default="train")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    args.steps, args.backend, args.capture_gradients = 3, "hccl", True
    if args.output.suffix != ".json" or args.timeout < 1:
        parser.error("output must be .json and timeout positive")
    if args.role == "critic" and args.algorithm in ("grpo", "rlvr"):
        parser.error("GRPO/RLVR have no critic role in this protocol")
    if args.prepare_checkpoint:
        if args.framework != "torch" or args.checkpoint.exists():
            parser.error("prepare requires independent Torch and a new checkpoint directory")
    else:
        if not args.checkpoint.is_dir():
            parser.error("checkpoint directory is missing")
        if os.environ.get("WORLD_SIZE") != "2" or os.environ.get("RANK") not in ("0", "1"):
            parser.error("real two-rank launcher environment is required")
        args.output = args.output.with_name(args.output.stem + ".rank" + os.environ["RANK"] + ".json")
    return args


def prepare_checkpoint(args, torch, record):
    from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForTokenClassification
    torch.manual_seed(protocol(args.role)["checkpoint_seed"])
    config = Qwen3Config(
        **{key: value for key, value in MODEL_SPEC.items() if key != "model_type"},
        max_position_embeddings=32, attention_dropout=0.0, num_labels=1,
        classifier_dropout=0.0, pad_token_id=0, bos_token_id=1, eos_token_id=31,
    )
    config._attn_implementation = "eager"
    cls = Qwen3ForCausalLM if args.role == "actor" else Qwen3ForTokenClassification
    model = cls(config).float().cpu()
    actual = dict(model.named_parameters())
    assert {name: tuple(value.shape) for name, value in actual.items()} == parameter_shapes(args.role)
    if args.role == "actor":
        assert model.model.embed_tokens.weight is model.lm_head.weight
    model.save_pretrained(args.checkpoint, safe_serialization=True)
    record.report.update(checkpoint_sha256=checkpoint_manifest(args.checkpoint),
                         checkpoint_parameter_count=sum(int(p.numel()) for p in actual.values()),
                         checkpoint_parameter_tensors=len(actual))


def run_worker(args, record, torch, jt):
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)

    def synchronize():
        jt.sync_all(True) if jt is not None else torch.npu.synchronize()

    def device_assert(value):
        if jt is None:
            assert value.device.type == "npu" and value.device.index == local_rank, str(value.device)
        else:
            native = getattr(value, "_var", value)
            native.sync()
            assert native.location() == "device" and native.device_id == local_rank

    def imports():
        # Resolve modeling modules before optional candidate dependencies join
        # sys.path. Loading only Config leaves HF's flash-attention discovery
        # deferred until the mixed site path is visible to the native oracle.
        from transformers import Qwen3ForCausalLM, Qwen3ForTokenClassification  # noqa: F401
        extra = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
        if extra and extra not in sys.path:
            sys.path.append(extra)
        import verl
        from verl.workers.engine_workers import TrainingWorker
        from verl.utils.distributed import initialize_global_process_group_ray
        from verl.utils.device import get_device_name
        assert verl.__version__ == "0.9.1" and get_device_name() == "npu"
        record.report["versions"] = {name: importlib.metadata.version(name) for name in
                                     ("verl", "transformers", "tensordict", "ray", "peft", "accelerate", "nvtx")}
        record.report["profiler_markers"] = observe_profiler_markers()
        record.report["compatibility"] = compatibility_metadata(args.framework)
        return TrainingWorker, initialize_global_process_group_ray

    TrainingWorker, initialize = record.stage("import", imports)
    if args.stop_after == "import":
        return

    def init_group():
        for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
            assert key in os.environ, "missing launcher variable " + key
        assert not any(key.startswith("RAY_EXPERIMENTAL_NOSET_") and value for key, value in os.environ.items())
        initialize(timeout_second=args.timeout, backend="hccl")
        assert torch.distributed.get_world_size() == 2 and torch.distributed.get_rank() == rank
        assert str(torch.distributed.get_backend()) == "hccl"
        record.report.update(world_size=2, rank=rank, local_rank=local_rank, actual_backend="hccl")
        value = torch.tensor([rank + 1.0, 3.0], device="npu:%d" % local_rank)
        torch.distributed.all_reduce(value)
        synchronize()
        device_assert(value)
        np.testing.assert_array_equal(record.tensor("bootstrap/all_reduce", value), [3.0, 6.0])

    record.stage("distributed_init", init_group)

    def init_worker():
        from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig, TrainingWorkerConfig
        model_type = "language_model" if args.role == "actor" else "value_model"
        model = HFModelConfig(path=str(args.checkpoint), model_type=model_type, load_tokenizer=False,
                              use_remove_padding=False, enable_gradient_checkpointing=False,
                              enable_activation_offload=False, use_fused_kernels=False, use_liger=False,
                              override_config={"attn_implementation": "eager"})
        assert model.hf_config.tie_word_embeddings is True
        config = TrainingWorkerConfig(
            model_type=model_type, model_config=model,
            engine_config=FSDPEngineConfig(
                strategy="fsdp2", fsdp_size=2, seed=42,
                wrap_policy={"transformer_layer_cls_to_wrap": ["Qwen3DecoderLayer"]},
                ulysses_sequence_parallel_size=1, model_dtype="fp32", forward_only=False,
                mixed_precision={"param_dtype": "fp32", "reduce_dtype": "fp32", "buffer_dtype": "fp32"},
                reshard_after_forward=True, forward_prefetch=False,
                use_torch_compile=False, use_dynamic_bsz=False,
                micro_batch_size_per_gpu=2, infer_micro_batch_size_per_gpu=2,
                param_offload=False, optimizer_offload=False, offload_policy=False,
                use_no_sync_for_gradient_accumulation=False,
            ),
            optimizer_config=FSDPOptimizerConfig(
                optimizer="ObservedSGD", optimizer_impl="worker_gradient_observer",
                lr=0.01, weight_decay=0.0, clip_grad=1.0, total_training_steps=3,
                lr_warmup_steps=0, lr_warmup_steps_ratio=0.0), checkpoint_config=getattr(args, "checkpoint_config", None))
        worker = TrainingWorker(config)
        record.report["npu_initialization"] = observe_npu_policy(args.framework)
        record.report.update(worker_class=type(worker).__module__ + "." + type(worker).__name__,
                             engine_class=type(worker.engine).__module__ + "." + type(worker.engine).__name__)
        return worker, model

    worker, model_config = record.stage("init", init_worker)
    if args.stop_after == "init":
        return

    def inventory(stage):
        module = worker.engine.module
        parameters = dict(module.named_parameters())
        assert set(parameters) == set(parameter_shapes(args.role)), sorted(parameters)
        all_paths = dict(module.named_parameters(remove_duplicate=False))
        assert set(all_paths) == {alias for paths in aliases(args.role).values() for alias in paths}
        for name, paths in aliases(args.role).items():
            assert all(all_paths[path] is parameters[name] for path in paths), "broken alias: " + name
        record.report.setdefault("alias_checks", []).append(stage)
        return parameters

    def snapshot(stage, gradients=False):
        parameters = inventory(stage)
        placements = {}
        # All ranks gather every logical tensor in the same name order, even
        # when a particular rank owns an empty local slice of that tensor.
        with torch.no_grad():
            for name in sorted(parameters):
                value = parameters[name].grad if gradients else parameters[name]
                assert value is not None, "missing gradient: " + name
                assert callable(getattr(value, "full_tensor", None)), "not a sharded tensor: " + name
                local = value.to_local()
                count = int(local.numel())
                if count:
                    device_assert(local)
                placements[name] = {"local_numel": count,
                                    "residency": "verified_npu" if count else "empty_not_materialized"}
                full = value.full_tensor()
                device_assert(full)
                assert tuple(full.shape) == parameter_shapes(args.role)[name]
                assert full.dtype == torch.float32 and parameters[name].requires_grad
                record.tensor(stage + "/" + name, full)
        record.report.setdefault("local_shard_observations", {})[stage] = placements

    def reset():
        worker.reset()
        synchronize()
        module = worker.engine.module
        mesh = worker.engine.device_mesh
        assert mesh.ndim == 1 and tuple(mesh.shape) == (2,)
        assert mesh.device_type == "npu" and mesh.mesh.tolist() == [0, 1]
        record.report.update(
            loading_provenance=observe_loading(args.framework),
            optimizer_class=type(worker.engine.optimizer).__module__ + "." + type(worker.engine.optimizer).__name__,
            actual_attention={name or "<root>": sub.config._attn_implementation
                              for name, sub in module.named_modules() if getattr(sub, "config", None) is not None},
            model_spec={key: getattr(module.config, key) for key in MODEL_SPEC},
        )
        snapshot("initial")
        record.report["parameter_names"] = sorted(inventory("reset_verified"))
        record.report["parameter_count"] = sum(record.arrays["initial/" + name].size
                                               for name in record.report["parameter_names"])
        capture_initial_buffers(module, record, device_assert)
        for name in record.report["initial_buffers"]:
            np.testing.assert_allclose(record.arrays["initial_buffer/" + name], expected_inv_freq(module.config),
                                       atol=1e-10, rtol=1e-7)

    record.stage("reset", reset)
    if args.stop_after == "reset":
        return
    from verl import DataProto
    from verl.utils import tensordict_utils as tu
    from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

    def batch(anchor=None, target=None):
        ids = torch.tensor(INPUTS[rank], dtype=torch.long, device="cpu")
        data = dict(input_ids=ids, prompts=ids[:, :4], responses=ids[:, 4:],
                    attention_mask=torch.ones_like(ids),
                    position_ids=torch.arange(8, dtype=torch.long, device="cpu").unsqueeze(0).repeat(2, 1),
                    response_mask=torch.ones((2, 4), dtype=torch.float32, device="cpu"))
        if anchor is not None:
            data["old_log_probs" if args.role == "actor" else "values"] = anchor.detach().cpu().clone()
            values = TARGETS[args.role][rank] if target is None else target
            data["advantages" if args.role == "actor" else "returns"] = torch.tensor(
                values, dtype=torch.float32, device="cpu")
        proto = DataProto.from_single_dict(data, meta_info={"temperature": 1.0, "global_token_num": [8, 8],
                                                           "compute_loss": anchor is not None, "calculate_entropy": False})
        packed = left_right_2_no_padding(proto.to_tensordict())
        if anchor is not None:
            tu.assign_non_tensor(packed, global_batch_size=4)
        return packed

    def infer(key):
        packed = batch()
        output = worker.infer_batch(packed)
        synchronize()
        values = no_padding_2_padding(tu.get(output, "log_probs" if args.role == "actor" else "values"), packed)
        assert tuple(values.shape) == (2, 4)
        record.tensor(key, values)
        return values.detach().cpu().clone()

    anchor = record.stage("infer", lambda: infer("infer/output"))
    record.tensor("fixture/anchor", anchor)
    record.arrays["fixture/input_ids"] = np.asarray(INPUTS[rank], dtype=np.int64)
    target = np.asarray(TARGETS[args.role][rank], dtype=np.float32)
    if args.algorithm != "fixed":
        from verl.trainer.ppo.core_algos import (
            compute_gae_advantage_return, compute_grpo_outcome_advantage, compute_rewards,
        )
        ids = torch.tensor(INPUTS[rank], dtype=torch.long, device="cpu")
        response = ids[:, 4:]
        response_mask = torch.ones((2, 4), dtype=torch.float32, device="cpu")
        terminal = torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float32, device="cpu")
        row_offset = torch.tensor([[0.0], [0.25]], dtype=torch.float32, device="cpu")
        token_scores = response_mask * 0.01 + (response == 7).float() + row_offset * terminal
        ref_log_prob = reward_model_scores = verifier_scores = None
        if args.algorithm == "rlhf":
            # The single-card workflow separately validates real frozen policy
            # and reward-model forwards. This distributed protocol isolates
            # target propagation through FSDP2: both inputs are explicit,
            # deterministic fixtures and are never described as model output.
            reward_model_scores = 1.0 / (1.0 + torch.abs(response[:, -1:].float() - ids[:, 3:4].float()))
            token_scores = reward_model_scores * terminal
            policy_log_prob = torch.zeros_like(response_mask)
            ref_log_prob = policy_log_prob - torch.tensor(
                [[0.01, -0.02, 0.03, -0.04], [-0.02, 0.01, -0.04, 0.03]],
                dtype=torch.float32, device="cpu",
            )
            rewards = compute_rewards(token_scores, policy_log_prob, ref_log_prob, kl_ratio=0.1)
        elif args.algorithm == "rlvr":
            verifier_scores = 1.0 / (
                1.0 + torch.abs(response[:, -1:].float() - ids[:, 3:4].float())
            )
            token_scores = verifier_scores * terminal
            rewards = token_scores
        else:
            rewards = token_scores
        if args.algorithm in ("ppo", "rlhf"):
            baseline = anchor.detach().cpu().clone() if args.role == "critic" else torch.zeros_like(response_mask)
            advantages, returns = compute_gae_advantage_return(
                token_level_rewards=rewards, values=baseline,
                response_mask=response_mask, gamma=0.99, lam=0.95,
            )
        else:
            advantages, returns = compute_grpo_outcome_advantage(
                token_level_rewards=rewards, response_mask=response_mask,
                index=np.asarray(["rank%d" % rank, "rank%d" % rank]),
            )
        target_tensor = advantages if args.role == "actor" else returns
        target = np.array(target_tensor.detach().cpu().numpy(), dtype=np.float32, copy=True)
        for name, value in (
            ("token_level_scores", token_scores), ("rewards", rewards),
            ("advantages", advantages), ("returns", returns),
            ("ref_log_prob", ref_log_prob), ("reward_model_scores", reward_model_scores),
            ("verifier_scores", verifier_scores),
        ):
            if value is not None:
                record.tensor("fixture/" + name, value)
    record.arrays["fixture/targets"] = target
    if args.stop_after == "infer":
        return

    def train():
        from verl.workers.config import ActorConfig, CriticConfig
        from verl.workers.utils.losses import ppo_loss, value_loss
        from worker_gradient_observer import ObservedSGD
        if args.role == "actor":
            loss_config = ActorConfig(strategy="fsdp2", model_config=model_config, rollout_n=1,
                                      ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=2,
                                      use_torch_compile=False, use_dynamic_bsz=False, ppo_epochs=1,
                                      clip_ratio=0.2, clip_ratio_low=0.2, clip_ratio_high=0.2,
                                      clip_ratio_c=3.0, loss_agg_mode="token-mean", entropy_coeff=0.0, use_kl_loss=False)
        else:
            loss_config = CriticConfig(strategy="fsdp2", model=model_config, rollout_n=1,
                                       ppo_mini_batch_size=4, ppo_micro_batch_size_per_gpu=2,
                                       cliprange_value=0.5, loss_agg_mode="token-mean")
        worker.set_loss_fn(partial(ppo_loss if args.role == "actor" else value_loss, config=loss_config))
        assert isinstance(worker.engine.optimizer, ObservedSGD)
        observed = []
        current = [None]
        def capture(optimizer):
            parameters = inventory(current[0] + "/observer")
            active = [p for group in optimizer.param_groups for p in group["params"]]
            assert len(active) == len(parameters) and {id(p) for p in active} == {id(p) for p in parameters.values()}
            for group in optimizer.param_groups:
                assert group["lr"] == 0.01 and group.get("momentum", 0) == 0 and group.get("weight_decay", 0) == 0
            snapshot(current[0] + "/param_grad", gradients=True)
            observed.append(True)
        worker.engine.optimizer.set_gradient_observer(capture)
        record.report["training_steps"] = []
        for step in range(3):
            prefix = "step_%03d" % step
            current[0] = prefix
            observed.clear()
            snapshot(prefix + "/pre_weight")
            output = worker.train_batch(batch(anchor, target))
            synchronize()
            assert len(observed) == 1, "expected exactly one upstream optimizer update"
            snapshot(prefix + "/post_weight")
            metrics = tu.get(output, "metrics")
            loss, grad_norm = float(metrics["loss"]), float(metrics["grad_norm"])
            assert np.isfinite([loss, grad_norm]).all() and grad_norm > 0
            record.arrays[prefix + "/loss"] = np.asarray(loss)
            record.arrays[prefix + "/grad_norm"] = np.asarray(grad_norm)
            infer(prefix + "/output")
            record.report["training_steps"].append(dict(step=step, loss=loss, grad_norm=grad_norm,
                                                        captured_gradient_count=len(parameter_shapes(args.role))))
            record.flush()
    record.stage("train", train)


def main():
    args = parse_args()
    record = Recorder(args)
    record.report.update(schema=2 if args.algorithm != "fixed" else 1,
                         artifact_type="verl_fsdp2_worker", role=args.role,
                         algorithm=args.algorithm, scope=scope(args.algorithm),
                         training_config=protocol(args.role, args.algorithm),
                         script_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                        for name in SCRIPT_NAMES},
                         limitations=["fixed responses and deterministic targets, not rollout-engine generation",
                                      "not actor-critic coupling, reward model or reference KL",
                                      "not Ray RPC, CPU Gloo, offload or performance acceptance",
                                      "NVTX marker selection only; not MSTX/NPU profiler validation",
                                      "empty local slices do not establish allocator residency"])
    if args.algorithm == "rlhf":
        record.report["limitations"].append(
            "distributed RLHF reward/reference values are fixtures; real model forwards are single-card only"
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
                record.report["fallback_count"] = int(jt.core.backend_fallback_count()) - record.report["fallback_before"]
                assert record.report["fallback_count"] == 0
            record.report["status"] = "passed_requested_scope"
            record.flush()
    except Exception as error:
        record.report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        record.flush()
        raise


if __name__ == "__main__":
    main()
