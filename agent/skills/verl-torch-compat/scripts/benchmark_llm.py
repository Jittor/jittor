"""Shared CPU/CUDA/NPU single-/multi-card Qwen actor microbenchmark.

The launcher selects the framework by choosing an isolated Python environment.
This script intentionally uses the same Transformers version, model files,
dtype, eager-attention implementation, inputs, and timing boundaries.
The PPO objective is an actor-update fixture with fixed old probabilities and
unit advantages, not a full verl trainer or evidence of task-quality learning.
The default short run and allocator-specific peaks are preliminary observations,
not an accepted repeated/interleaved performance or cross-runtime memory gate.
"""

import argparse
import contextlib
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


TRAIN_TEXTS = (
    "用户：请用三句话解释梯度下降。助手：梯度下降通过计算目标函数的梯度，"
    "沿着下降方向更新参数，并重复迭代直到损失收敛。",
    "User: Explain why the sky appears blue. Assistant: Shorter blue wavelengths "
    "are scattered more strongly by molecules in Earth's atmosphere.",
)

GRADIENT_SAMPLE_NAMES = frozenset((
    "model.layers.0.input_layernorm.weight",
    "model.layers.0.linear_attn.dt_bias",
    "model.layers.3.input_layernorm.weight",
    "model.layers.3.self_attn.q_norm.weight",
    "model.norm.weight",
))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "npu"), default="npu")
    parser.add_argument("--task", choices=("train", "infer"), required=True)
    parser.add_argument("--objective", choices=("causal", "ppo"), default="causal")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--new-tokens", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    return parser.parse_args()


def percentile(values, fraction):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return ordered[index]


def distributed_backend(device):
    return {"cpu": "gloo", "cuda": "nccl", "npu": "hccl"}[device]


def performance_context(device):
    """Record launcher policy; shared CUDA never qualifies for performance acceptance."""
    authorization = {
        "VERL_ALLOW_SHARED_CUDA": os.environ.get("VERL_ALLOW_SHARED_CUDA", "0"),
        "VERL_L4_PERFORMANCE_ELIGIBLE": os.environ.get("VERL_L4_PERFORMANCE_ELIGIBLE", "1"),
    }
    if device == "cuda":
        for name, value in authorization.items():
            if value not in ("0", "1"):
                raise ValueError(name + " must be 0 or 1")
        shared = authorization["VERL_ALLOW_SHARED_CUDA"] == "1"
        eligible = authorization["VERL_L4_PERFORMANCE_ELIGIBLE"] == "1" and not shared
    else:
        # CUDA sharing authorization does not change legacy CPU/NPU semantics.
        shared, eligible = False, True
    return {
        "shared_cuda": shared,
        "performance_eligible": eligible,
        "runtime_authorization": authorization,
    }


def initialize_runtime(args, guard, rank, local_rank, world_size):
    """Configure the requested backend without substituting host execution."""

    jt = None
    fallback_before = None
    if args.framework == "torch":
        assert hasattr(torch, "_C"), "oracle must be an official PyTorch build"
        assert not hasattr(torch, "_torch_compat_install_context"), (
            "oracle resolved to the Jittor shim"
        )
        if args.device == "npu":
            import torch_npu  # noqa: F401

            torch.npu.set_device(local_rank)
            device = torch.device("npu", local_rank)

            def synchronize():
                torch.npu.synchronize()

            def reset_peak_memory():
                torch.npu.reset_peak_memory_stats()

            def peak_memory():
                return int(torch.npu.max_memory_allocated())
        elif args.device == "cuda":
            assert torch.cuda.is_available(), "official PyTorch CUDA unavailable"
            torch.cuda.set_device(local_rank)
            assert torch.cuda.current_device() == local_rank, "CUDA device selection failed"
            device = torch.device("cuda", local_rank)

            def synchronize():
                torch.cuda.synchronize()

            def reset_peak_memory():
                torch.cuda.reset_peak_memory_stats(local_rank)

            def peak_memory():
                peak = int(torch.cuda.max_memory_allocated(local_rank))
                assert peak > 0, "CUDA allocator peak is unavailable or empty"
                return peak
        else:
            device = torch.device("cpu")

            def synchronize():
                return None

            def reset_peak_memory():
                return None

            def peak_memory():
                # Linux VmHWM includes model loading and is not allocator-only.
                # Keep the field explicit so it cannot be compared to an NPU
                # allocator peak under the same label.
                try:
                    for line in Path("/proc/self/status").read_text().splitlines():
                        if line.startswith("VmHWM:"):
                            return int(line.split()[1]) * 1024
                except OSError:
                    pass
                return None

        runtime_version = torch.__version__
    else:
        import jittor as jt_module
        from jittor._runtime.fallback import forbid_backend_fallbacks
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner

        jt = jt_module
        assert namespace_owner(torch) is jt and activation_status(jt).active
        if args.device == "npu":
            assert getattr(jt.compiler, "has_acl", 0), "ACL was not detected"
            guard.enter_context(jt.runtime.scope(use_cuda=1, use_acl=1))
        elif args.device == "cuda":
            assert jt.compiler.has_cuda, "Jittor CUDA unavailable"
            assert not any(getattr(jt.compiler, name, False) for name in
                           ("has_acl", "has_rocm", "has_corex")), "build is not native CUDA"
            guard.enter_context(jt.runtime.scope(use_cuda=1))
            torch.cuda.set_device(local_rank)
            assert torch.cuda.current_device() == local_rank, "CUDA device selection failed"
        else:
            guard.enter_context(jt.runtime.scope(use_cuda=0))
        guard.enter_context(jt.runtime.scope(backend_fallback="error"))
        assert world_size == jt.world_size and rank == jt.rank
        device = None

        def synchronize():
            jt.sync_all(args.device != "cpu")

        def reset_peak_memory():
            if args.device == "cuda":
                torch.cuda.reset_peak_memory_stats(local_rank)
            elif args.device == "npu":
                torch.cuda.reset_peak_memory_stats()

        def peak_memory():
            if args.device == "cuda":
                peak = int(torch.cuda.max_memory_allocated(local_rank))
                assert peak > 0, "CUDA allocator peak is unavailable or empty"
                return peak
            if args.device == "npu":
                return int(torch.cuda.max_memory_allocated())
            try:
                for line in Path("/proc/self/status").read_text().splitlines():
                    if line.startswith("VmHWM:"):
                        return int(line.split()[1]) * 1024
            except OSError:
                pass
            return None

        runtime_version = getattr(torch, "__torch_version__", torch.__version__)
        fallback_before = jt.core.backend_fallback_count()
        guard.enter_context(forbid_backend_fallbacks())

    def runtime_conditions():
        if args.device != "cuda":
            return None
        conditions = {
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        }
        assert all(conditions[name] is False for name in
                   ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark")), \
            "CUDA precision policy was not applied or changed"
        assert conditions["float32_matmul_precision"] == "highest", \
            "CUDA precision policy was not applied or changed"
        return conditions

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        runtime_conditions()

    def check_location(value):
        if jt is not None:
            assert bool(jt.flags.use_cuda) == (args.device != "cpu"), \
                "Jittor device dispatch changed during benchmark"
            native = getattr(value, "_var", value)
            location = native.location()
            assert location == ("cpu" if args.device == "cpu" else "device"), \
                "tensor left requested backend"
            if args.device == "cuda":
                assert torch.cuda.current_device() == local_rank, "CUDA device changed"
                assert native.device_id == local_rank, "tensor on wrong CUDA device"
        else:
            location = str(value.device)
            assert value.device.type == args.device, "tensor left requested backend"
            if args.device == "cuda":
                assert value.device.index == local_rank, "tensor on wrong CUDA device"
        return location

    def check_fallbacks():
        if jt is None:
            return None
        delta = int(jt.core.backend_fallback_count() - fallback_before)
        assert delta == 0, "backend fallback occurred during benchmark"
        return delta

    return SimpleNamespace(
        jt=jt, device=device, version=runtime_version,
        synchronize=synchronize, reset_peak_memory=reset_peak_memory,
        peak_memory=peak_memory, check_location=check_location,
        check_fallbacks=check_fallbacks, runtime_conditions=runtime_conditions,
    )


def main():
    process_started = time.perf_counter()
    args = parse_args()
    performance_evidence = performance_context(args.device)
    native_prefix = "JT_NCCL_" if args.device == "cuda" else "JT_HCCL_"
    rank = int(os.environ.get("RANK", os.environ.get(native_prefix + "RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get(native_prefix + "LOCAL_RANK", rank)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get(native_prefix + "WORLD_SIZE", "1")))
    guard = contextlib.ExitStack()
    runtime = initialize_runtime(args, guard, rank, local_rank, world_size)
    jt, device = runtime.jt, runtime.device
    synchronize = runtime.synchronize
    reset_peak_memory = runtime.reset_peak_memory
    peak_memory = runtime.peak_memory
    runtime_version = runtime.version

    import torch.distributed as dist
    from transformers import AutoModelForCausalLM, AutoTokenizer

    distributed = world_size > 1
    distributed_init_s = 0.0
    if distributed:
        started = time.perf_counter()
        dist.init_process_group(
            backend=distributed_backend(args.device),
            rank=rank, world_size=world_size
        )
        distributed_init_s = time.perf_counter() - started

    torch.manual_seed(20260922)
    np.random.seed(20260922)

    blocked_optional_packages = frozenset(
        item.strip() for item in
        os.environ.get("VERL_CPU_BLOCK_OPTIONAL_PACKAGES", "").split(",")
        if item.strip()
    )
    if blocked_optional_packages:
        if args.device != "cpu":
            raise RuntimeError("optional-package blocking is only valid for CPU runs")
        original_find_spec = importlib.util.find_spec

        def filtered_find_spec(name, package=None):
            if name.split(".", 1)[0] in blocked_optional_packages:
                return None
            return original_find_spec(name, package)

        importlib.util.find_spec = filtered_find_spec

    materialized_init = False
    if args.framework == "jittor":
        if os.environ.get("JITTOR_TRANSFORMERS_MATERIALIZED_INIT") != "1":
            raise RuntimeError(
                "Jittor real-model loading requires "
                "JITTOR_TRANSFORMERS_MATERIALIZED_INIT=1"
            )
        from transformers import PreTrainedModel
        descriptor = vars(PreTrainedModel).get("get_init_context")
        materialized_init = getattr(
            getattr(descriptor, "__func__", None),
            "_jittor_transformers_materialized_init", False
        ) is True
        if not materialized_init:
            raise RuntimeError("Transformers materialized-loading adapter is inactive")

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False
    )
    requested_dtype = getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=requested_dtype,
        attn_implementation="eager",
    )
    # Nested multimodal configs such as Qwen3.5 may retain the text submodel's
    # checkpoint dtype even when ``from_pretrained(dtype=...)`` is explicit.
    # Cast after load so the recorded benchmark dtype is the actual parameter
    # dtype on both runtimes, not merely the caller's request.
    model = model.to(dtype=requested_dtype)
    if hasattr(model.config, "dtype"):
        model.config.dtype = requested_dtype
    if args.framework == "torch":
        model = model.to(device)
    elif args.device != "cpu":
        # The accelerator runtime scope selects where operations execute, but
        # it must not be used as a substitute for explicit module placement.
        # Device evidence requires parameters and outputs on the real backend.
        model = model.cuda()
    synchronize()
    model_load_s = time.perf_counter() - load_started
    parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    if len(parameter_dtypes) != 1 or not parameter_dtypes[0].endswith("." + args.dtype):
        raise AssertionError("model parameters did not honor requested dtype: " + repr(parameter_dtypes))
    qwen35_functional_fallback = False
    if "qwen3_5" in type(model).__module__:
        from transformers.models.qwen3_5 import modeling_qwen3_5
        qwen35_functional_fallback = all(
            getattr(getattr(modeling_qwen3_5, name),
                    "_jittor_transformers_qwen35_functional", False)
            for name in ("torch_chunk_gated_delta_rule",
                         "torch_recurrent_gated_delta_rule")
        )
        if args.framework == "jittor" and not qwen35_functional_fallback:
            raise RuntimeError("Qwen3.5 autograd-safe fallback adapter is inactive")

    ddp_wrap_s = 0.0
    if distributed and args.task == "train":
        from torch.nn.parallel import DistributedDataParallel

        started = time.perf_counter()
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        synchronize()
        ddp_wrap_s = time.perf_counter() - started

    def to_accelerator(value):
        if args.framework == "torch":
            return value.to(device)
        if args.device != "cpu":
            return value.cuda()
        return value

    def barrier():
        if distributed:
            dist.barrier()

    def value_location(value):
        return runtime.check_location(value)

    first_parameter = next(model.parameters())
    result_parameter_location = value_location(first_parameter)

    result = {
        **performance_evidence,
        "scope": ("real-model actor-only fixed-advantage PPO update; no learned critic, "
                  "rollout server or Ray trainer" if args.task == "train" and args.objective == "ppo" else
                  "real-model causal training/inference microbenchmark"),
        "framework": args.framework,
        "task": args.task,
        "objective": args.objective,
        "device": args.device,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "distributed_backend": distributed_backend(args.device) if distributed else None,
        "memory_measurement_scope": ("Linux process VmHWM including model load" if args.device == "cpu"
                                     else "runtime allocator peak after warmup; not driver total memory"),
        "model": str(Path(args.model).resolve()),
        "parameter_count": parameter_count,
        "dtype": args.dtype,
        "parameter_dtypes": parameter_dtypes,
        "attention": "eager",
        "learning_rate": args.learning_rate,
        "adam_eps": args.adam_eps,
        "blocked_optional_packages": sorted(blocked_optional_packages),
        "cpu_thread_env": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")
        } if args.device == "cpu" else None,
        "torch_api_version": runtime_version,
        "transformers_version": __import__("transformers").__version__,
        "transformers_materialized_init": materialized_init,
        "qwen35_functional_fallback": qwen35_functional_fallback,
        "distributed_init_s": distributed_init_s,
        "model_load_s": model_load_s,
        "ddp_wrap_s": ddp_wrap_s,
        "parameter_location": result_parameter_location,
    }

    with guard:
        if args.task == "train":
            model.train()
            base_text = TRAIN_TEXTS[rank % len(TRAIN_TEXTS)]
            encoded = tokenizer(
                base_text * 32,
                max_length=args.seq_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = to_accelerator(encoded["input_ids"].repeat(args.batch_size, 1))
            attention_mask = to_accelerator(
                encoded["attention_mask"].repeat(args.batch_size, 1)
            )
            labels = input_ids.clone()
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.learning_rate, eps=args.adam_eps)

            old_log_prob = None
            advantages = None
            response_mask = None
            compute_policy_loss = None
            if args.objective == "ppo":
                verl_site_packages = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
                if verl_site_packages and verl_site_packages not in sys.path:
                    # Delay this until Transformers has materialized the text
                    # model.  The shared dependency environment also contains
                    # shim-only optional packages such as torchvision, which
                    # must never participate in the official oracle.
                    sys.path.append(verl_site_packages)
                from verl.trainer.ppo.core_algos import compute_policy_loss

                with torch.no_grad():
                    baseline_logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                    baseline_log_probs = torch.log_softmax(
                        baseline_logits[:, :-1, :], dim=-1
                    )
                    old_log_prob = torch.gather(
                        baseline_log_probs,
                        -1,
                        input_ids[:, 1:].unsqueeze(-1),
                    ).squeeze(-1).detach()
                advantages = torch.ones_like(old_log_prob)
                response_mask = attention_mask[:, 1:].to(dtype=old_log_prob.dtype)
                synchronize()
                result["initial_old_log_prob"] = old_log_prob.detach().cpu().tolist()
                result["initial_old_log_prob_dtype"] = str(old_log_prob.dtype)

            def train_step(validate_location=False):
                optimizer.zero_grad(set_to_none=True)
                selected_log_prob = None
                if args.objective == "ppo":
                    logits = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    ).logits
                    token_log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                    log_prob = torch.gather(
                        token_log_probs,
                        -1,
                        input_ids[:, 1:].unsqueeze(-1),
                    ).squeeze(-1)
                    loss, _, _, _ = compute_policy_loss(
                        old_log_prob,
                        log_prob,
                        advantages,
                        response_mask,
                        cliprange=0.2,
                    )
                    selected_log_prob = log_prob
                else:
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                        use_cache=False,
                    )
                    loss = output.loss
                forward_location = None
                if validate_location:
                    synchronize()
                    forward_location = value_location(loss)
                loss.backward()
                gradient_samples = None
                parameter_samples = None
                if validate_location:
                    gradient_samples = {}
                    parameter_samples = {}
                    missing_gradient_samples = []
                    for name, parameter in model.named_parameters():
                        normalized = name.removeprefix("module.")
                        if normalized not in GRADIENT_SAMPLE_NAMES:
                            continue
                        gradient = parameter.grad
                        if gradient is None:
                            missing_gradient_samples.append(normalized)
                            parameter_samples[normalized] = (
                                parameter.detach().float().cpu().numpy().astype("float64").tolist()
                            )
                            continue
                        gradient_samples[normalized] = (
                            gradient.detach().float().cpu().numpy().astype("float64").tolist()
                        )
                        parameter_samples[normalized] = (
                            parameter.detach().float().cpu().numpy().astype("float64").tolist()
                        )
                    if set(gradient_samples) | set(missing_gradient_samples) != GRADIENT_SAMPLE_NAMES:
                        raise AssertionError("sampled PPO parameters were not found")
                optimizer.step()
                synchronize()
                update_samples = None
                if validate_location:
                    update_samples = {}
                    for name, parameter in model.named_parameters():
                        normalized = name.removeprefix("module.")
                        if normalized in parameter_samples:
                            before = np.asarray(parameter_samples[normalized])
                            after = parameter.detach().float().cpu().numpy().astype("float64")
                            update_samples[normalized] = (after - before).tolist()
                return (loss, forward_location, selected_log_prob,
                        gradient_samples, update_samples,
                        missing_gradient_samples if validate_location else None)

            barrier()
            cold_started = time.perf_counter()
            (cold_loss, cold_forward_location, cold_log_prob,
             cold_gradient_samples, cold_update_samples,
             cold_missing_gradient_samples) = train_step(validate_location=True)
            cold_step_s = time.perf_counter() - cold_started
            cold_loss_value = float(cold_loss.item())
            for _ in range(max(0, args.warmup - 1)):
                barrier()
                train_step()
            reset_peak_memory()
            step_times = []
            losses = []
            timed_log_probs = []
            for _ in range(args.steps):
                barrier()
                started = time.perf_counter()
                loss, _, selected_log_prob, _, _, _ = train_step()
                step_times.append(time.perf_counter() - started)
                losses.append(float(loss.item()))
                if selected_log_prob is not None:
                    timed_log_probs.append(selected_log_prob.detach().cpu().tolist())
            local_tokens = args.batch_size * (
                args.seq_len - 1 if args.objective == "ppo" else args.seq_len
            )
            result.update(
                {
                    "batch_size_per_rank": args.batch_size,
                    "seq_len": args.seq_len,
                    "warmup_steps": args.warmup,
                    "timed_steps": args.steps,
                    "cold_step_s": cold_step_s,
                    "cold_loss": cold_loss_value,
                    "step_times_s": step_times,
                    "step_median_s": statistics.median(step_times),
                    "step_p90_s": percentile(step_times, 0.9),
                    "train_tokens_per_s_per_rank": local_tokens
                    / statistics.median(step_times),
                    "losses": losses,
                    "loss_delta": losses[-1] - losses[0],
                    "peak_memory_bytes": peak_memory(),
                    "cold_gradient_samples": cold_gradient_samples,
                    "cold_update_samples": cold_update_samples,
                    "cold_missing_gradient_samples": cold_missing_gradient_samples,
                }
            )
            if cold_log_prob is not None:
                result["cold_log_prob"] = cold_log_prob.detach().cpu().tolist()
                result["timed_log_probs"] = timed_log_probs
            result["output_location"] = cold_forward_location
        else:
            model.eval()
            prompt = "请简要说明参数高效微调的用途。"
            encoded = tokenizer(
                prompt,
                max_length=args.seq_len,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = to_accelerator(encoded["input_ids"])
            attention_mask = to_accelerator(encoded["attention_mask"])

            def prefill_once():
                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                    )
                synchronize()
                return output.logits

            def generate_once():
                with torch.no_grad():
                    output = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.new_tokens,
                        min_new_tokens=args.new_tokens,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                synchronize()
                return output

            barrier()
            cold_started = time.perf_counter()
            generated = generate_once()
            cold_generate_s = time.perf_counter() - cold_started
            for _ in range(max(0, args.warmup - 1)):
                barrier()
                generate_once()
            # generate() need not compile the same graph as a standalone
            # prefill. Keep its first call and subsequent warmups out of the
            # timed samples, without changing the model/cache options.
            barrier()
            synchronize()
            cold_prefill_started = time.perf_counter()
            prefill_once()
            cold_prefill_s = time.perf_counter() - cold_prefill_started
            # Unlike the separate cold call, these are additional warmups.
            prefill_warmup_runs = max(0, args.warmup)
            for _ in range(prefill_warmup_runs):
                barrier()
                prefill_once()
            reset_peak_memory()
            prefill_times = []
            total_times = []
            for _ in range(args.steps):
                barrier()
                synchronize()
                started = time.perf_counter()
                logits = prefill_once()
                prefill_times.append(time.perf_counter() - started)
                barrier()
                started = time.perf_counter()
                generated = generate_once()
                total_times.append(time.perf_counter() - started)
            output_tokens = generated[0][input_ids.shape[-1]:]
            token_list = output_tokens.detach().cpu().tolist()
            result.update(
                {
                    "prompt": prompt,
                    "prompt_tokens": int(input_ids.shape[-1]),
                    "new_tokens": args.new_tokens,
                    "warmup_runs": args.warmup,
                    "timed_runs": args.steps,
                    "cold_generate_s": cold_generate_s,
                    "cold_prefill_s": cold_prefill_s,
                    "prefill_cold_runs": 1,
                    "prefill_warmup_runs": prefill_warmup_runs,
                    "prefill_times_s": prefill_times,
                    "prefill_median_s": statistics.median(prefill_times),
                    "generate_times_s": total_times,
                    "generate_median_s": statistics.median(total_times),
                    "generate_p90_s": percentile(total_times, 0.9),
                    "generated_tokens_per_s_per_rank": args.new_tokens
                    / statistics.median(total_times),
                    "generated_token_ids": token_list,
                    "generated_text": tokenizer.decode(
                        token_list, skip_special_tokens=True
                    ),
                    "peak_memory_bytes": peak_memory(),
                }
            )
            result["output_location"] = value_location(logits)

    if jt is not None:
        result["fallback_delta"] = runtime.check_fallbacks()
        expected_location = "device" if args.device != "cpu" else "cpu"
        assert result["parameter_location"] == expected_location
        assert result["output_location"] == expected_location
    else:
        assert result["parameter_location"].startswith(args.device)
        assert result["output_location"].startswith(args.device)
    if args.device == "cuda":
        result["runtime_conditions"] = runtime.runtime_conditions()

    result["process_wall_s"] = time.perf_counter() - process_started

    output_path = Path("{}.rank{}.json".format(args.output, rank))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    console_result = {
        key: value for key, value in result.items()
        if key not in ("cold_gradient_samples", "cold_update_samples")
    }
    console_result["cold_gradient_sample_names"] = (
        sorted(result.get("cold_gradient_samples", ()))
    )
    print("LLM_BENCH_PASS", json.dumps(console_result, ensure_ascii=False), flush=True)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
