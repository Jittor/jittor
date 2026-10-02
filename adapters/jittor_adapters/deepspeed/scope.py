"""Executable boundaries for the validated eager NPU configurations."""
import inspect
import os
from functools import wraps

_ALLOWED_CONFIG = frozenset({
    "train_batch_size", "train_micro_batch_size_per_gpu", "gradient_accumulation_steps",
    "steps_per_print", "zero_optimization", "fp16", "bf16", "wall_clock_breakdown",
    "memory_breakdown", "gradient_clipping",
})


def validate_config(config):
    if not isinstance(config, dict):
        raise NotImplementedError("DeepSpeed adapter requires an explicit configuration dict")
    unknown = set(config) - _ALLOWED_CONFIG
    if unknown:
        raise NotImplementedError("Unverified DeepSpeed configuration keys: " + ", ".join(sorted(unknown)))
    zero = config.get("zero_optimization", {"stage": 0})
    if not isinstance(zero, dict):
        raise NotImplementedError("zero_optimization must be an explicit dictionary")
    stage = zero.get("stage", 0)
    if stage in (0, 3):
        allowed_zero_keys = {"stage"}
    else:
        allowed_zero_keys = {"stage", "contiguous_gradients"}
    if stage not in (0, 1, 2, 3) or set(zero) - allowed_zero_keys:
        raise NotImplementedError(
            "Only ZeRO Stage 0-3 without offload or extra ZeRO options are "
            "verified; Stage 3 uses its default contiguous-gradient mode")
    if "contiguous_gradients" in zero and not isinstance(zero["contiguous_gradients"], bool):
        raise NotImplementedError("contiguous_gradients must be boolean")
    for key in ("fp16", "bf16"):
        if key in config and config[key] != {"enabled": False}:
            raise NotImplementedError(key + " must be explicitly disabled")
    for key in ("wall_clock_breakdown", "memory_breakdown"):
        if config.get(key, False) is not False:
            raise NotImplementedError(key + " is not verified")
    if config.get("gradient_accumulation_steps", 1) != 1:
        raise NotImplementedError("Only one gradient accumulation step is verified")
    if config.get("gradient_clipping", 0) != 0:
        raise NotImplementedError("Gradient clipping is not verified")


def validate_engine(device, arguments):
    if device != "npu":
        raise NotImplementedError(
            "CPU covers import/config/model construction only; no real Gloo backend is provided")
    validate_config(arguments.get("config"))
    for key in ("training_data", "lr_scheduler", "mpu", "mesh_param", "mesh_device", "collate_fn"):
        if arguments.get(key) is not None:
            raise NotImplementedError("Unverified DeepSpeed initialization argument: " + key)
    for key in ("LOCAL_RANK", "OMPI_COMM_WORLD_LOCAL_RANK", "JT_HCCL_LOCAL_RANK"):
        if key in os.environ and os.environ[key] != "0":
            raise NotImplementedError(key + " must select logical NPU 0")
    local_rank = getattr(arguments.get("args"), "local_rank", None)
    if local_rank not in (None, -1, 0):
        raise NotImplementedError("args.local_rank must select logical NPU 0")
    import torch
    from jittor.distributed import get_hccl_world_info
    info = get_hccl_world_info()
    global_rank = int(os.environ.get("RANK", os.environ.get("JT_HCCL_RANK", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("JT_HCCL_WORLD_SIZE", "1")))
    if world_size not in (1, 2):
        raise NotImplementedError("Only one-rank and single-node two-rank HCCL WORLDs are verified")
    if arguments["config"].get("zero_optimization", {"stage": 0}).get("stage", 0) in (1, 2, 3) and world_size != 2:
        raise NotImplementedError(
            "ZeRO Stage 1/2/3 is verified only on a single-node two-rank "
            "HCCL WORLD")
    expected_world = {"initialized": True, "rank": global_rank, "world_size": world_size}
    if info != expected_world:
        raise RuntimeError("DeepSpeed adapter requires the requested initialized HCCL WORLD: %r != %r" % (info, expected_world))
    if torch.npu.current_device() != 0:
        raise NotImplementedError("Only logical NPU 0 is verified")
    model = arguments.get("model")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("DeepSpeed adapter requires an explicit torch.nn.Module")
    optimizer = arguments.get("optimizer")
    if not isinstance(optimizer, torch.optim.AdamW):
        raise NotImplementedError("Pass an explicit public torch.optim.AdamW optimizer")
    # The native AdamW stores fused on the optimizer; Torch-style flags can
    # also be specified per group. Missing/None/False defaults remain allowed.
    for group in optimizer.param_groups:
        for flag in ("foreach", "fused", "amsgrad", "capturable", "differentiable", "maximize"):
            value = group.get(flag, getattr(optimizer, flag, None))
            if value is not None and value is not False:
                raise NotImplementedError("Unverified AdamW execution mode: " + flag)
    for name, parameter in model.named_parameters():
        if parameter.dtype != torch.float32 or str(parameter.device) != "npu:0":
            raise NotImplementedError("Unverified parameter dtype/device: " + name)
    for name, buffer in model.named_buffers():
        if buffer.dtype not in (torch.float32, torch.bool) or str(buffer.device) != "npu:0":
            raise NotImplementedError("Unverified buffer dtype/device: " + name)


def guard(function, device):
    signature = inspect.signature(function)

    @wraps(function)
    def checked(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        validate_engine(device, bound.arguments)
        return function(*args, **kwargs)

    return checked
