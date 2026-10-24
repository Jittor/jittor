"""Process bootstrap for the isolated verl/Jittor validation environment."""
from __future__ import annotations

import os
import json


def _install_optional_fault_observer():
    """Task-only diagnostic opt-in; never needed for normal training."""
    mode = os.environ.get("VERL_AGENT_FAULT_DIAGNOSTIC", "0")
    if mode not in ("0", "1"):
        raise SystemExit("VERL_AGENT_FAULT_DIAGNOSTIC must be absent, 0 or 1")
    if mode == "0":
        return
    if os.environ.get("JITTOR_TORCH_SHIM") != "1":
        raise SystemExit("fault observer requires the candidate bootstrap")
    import hashlib
    import importlib
    import importlib.util
    import sys
    from pathlib import Path

    directory = os.environ.get("VERL_AGENT_FAULT_MODULE_DIR", "")
    expected = os.environ.get("VERL_AGENT_FAULT_MODULE_SHA256", "")
    path = Path(directory)
    if (not directory or not path.is_absolute() or not path.is_dir()
            or path.resolve(strict=True) != path
            or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
        raise SystemExit("fault observer requires a canonical module directory and SHA256")
    name = "verl_agent_fault_diagnostic"
    if name in sys.modules:
        raise SystemExit("fault observer must not be loaded before candidate initialization")
    source = path / (name + ".py")
    spec = importlib.util.find_spec(name)
    if (spec is None or spec.origin is None or Path(spec.origin).resolve(strict=True) != source
            or source.resolve(strict=True) != source
            or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
        raise SystemExit("fault observer module origin/SHA256 mismatch")
    observer = importlib.import_module(name)
    if not callable(getattr(observer, "install", None)) or observer.install() is not True:
        raise RuntimeError("fault observer registration failed")


def _configure_cuda_strict_fp32(torch_api):
    profile = os.environ.get("VERL_CUDA_STRICT_FP32", "0")
    if profile not in ("0", "1"):
        raise SystemExit("VERL_CUDA_STRICT_FP32 must be absent, 0 or 1")
    if profile == "0":
        return
    if (os.environ.get("VERL_DEVICE") != "cuda"
            or os.environ.get("VERL_TRAIN_DTYPE", "fp32") != "fp32"
            or os.environ.get("JITTOR_VERL_RAY_NPU_CONTEXT") == "1"):
        raise SystemExit("VERL_CUDA_STRICT_FP32 requires CUDA fp32 training")
    torch_api.backends.cuda.matmul.allow_tf32 = False
    torch_api.backends.cudnn.allow_tf32 = False
    torch_api.backends.cudnn.benchmark = False
    torch_api.set_float32_matmul_precision("highest")
    conditions = {
        "cuda_matmul_allow_tf32": torch_api.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch_api.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch_api.backends.cudnn.benchmark,
        "float32_matmul_precision": torch_api.get_float32_matmul_precision(),
    }
    if (any(conditions[key] is not False for key in
            ("cuda_matmul_allow_tf32", "cudnn_allow_tf32", "cudnn_benchmark"))
            or conditions["float32_matmul_precision"] != "highest"):
        raise RuntimeError("strict FP32 precision readback mismatch: " + repr(conditions))
    # Public settings are observable per process, not device execution or
    # zero-fallback/full-training/performance proof.
    print("VERL_CUDA_STRICT_FP32_READY " + json.dumps({
        "framework": "jittor", "pid": os.getpid(),
        "runtime_conditions": conditions, "scope": "public_precision_readback_only",
    }, sort_keys=True), flush=True)


def _initialize_candidate():
    if (os.environ.get("JITTOR_VERL_RAY_NPU_CONTEXT") == "1"
            and os.environ.get("JITTOR_VERL_RAY_CUDA_CONTEXT") == "1"):
        raise SystemExit("conflicting candidate NPU and CUDA rollout context markers")
    import torch

    reporter = getattr(torch, "compat_report_torch_api_version", None)
    if not callable(reporter):
        raise RuntimeError("Jittor Torch frontend lacks compat_report_torch_api_version()")
    reported = reporter(True)
    if reported != getattr(torch, "__torch_version__", None):
        raise RuntimeError("Jittor Torch API version bootstrap reported an inconsistent value")
    _configure_cuda_strict_fp32(torch)
    # Ray deserializes actor constructor arguments before calling __init__.  A
    # serialized verl config imports transfer_queue, whose module-level tensor
    # factories therefore need an ACL context even earlier.  Only the custom
    # rollout actor receives this marker; generic Ray workers stay CPU-only at
    # startup and initialize their context at verl Worker.__init__ instead.
    if os.environ.get("JITTOR_VERL_RAY_NPU_CONTEXT") == "1":
        torch.npu.set_device(0)
    if os.environ.get("JITTOR_VERL_RAY_CUDA_CONTEXT") == "1":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible.isdecimal():
            raise SystemExit("rollout candidate requires exactly one pre-bound CUDA device")
        if not torch.cuda.is_available():
            raise SystemExit("rollout candidate CUDA is unavailable")
        torch.cuda.set_device(0)
    _install_optional_fault_observer()


if (os.environ.get("VERL_AGENT_FAULT_DIAGNOSTIC", "0") != "0"
        and os.environ.get("JITTOR_TORCH_SHIM") != "1"):
    raise SystemExit("fault observer requires the candidate bootstrap")

if os.environ.get("JITTOR_TORCH_SHIM") == "1" and (
    os.environ.get("JITTOR_VERL_NATIVE_NPU_KERNELS") == "1"
    or os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1"
    or os.environ.get("VERL_CUDA_STRICT_FP32") == "1"
    or os.environ.get("VERL_AGENT_FAULT_DIAGNOSTIC", "0") != "0"
):
    try:
        _initialize_candidate()
    except (ImportError, OSError, RuntimeError, AttributeError, ValueError, TypeError) as error:
        # Python swallows ordinary sitecustomize exceptions and proceeds with
        # the payload. Never execute a Ray worker after failed initialization.
        raise SystemExit("candidate bootstrap failed: " + str(error)) from error
