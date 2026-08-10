#!/usr/bin/env python3
"""Benchmark the same Transformers 4.56.2 tiny models on Jittor and PyTorch."""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np


from _paths import REPO_ROOT as ROOT, RUNTIME_ROOT as RUNTIME, WORK_ROOT as WORKDIR
JT_SITE = pathlib.Path(
    "/home/zy/miniconda3/envs/jt311/lib/python3.11/site-packages"
).resolve()
RT_SITE = pathlib.Path("/home/zy/rt_venv/lib/python3.11/site-packages").resolve()
EXPECTED_TRANSFORMERS_VERSION = "4.56.2"


def _setup_env() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("REAL_HOME", os.environ.get("HOME", str(pathlib.Path.home())))
    os.environ.setdefault("JITTOR_TORCH_PROJECT_ROOT", str(WORKDIR))
    os.environ.setdefault("JITTOR_TORCH_RUNTIME_ROOT", str(RUNTIME / "hf_tiny"))
    os.environ.setdefault("HF_HOME", str(RUNTIME / "hf_home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", "8")


def _prepend_path(path: pathlib.Path) -> None:
    value = str(path)
    sys.path[:] = [entry for entry in sys.path if entry != value]
    sys.path.insert(0, value)


def _import_stack(backend: str):
    if backend == "torch":
        _prepend_path(RT_SITE)
        import torch  # type: ignore
        import torchvision  # type: ignore

        torch_file = pathlib.Path(str(getattr(torch, "__file__", ""))).resolve()
        if RT_SITE not in torch_file.parents or "jittor" in str(torch_file).lower():
            raise RuntimeError(f"expected real PyTorch from {RT_SITE}, got {torch_file}")
        torchvision_file = pathlib.Path(str(getattr(torchvision, "__file__", ""))).resolve()
        if RT_SITE not in torchvision_file.parents or "jittor" in str(torchvision_file).lower():
            raise RuntimeError(
                f"expected real torchvision from {RT_SITE}, got {torchvision_file}"
            )
        # torch is now pinned in sys.modules. Put jt311 first only for Transformers
        # and its Python dependencies; this cannot replace the already-loaded
        # torch/torchvision modules.
        _prepend_path(JT_SITE)
    else:
        _prepend_path(ROOT / "python")
        import jittor as torch  # type: ignore

        sys.modules["torch"] = torch
        torch.flags.use_cuda = 1
        _prepend_path(JT_SITE)

    import transformers  # type: ignore

    transformers_file = pathlib.Path(str(transformers.__file__)).resolve()
    if transformers.__version__ != EXPECTED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"expected transformers {EXPECTED_TRANSFORMERS_VERSION}, "
            f"got {transformers.__version__} from {transformers_file}"
        )
    if JT_SITE not in transformers_file.parents:
        raise RuntimeError(f"expected Transformers from {JT_SITE}, got {transformers_file}")
    if backend == "jittor" and "python/jittor" not in str(pathlib.Path(torch.__file__).resolve()):
        raise RuntimeError(f"expected repository Jittor, got {torch.__file__}")
    return torch, transformers


def _sync(torch) -> None:
    try:
        torch.cuda.synchronize()
    except Exception:
        import jittor as jt

        jt.sync_all(True)


def _set_tf32(torch, backend: str, enabled: bool) -> None:
    if hasattr(torch, "backends") and hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = enabled
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    if backend == "jittor":
        torch.flags.cuda_allow_tf32 = int(enabled)


def _set_attention(config, implementation: str) -> None:
    config._attn_implementation = implementation


def _build_model(transformers, name: str, attention: str):
    if name == "llama":
        config = transformers.LlamaConfig(
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=768,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,
            max_position_embeddings=256,
            attention_dropout=0.0,
            hidden_act="silu",
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
            use_cache=False,
        )
        _set_attention(config, attention)
        model = transformers.LlamaModel(config)
        expected_shape = (2, 128, 256)
    elif name == "bert":
        config = transformers.BertConfig(
            vocab_size=2048,
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            max_position_embeddings=256,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            pad_token_id=0,
        )
        _set_attention(config, attention)
        model = transformers.BertModel(config, add_pooling_layer=False)
        expected_shape = (2, 128, 256)
    elif name == "vit":
        config = transformers.ViTConfig(
            image_size=64,
            patch_size=8,
            num_channels=3,
            hidden_size=256,
            intermediate_size=1024,
            num_hidden_layers=2,
            num_attention_heads=8,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )
        _set_attention(config, attention)
        model = transformers.ViTModel(config, add_pooling_layer=False)
        expected_shape = (2, 65, 256)
    else:
        raise ValueError(name)
    model.eval()
    return model, config, expected_shape


def _tensor(torch, array: np.ndarray, dtype):
    return torch.tensor(array, device=torch.device("cuda"), dtype=dtype)


def _make_slot(torch, name: str, seed: int, backward: bool) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if name in ("llama", "bert"):
        slot = {
            "inputs": {
                "input_ids": _tensor(
                    torch, rng.integers(1, 2048, size=(2, 128), dtype=np.int64), torch.long
                ),
                "attention_mask": _tensor(torch, np.ones((2, 128), dtype=np.int64), torch.long),
            }
        }
        output_shape = (2, 128, 256)
        if name == "llama":
            slot["inputs"]["use_cache"] = False
    else:
        slot = {
            "inputs": {
                "pixel_values": _tensor(
                    torch, rng.standard_normal((2, 3, 64, 64)).astype("float32"), torch.float32
                )
            }
        }
        output_shape = (2, 65, 256)
    if backward:
        slot["go"] = _tensor(
            torch, rng.standard_normal(output_shape).astype("float32"), torch.float32
        )
    return slot


def _forward(model, slot: dict[str, Any]):
    return model(**slot["inputs"], return_dict=True).last_hidden_state


def _run_one(torch, model, params, slot: dict[str, Any], backward: bool):
    output = _forward(model, slot)
    if not backward:
        return (output,)
    loss = (output * slot["go"]).sum()
    grads = torch.autograd.grad(loss, params, retain_graph=False)
    return (output,) + tuple(grads)


def _as_numpy(value) -> np.ndarray:
    try:
        return np.asarray(value.detach().float().cpu().numpy(), dtype="float32")
    except Exception:
        return np.asarray(value.float32().numpy(), dtype="float32")


def _verify_result(values, expected_shape: tuple[int, ...], backward: bool) -> dict[str, Any]:
    output = _as_numpy(values[0])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(f"expected output shape {expected_shape}, got {output.shape}")
    if not np.isfinite(output).all():
        raise RuntimeError("model output contains NaN or Inf")
    grad_max = []
    if backward:
        for index, value in enumerate(values[1:]):
            array = _as_numpy(value)
            maximum = float(np.max(np.abs(array))) if array.size else 0.0
            if not np.isfinite(array).all() or maximum == 0.0:
                raise RuntimeError(f"gradient {index} is missing, zero, or non-finite")
            grad_max.append(maximum)
    return {
        "output_shape": list(output.shape),
        "output_sum": float(output.sum(dtype=np.float64)),
        "output_abs_max": float(np.max(np.abs(output))),
        "output_finite": True,
        "gradient_count": len(grad_max),
        "gradient_abs_max_min": min(grad_max) if grad_max else None,
        "gradient_abs_max_max": max(grad_max) if grad_max else None,
    }


def _memory(torch) -> dict[str, int]:
    result = {}
    for name in (
        "memory_allocated",
        "max_memory_allocated",
        "memory_reserved",
        "max_memory_reserved",
    ):
        function = getattr(getattr(torch, "cuda", None), name, None)
        if callable(function):
            try:
                result[name] = int(function())
            except Exception:
                pass
    return result


def _project_file(path: str) -> pathlib.Path:
    output = pathlib.Path(path)
    if not output.is_absolute():
        output = WORKDIR / output
    output = output.resolve()
    try:
        output.relative_to(WORKDIR)
    except ValueError as exc:
        raise SystemExit(f"output must stay under project root: {output}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("jittor", "torch"), required=True)
    parser.add_argument("--model", choices=("llama", "bert", "vit"), required=True)
    parser.add_argument("--phase", choices=("forward", "fwd_bwd"), default="forward")
    parser.add_argument("--attention", choices=("eager", "sdpa"), default="sdpa")
    parser.add_argument("--tf32", choices=("on", "off"), default="on")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument(
        "--jsonl", default="results/hf_tiny_models.jsonl"
    )
    args = parser.parse_args(argv)

    _setup_env()
    torch, transformers = _import_stack(args.backend)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{args.backend} CUDA is unavailable")
    _set_tf32(torch, args.backend, args.tf32 == "on")

    model, config, expected_shape = _build_model(transformers, args.model, args.attention)
    if args.backend == "torch":
        model = model.to(torch.device("cuda"))
    # Jittor parameters are trainable by default, but the torch-style
    # requires_grad property is only populated after an explicit setter. Using
    # it as a filter would incorrectly drop every native Jittor model parameter.
    params = tuple(model.parameters())
    if not params:
        raise RuntimeError(f"{args.model} exposed no model parameters")
    parameter_count = sum(int(param.numel()) for param in params)
    backward = args.phase == "fwd_bwd"
    if backward:
        # Use the same explicit leaf contract on both backends. Native Jittor
        # model parameters are normally activated by an optimizer, while this
        # benchmark calls autograd.grad directly without constructing one.
        for param in params:
            param.requires_grad_(True)

    verify_slot = _make_slot(torch, args.model, 20260711, backward)
    grad_context = contextlib.nullcontext if backward else torch.no_grad
    with grad_context():
        verified = _run_one(torch, model, params, verify_slot, backward)
        _sync(torch)
    verification = _verify_result(verified, expected_shape, backward)

    latency_ms = None
    warmup_iterations = 0
    before_memory = _memory(torch)
    after_memory = before_memory
    if not args.smoke_only:
        slots = [
            _make_slot(torch, args.model, 20260720 + index, backward)
            for index in range(args.repeats)
        ]
        with grad_context():
            warm = []
            warmup_iterations = max(args.warmup, len(slots))
            for index in range(warmup_iterations):
                warm.append(_run_one(torch, model, params, slots[index % len(slots)], backward))
            _sync(torch)
            warm.clear()
            gc.collect()

            reset_peak = getattr(getattr(torch, "cuda", None), "reset_peak_memory_stats", None)
            if callable(reset_peak):
                try:
                    reset_peak()
                except Exception:
                    pass
            before_memory = _memory(torch)
            kept = []
            start = time.perf_counter()
            for slot in slots:
                kept.append(_run_one(torch, model, params, slot, backward))
            _sync(torch)
            latency_ms = (time.perf_counter() - start) * 1000.0 / args.repeats
            after_memory = _memory(torch)
        verification = _verify_result(kept[-1], expected_shape, backward)

    row = {
        "backend": args.backend,
        "backend_file": str(pathlib.Path(torch.__file__).resolve()),
        "backend_version": str(getattr(torch, "__version__", "")),
        "python": sys.executable,
        "transformers_file": str(pathlib.Path(transformers.__file__).resolve()),
        "transformers_version": transformers.__version__,
        "model": args.model,
        "model_class": type(model).__name__,
        "phase": args.phase,
        "attention": args.attention,
        "config_attention": str(config._attn_implementation),
        "tf32": args.tf32 == "on",
        "smoke_only": args.smoke_only,
        "requested_warmup": 0 if args.smoke_only else args.warmup,
        "warmup": warmup_iterations,
        "repeats": 1 if args.smoke_only else args.repeats,
        "latency_ms": latency_ms,
        "parameter_count": parameter_count,
        "config_hidden_size": int(config.hidden_size),
        "config_num_hidden_layers": int(config.num_hidden_layers),
        "memory_before": before_memory,
        "memory_after": after_memory,
        **verification,
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    output = _project_file(args.jsonl)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
