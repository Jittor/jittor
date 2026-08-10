#!/usr/bin/env python3
"""Low-memory forward/backward benchmarks for transformer CUDA hotspots."""

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


def _setup_env() -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("REAL_HOME", os.environ.get("HOME", str(pathlib.Path.home())))
    os.environ.setdefault("JITTOR_TORCH_PROJECT_ROOT", str(WORKDIR))
    os.environ.setdefault("JITTOR_TORCH_RUNTIME_ROOT", str(RUNTIME / "jittor"))
    os.environ.setdefault("HF_HOME", str(RUNTIME / "hf_home"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "8")


def _import_backend(name: str):
    if name == "torch":
        import torch  # type: ignore

        if "jittor" in str(getattr(torch, "__file__", "")):
            raise RuntimeError("the torch benchmark imported the Jittor shim, not real PyTorch")
        return torch
    sys.path.insert(0, str(ROOT / "python"))
    import jittor as torch  # type: ignore

    sys.modules["torch"] = torch
    torch.flags.use_cuda = 1
    return torch


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


def _dtype(torch, name: str):
    return getattr(torch, name)


def _tensor(torch, rng: np.random.Generator, shape, dtype, requires_grad=False):
    arr = rng.standard_normal(shape).astype("float32")
    out = torch.tensor(arr, device=torch.device("cuda"), dtype=dtype)
    if requires_grad:
        out.requires_grad_(True)
    return out


def _slot(torch, case: str, dtype, seed: int, backward: bool):
    rng = np.random.default_rng(seed)
    req = bool(backward)
    if case in ("relu", "gelu", "softmax"):
        shape = (4096, 1024)
        out = {"x": _tensor(torch, rng, shape, dtype, req)}
        if backward:
            out["go"] = _tensor(torch, rng, shape, dtype)
        return out
    if case == "layernorm":
        shape = (2048, 768)
        out = {
            "x": _tensor(torch, rng, shape, dtype, req),
            "weight": _tensor(torch, rng, (768,), dtype, req),
            "bias": _tensor(torch, rng, (768,), dtype, req),
        }
        if backward:
            out["go"] = _tensor(torch, rng, shape, dtype)
        return out
    if case == "sdpa":
        qshape = (4, 12, 128, 64)
        out = {
            "q": _tensor(torch, rng, qshape, dtype, req),
            "k": _tensor(torch, rng, qshape, dtype, req),
            "v": _tensor(torch, rng, qshape, dtype, req),
        }
        if backward:
            out["go"] = _tensor(torch, rng, qshape, dtype)
        return out
    if case == "mlp":
        xshape = (4, 128, 768)
        out = {
            "x": _tensor(torch, rng, xshape, dtype, req),
            "w1": _tensor(torch, rng, (3072, 768), dtype, req),
            "b1": _tensor(torch, rng, (3072,), dtype, req),
            "w2": _tensor(torch, rng, (768, 3072), dtype, req),
            "b2": _tensor(torch, rng, (768,), dtype, req),
        }
        if backward:
            out["go"] = _tensor(torch, rng, xshape, dtype)
        return out
    if case == "transformer_block":
        batch, seq, hidden, heads = 2, 128, 768, 12
        xshape = (batch, seq, hidden)
        params = {
            "x": _tensor(torch, rng, xshape, dtype, req),
            "ln1w": _tensor(torch, rng, (hidden,), dtype, req),
            "ln1b": _tensor(torch, rng, (hidden,), dtype, req),
            "wqkv": _tensor(torch, rng, (3 * hidden, hidden), dtype, req),
            "bqkv": _tensor(torch, rng, (3 * hidden,), dtype, req),
            "wo": _tensor(torch, rng, (hidden, hidden), dtype, req),
            "bo": _tensor(torch, rng, (hidden,), dtype, req),
            "ln2w": _tensor(torch, rng, (hidden,), dtype, req),
            "ln2b": _tensor(torch, rng, (hidden,), dtype, req),
            "w1": _tensor(torch, rng, (4 * hidden, hidden), dtype, req),
            "b1": _tensor(torch, rng, (4 * hidden,), dtype, req),
            "w2": _tensor(torch, rng, (hidden, 4 * hidden), dtype, req),
            "b2": _tensor(torch, rng, (hidden,), dtype, req),
        }
        if backward:
            params["go"] = _tensor(torch, rng, xshape, dtype)
        params["heads"] = heads
        return params
    raise ValueError(case)


def _sdpa(torch, backend: str, mode: str, q, k, v):
    if mode == "default" or backend == "jittor":
        if mode == "flash" and backend == "jittor":
            raise RuntimeError("Jittor native flash SDPA is not available for this training benchmark")
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    selected = SDPBackend.MATH if mode == "math" else SDPBackend.FLASH_ATTENTION
    with sdpa_kernel(selected):
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=0.0)


def _forward(torch, backend: str, sdpa_backend: str, case: str, s: dict[str, Any]):
    F = torch.nn.functional
    if case == "relu":
        return F.relu(s["x"])
    if case == "gelu":
        return F.gelu(s["x"])
    if case == "softmax":
        return F.softmax(s["x"], dim=-1)
    if case == "layernorm":
        return F.layer_norm(s["x"], (768,), s["weight"], s["bias"], 1e-5)
    if case == "sdpa":
        return _sdpa(torch, backend, sdpa_backend, s["q"], s["k"], s["v"])
    if case == "mlp":
        return F.linear(F.gelu(F.linear(s["x"], s["w1"], s["b1"])), s["w2"], s["b2"])
    if case == "transformer_block":
        x = s["x"]
        batch, seq, hidden = x.shape
        heads = s["heads"]
        head_dim = hidden // heads
        h = F.layer_norm(x, (hidden,), s["ln1w"], s["ln1b"], 1e-5)
        qkv = F.linear(h, s["wqkv"], s["bqkv"])
        qkv = qkv.reshape(batch, seq, 3, heads, head_dim).permute(2, 0, 3, 1, 4)
        attn = _sdpa(torch, backend, sdpa_backend, qkv[0], qkv[1], qkv[2])
        attn = attn.permute(0, 2, 1, 3).reshape(batch, seq, hidden)
        x = x + F.linear(attn, s["wo"], s["bo"])
        h = F.layer_norm(x, (hidden,), s["ln2w"], s["ln2b"], 1e-5)
        h = F.linear(F.gelu(F.linear(h, s["w1"], s["b1"])), s["w2"], s["b2"])
        return x + h
    raise ValueError(case)


def _grad_targets(case: str, s: dict[str, Any]):
    if case in ("relu", "gelu", "softmax"):
        return (s["x"],)
    if case == "layernorm":
        return s["x"], s["weight"], s["bias"]
    if case == "sdpa":
        return s["q"], s["k"], s["v"]
    return tuple(v for k, v in s.items() if k not in ("go", "heads"))


def _run_one(torch, backend: str, sdpa_backend: str, case: str,
             s: dict[str, Any], backward: bool):
    out = _forward(torch, backend, sdpa_backend, case, s)
    if not backward:
        return (out,)
    loss = (out * s["go"]).sum()
    grads = torch.autograd.grad(loss, _grad_targets(case, s), retain_graph=False)
    return (out,) + tuple(grads)


def _checksum(value) -> float:
    if isinstance(value, (tuple, list)):
        return sum(_checksum(v) for v in value)
    try:
        return float(value.float().sum().detach().cpu().numpy().reshape(-1)[0])
    except Exception:
        return float(value.float32().sum().numpy().reshape(-1)[0])


def _as_float32_numpy(value) -> np.ndarray:
    try:
        return np.asarray(value.detach().float().cpu().numpy(), dtype="float32")
    except Exception:
        return np.asarray(value.float32().numpy(), dtype="float32")


def _component_stats(values) -> list[dict[str, float]]:
    stats = []
    for value in values:
        arr = _as_float32_numpy(value)
        stats.append({
            "sum": float(arr.sum(dtype=np.float64)),
            "abs_sum": float(np.abs(arr).sum(dtype=np.float64)),
            "abs_max": float(np.abs(arr).max()) if arr.size else 0.0,
            "finite": bool(np.isfinite(arr).all()),
        })
    return stats


def _memory(torch):
    out = {}
    for name in ("memory_allocated", "max_memory_allocated", "memory_reserved", "max_memory_reserved"):
        fn = getattr(getattr(torch, "cuda", None), name, None)
        if callable(fn):
            try:
                out[name] = int(fn())
            except Exception:
                pass
    return out


def _project_file(path: str) -> pathlib.Path:
    out = pathlib.Path(path)
    if not out.is_absolute():
        out = WORKDIR / out
    out = out.resolve()
    try:
        out.relative_to(WORKDIR)
    except ValueError as exc:
        raise SystemExit(f"output must stay under project root: {out}") from exc
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jittor"), required=True)
    parser.add_argument("--case", choices=("relu", "gelu", "softmax", "layernorm", "sdpa", "mlp", "transformer_block"), required=True)
    parser.add_argument("--phase", choices=("forward", "fwd_bwd"), default="forward")
    parser.add_argument("--dtype", choices=("float64", "float32", "float16", "bfloat16"),
                        default="float32")
    parser.add_argument("--tf32", choices=("on", "off"), default="off")
    parser.add_argument("--sdpa-backend", choices=("default", "math", "flash"),
                        default="default")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--jsonl", default="results/training_hotspots.jsonl")
    parser.add_argument("--dump-npz")
    args = parser.parse_args(argv)

    _setup_env()
    torch = _import_backend(args.backend)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{args.backend} CUDA is unavailable")
    _set_tf32(torch, args.backend, args.tf32 == "on")
    dtype = _dtype(torch, args.dtype)
    backward = args.phase == "fwd_bwd"
    slots = [_slot(torch, args.case, dtype, 20260710 + i, backward)
             for i in range(args.repeats)]
    _sync(torch)

    no_grad = torch.no_grad if not backward else contextlib.nullcontext
    with no_grad():
        # Touch every timed slot before measurement. Jittor's dual allocator can
        # otherwise charge host-to-device page migration to the first use of each
        # distinct input, which measures cold residency rather than the operator.
        warm = []
        for i in range(max(args.warmup, len(slots))):
            warm.append(_run_one(torch, args.backend, args.sdpa_backend, args.case,
                                 slots[i % len(slots)], backward))
        _sync(torch)
        del warm
        gc.collect()

        reset_peak = getattr(getattr(torch, "cuda", None), "reset_peak_memory_stats", None)
        if callable(reset_peak):
            try:
                reset_peak()
            except Exception:
                pass
        before_mem = _memory(torch)
        kept = []
        start = time.perf_counter()
        for s in slots:
            kept.append(_run_one(torch, args.backend, args.sdpa_backend, args.case,
                                 s, backward))
        _sync(torch)
        latency_ms = (time.perf_counter() - start) * 1000.0 / args.repeats
        after_mem = _memory(torch)

    components = _component_stats(kept[-1])
    if backward:
        grad_stats = components[1:]
        if not grad_stats or any(stat["abs_max"] == 0.0 for stat in grad_stats):
            raise RuntimeError(f"{args.case} produced a missing or zero gradient: {grad_stats}")
        if not all(stat["finite"] for stat in grad_stats):
            raise RuntimeError(f"{args.case} produced a non-finite gradient: {grad_stats}")

    row = {
        "backend": args.backend,
        "backend_file": str(getattr(torch, "__file__", "")),
        "version": str(getattr(torch, "__version__", "")),
        "case": args.case,
        "phase": args.phase,
        "dtype": args.dtype,
        "tf32": args.tf32 == "on",
        "sdpa_backend": args.sdpa_backend,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "latency_ms": latency_ms,
        "checksum": _checksum(kept[-1]),
        "components": components,
        "memory_before": before_mem,
        "memory_after": after_mem,
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    out = _project_file(args.jsonl)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")

    if args.dump_npz:
        dump = _project_file(args.dump_npz)
        np.savez_compressed(
            dump, **{f"component_{i}": _as_float32_numpy(v)
                     for i, v in enumerate(kept[-1])})

    kept.clear()
    slots.clear()
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
