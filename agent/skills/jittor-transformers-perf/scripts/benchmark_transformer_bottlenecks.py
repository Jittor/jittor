#!/usr/bin/env python3
"""Benchmark transformer bottlenecks on real PyTorch or jittor-as-torch."""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Any, Callable

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

        return torch
    if str(ROOT / "python") not in sys.path:
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
    if hasattr(torch, "backends") and hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = enabled
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high" if enabled else "highest")
    if backend == "jittor":
        torch.flags.cuda_allow_tf32 = int(enabled)


def _tensor_np(torch, shape: tuple[int, ...], rng: np.random.Generator, device, dtype=None):
    arr = rng.standard_normal(shape).astype("float32")
    if dtype is None:
        return torch.tensor(arr, device=device)
    return torch.tensor(arr, device=device, dtype=dtype)


def _bench(torch, fn: Callable[[int], Any], repeats: int, warmup: int) -> dict[str, float]:
    last = None
    for i in range(warmup):
        last = fn(i)
    _sync(torch)
    start = time.perf_counter()
    outputs = []
    for i in range(repeats):
        last = fn(i)
        outputs.append(last)
    _sync(torch)
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / repeats
    checksum = 0.0
    try:
        checksum = float(last.float().sum().detach().cpu().numpy().reshape(-1)[0])
    except Exception:
        try:
            checksum = float(last.float().sum().data.reshape(-1)[0])
        except Exception:
            checksum = 0.0
    outputs.clear()
    return {"latency_ms": elapsed_ms, "checksum": checksum}


def _bench_cases(torch, backend: str, dtype_name: str, repeats: int, warmup: int, slots: int):
    device = torch.device("cuda")
    dtype = getattr(torch, dtype_name) if dtype_name != "float32" else None
    rng = np.random.default_rng(20260707)

    def pool(shape):
        return [_tensor_np(torch, shape, rng, device, dtype) for _ in range(slots)]

    a_big = pool((16, 1024, 1024))
    b_big = pool((16, 1024, 1024))
    a_small = pool((32, 512, 512))
    b_small = pool((32, 512, 512))
    q = pool((8, 16, 128, 64))
    k = pool((8, 16, 128, 64))
    v = pool((8, 16, 128, 64))
    x_softmax = pool((32, 256, 1024))
    x_elem = pool((32, 256, 1024))
    x_ln = pool((16, 128, 768))
    w_ln = _tensor_np(torch, (768,), rng, device, dtype)
    b_ln = _tensor_np(torch, (768,), rng, device, dtype)
    x_mlp = pool((16, 128, 768))
    w1 = _tensor_np(torch, (3072, 768), rng, device, dtype)
    b1 = _tensor_np(torch, (3072,), rng, device, dtype)
    w2 = _tensor_np(torch, (768, 3072), rng, device, dtype)
    b2 = _tensor_np(torch, (768,), rng, device, dtype)

    F = torch.nn.functional

    def t(i: int, values):
        return values[i % len(values)]

    cases: list[tuple[str, Callable[[int], Any]]] = [
        ("matmul_big_batched_16x1024", lambda i: torch.matmul(t(i, a_big), t(i, b_big))),
        ("matmul_small_batched_32x512", lambda i: torch.matmul(t(i, a_small), t(i, b_small))),
        ("softmax_32x256x1024", lambda i: F.softmax(t(i, x_softmax), dim=-1)),
        ("gelu_32x256x1024", lambda i: F.gelu(t(i, x_elem))),
        ("relu_32x256x1024", lambda i: F.relu(t(i, x_elem))),
        ("layernorm_16x128x768", lambda i: F.layer_norm(t(i, x_ln), (768,), w_ln, b_ln, 1e-5)),
        ("sdpa_math_8x16x128x64", lambda i: F.scaled_dot_product_attention(t(i, q), t(i, k), t(i, v), dropout_p=0.0)),
        ("mlp_16x128x768_3072", lambda i: F.linear(F.gelu(F.linear(t(i, x_mlp), w1, b1)), w2, b2)),
    ]

    rows = []
    with torch.no_grad():
        for name, fn in cases:
            result = _bench(torch, fn, repeats, warmup)
            rows.append({
                "backend": backend,
                "case": name,
                "dtype": dtype_name,
                "tf32": bool(getattr(torch.backends.cuda.matmul, "allow_tf32", False)),
                "repeats": repeats,
                "warmup": warmup,
                **result,
            })
    return rows


def _project_file(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    if not p.is_absolute():
        p = WORKDIR / p
    p = p.resolve()
    try:
        p.relative_to(WORKDIR)
    except ValueError as exc:
        raise SystemExit(f"output must stay under project root: {p}") from exc
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("torch", "jittor"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--tf32", choices=("on", "off"), default="off")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--slots", type=int, default=20)
    parser.add_argument("--jsonl", default="results/bottlenecks.jsonl")
    args = parser.parse_args(argv)

    _setup_env()
    torch = _import_backend(args.backend)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{args.backend} CUDA is not available")
    _set_tf32(torch, args.backend, args.tf32 == "on")

    rows = _bench_cases(torch, args.backend, args.dtype, args.repeats, args.warmup, max(args.slots, args.repeats))
    out = _project_file(args.jsonl)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            print(json.dumps(row, sort_keys=True), flush=True)
            f.write(json.dumps(row, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
