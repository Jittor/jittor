#!/usr/bin/env python3
"""Independent CUDA audit for the in-flight softmax/GELU/LayerNorm changes.

Run through ``run_perf_env.sh``. The script writes no files itself and emits one
JSON object per check, so callers can keep all logs under this project.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import subprocess
import time
from typing import Callable, Iterable

import numpy as np

import jittor as jt


def emit(**row) -> None:
    print(json.dumps(row, sort_keys=True), flush=True)


def as_numpy(x: jt.Var) -> np.ndarray:
    if str(x.dtype) in ("float16", "bfloat16"):
        x = x.float32()
    return np.asarray(x.numpy())


def error_stats(got: np.ndarray, ref: np.ndarray) -> dict:
    got64 = np.asarray(got, dtype=np.float64)
    ref64 = np.asarray(ref, dtype=np.float64)
    same_special = ((np.isnan(got64) & np.isnan(ref64)) |
                    (np.isposinf(got64) & np.isposinf(ref64)) |
                    (np.isneginf(got64) & np.isneginf(ref64)))
    finite = np.isfinite(got64) & np.isfinite(ref64)
    mismatch_special = ~(same_special | finite)
    if finite.any():
        delta = np.abs(got64[finite] - ref64[finite])
        denom = np.maximum(np.abs(ref64[finite]), 1e-30)
        max_abs = float(delta.max())
        max_rel = float((delta / denom).max())
        l2_rel = float(np.linalg.norm(delta) /
                       max(np.linalg.norm(ref64[finite]), 1e-30))
    else:
        max_abs = max_rel = l2_rel = 0.0
    return {
        "max_abs": max_abs,
        "max_rel": max_rel,
        "l2_rel": l2_rel,
        "special_mismatch": int(mismatch_special.sum()),
        "got_nonfinite": int((~np.isfinite(got64)).sum()),
        "ref_nonfinite": int((~np.isfinite(ref64)).sum()),
    }


def softmax_refs(x: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, ...]:
    x64 = np.asarray(x, dtype=np.float64)
    g64 = np.asarray(g, dtype=np.float64)
    shifted = x64 - np.max(x64, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    soft = exp / np.sum(exp, axis=-1, keepdims=True)
    log_soft = shifted - np.log(np.sum(exp, axis=-1, keepdims=True))
    dsoft = soft * (g64 - np.sum(soft * g64, axis=-1, keepdims=True))
    dlog = g64 - soft * np.sum(g64, axis=-1, keepdims=True)
    return soft, log_soft, dsoft, dlog


def run_softmax_case(length: int, dtype: str, rows: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x_src = rng.standard_normal((rows, length)).astype("float32")
    g_src = rng.standard_normal((rows, length)).astype("float32")
    x = jt.array(x_src).cast(dtype)
    g = jt.array(g_src).cast(dtype)
    # Reference the values after low-precision input rounding, not the fp32 seed.
    x_ref = as_numpy(x.float32())
    g_ref = as_numpy(g.float32())
    refs = softmax_refs(x_ref, g_ref)
    started = time.perf_counter()
    soft = jt.nn.softmax(x, dim=-1)
    log_soft = jt.nn.log_softmax(x, dim=-1)
    dsoft = jt.grad((soft * g).sum(), x)
    dlog = jt.grad((log_soft * g).sum(), x)
    got = jt.fetch_sync([soft.float32(), log_soft.float32(),
                         dsoft.float32(), dlog.float32()])
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    from jittor.other import code_softmax
    schedule = code_softmax._softmax_schedule(length)
    emit(
        kind="softmax_correctness",
        length=length,
        rows=rows,
        dtype=dtype,
        schedule=list(schedule),
        elapsed_ms=elapsed_ms,
        forward=error_stats(got[0], refs[0]),
        log_forward=error_stats(got[1], refs[1]),
        backward=error_stats(got[2], refs[2]),
        log_backward=error_stats(got[3], refs[3]),
        row_sum_max_abs=float(np.max(np.abs(
            np.asarray(got[0], dtype=np.float64).sum(-1) - 1.0))),
    )


def run_softmax_special(dtype: str) -> None:
    values = np.array([
        [0.0, 1.0, -1.0, np.nan, 2.0, -2.0, 3.0, -3.0],
        [-np.inf] * 8,
        [np.inf, 0.0, -1.0, 2.0, -2.0, 1.0, 3.0, -3.0],
        [10000.0, -10000.0, 0.0, 1.0, -1.0, 2.0, -2.0, 3.0],
    ], dtype="float32")
    x = jt.array(values).cast(dtype)
    soft, log_soft = jt.fetch_sync([
        jt.nn.softmax(x, -1).float32(),
        jt.nn.log_softmax(x, -1).float32(),
    ])
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        refs = softmax_refs(as_numpy(x.float32()), np.ones_like(values))
    emit(kind="softmax_special", dtype=dtype,
         forward=error_stats(soft, refs[0]),
         log_forward=error_stats(log_soft, refs[1]),
         finite_row_sum=float(np.asarray(soft[-1], dtype=np.float64).sum()))


def gelu_old(x: jt.Var) -> jt.Var:
    return (jt.erf(x / 1.4142135623730951) + 1) * x * .5


def generic_softmax(x: jt.Var, log: bool = False) -> jt.Var:
    shifted = x - x.max(-1, keepdims=True)
    if log:
        return shifted - jt.log(jt.exp(shifted).sum(-1, keepdims=True))
    exp = jt.exp(shifted)
    return exp / exp.sum(-1, keepdims=True)


def old_ln_normalize(x: jt.Var, dims: Iterable[int], eps: float) -> jt.Var:
    dims = tuple(dims)

    class OldLN(jt.Function):
        def execute(self, value):
            mean = jt.mean(value, dims=dims, keepdims=1)
            var = jt.mean((value - mean) * (value - mean),
                          dims=dims, keepdims=1)
            rstd = 1.0 / jt.sqrt(var + eps)
            xhat = (value - mean) * rstd
            self.xhat = xhat
            self.rstd = rstd
            return xhat

        def grad(self, grad):
            mg = jt.mean(grad, dims=dims, keepdims=1)
            mgx = jt.mean(grad * self.xhat, dims=dims, keepdims=1)
            return self.rstd * (grad - mg - self.xhat * mgx)

    return OldLN.apply(x)


def sync_value(value) -> None:
    if isinstance(value, (tuple, list)):
        jt.sync(list(value))
    else:
        value.sync()


def resource_usage(source: pathlib.Path) -> dict:
    so = source.with_suffix(".so")
    if not so.is_file():
        return {}
    tool = pathlib.Path(jt.compiler.nvcc_path).with_name("cuobjdump")
    if not tool.is_file():
        return {"binary": str(so)}
    proc = subprocess.run([str(tool), "--dump-resource-usage", str(so)],
                          text=True, capture_output=True, check=False)
    lines = [line.strip() for line in (proc.stdout + proc.stderr).splitlines()
             if any(token in line for token in ("REG:", "SHARED:", "STACK:",
                                                  "Function "))]
    return {"binary": str(so), "resource_lines": lines[-20:],
            "cuobjdump_rc": proc.returncode}


def profile(label: str, fn: Callable, args: tuple, warmup: int, rerun: int) -> None:
    value = fn(*args)
    sync_value(value)
    with jt.profile_scope(warmup, rerun, profiler_hide_relay=1) as report:
        value = fn(*args)
        sync_value(value)
    header = report[0]
    rows = [dict(zip(header, row)) for row in report[1:]]
    sources = []
    launch_sites = 0
    float64_tokens = 0
    resources = []
    for row in rows:
        source = pathlib.Path(row["FileName"])
        sources.append(str(source))
        if source.is_file():
            text = source.read_text(encoding="utf-8", errors="replace")
            launch_sites += text.count("<<<")
            float64_tokens += text.count("float64")
            resources.append(resource_usage(source))
    emit(kind="profile", label=label,
         avg_kernel_us=sum(float(row["AvgTime"]) for row in rows) / 1000.0,
         profile_rows=len(rows), source_launch_sites=launch_sites,
         source_float64_tokens=float64_tokens, source_files=sources,
         resources=resources, rows=rows)


def wall(label: str, fn: Callable[[jt.Var], jt.Var], shape: tuple[int, ...],
         slots: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pool = [jt.array(rng.standard_normal(shape).astype("float32"))
            for _ in range(slots)]
    warm = [fn(x) for x in pool]
    jt.sync(warm)
    del warm
    start = time.perf_counter()
    outputs = [fn(x) for x in pool]
    built = time.perf_counter()
    jt.sync(outputs)
    finished = time.perf_counter()
    emit(kind="wall", label=label, shape=list(shape), slots=slots,
         total_ms=(finished - start) * 1000.0 / slots,
         build_ms=(built - start) * 1000.0 / slots,
         sync_ms=(finished - built) * 1000.0 / slots)


def run_profile(op: str, length: int, rows: int, warmup: int,
                rerun: int, slots: int) -> None:
    rng = np.random.default_rng(2026071103 + length)
    if op == "softmax":
        from jittor.other import code_softmax
        x = jt.array(rng.standard_normal((rows, length)).astype("float32"))
        g = jt.array(rng.standard_normal((rows, length)).astype("float32"))
        current = lambda z: jt.nn.softmax(z, -1)
        streaming_cls = code_softmax._softmax_streaming_cls(512, False)
        streaming = lambda z: streaming_cls()(z)
        def current_pair(z, go):
            value = current(z)
            return value, jt.grad((value * go).sum(), z)
        def streaming_pair(z, go):
            value = streaming(z)
            return value, jt.grad((value * go).sum(), z)
        profile(f"softmax_current_{rows}x{length}", current, (x,), warmup, rerun)
        profile(f"softmax_current_pair_{rows}x{length}", current_pair,
                (x, g), warmup, rerun)
        profile(f"softmax_streaming_{rows}x{length}", streaming,
                (x,), warmup, rerun)
        profile(f"softmax_streaming_pair_{rows}x{length}", streaming_pair,
                (x, g), warmup, rerun)
        wall(f"softmax_current_{rows}x{length}", current, (rows, length),
             slots, 20260712 + length)
        wall(f"softmax_streaming_{rows}x{length}", streaming, (rows, length),
             slots, 20260718 + length)
        if length > 10000:
            profile(f"softmax_generic_{rows}x{length}", generic_softmax,
                    (x,), warmup, rerun)
            wall(f"softmax_generic_{rows}x{length}", generic_softmax,
                 (rows, length), slots, 20260713 + length)
    elif op == "gelu":
        shape = (rows, length)
        x = jt.array(rng.standard_normal(shape).astype("float32"))
        current = lambda z: jt.nn.gelu(z, approximate="none")
        profile("gelu_current", current, (x,), warmup, rerun)
        profile("gelu_old", gelu_old, (x,), warmup, rerun)
        wall("gelu_current", current, shape, slots, 20260714)
        wall("gelu_old", gelu_old, shape, slots, 20260715)
    elif op == "layernorm":
        shape = (rows, length)
        x = jt.array(rng.standard_normal(shape).astype("float32"))
        g = jt.array(rng.standard_normal(shape).astype("float32"))
        w = jt.array(rng.standard_normal((length,)).astype("float32"))
        b = jt.array(rng.standard_normal((length,)).astype("float32"))
        current = lambda z: jt.nn.layer_norm(z, (length,), w, b, 1e-5)
        old = lambda z: old_ln_normalize(z, (-1,), 1e-5) * w + b
        def current_pair(z, go):
            value = current(z)
            return value, *jt.grad((value * go).sum(), [z, w, b])
        def old_pair(z, go):
            value = old(z)
            return value, *jt.grad((value * go).sum(), [z, w, b])
        profile("layernorm_current_forward", current, (x,), warmup, rerun)
        profile("layernorm_old_forward", old, (x,), warmup, rerun)
        profile("layernorm_current_pair", current_pair, (x, g), warmup, rerun)
        profile("layernorm_old_pair", old_pair, (x, g), warmup, rerun)
        wall("layernorm_current", current, shape, slots, 20260716)
        wall("layernorm_old", old, shape, slots, 20260717)
    else:
        raise ValueError(op)


def run_softmax_forced_schedule(length: int, threads: int, rows: int,
                                warmup: int, rerun: int, slots: int) -> None:
    """Profile a register schedule without changing the production selector."""
    from jittor.other import code_softmax

    original_schedule = code_softmax._softmax_schedule
    code_softmax._softmax_v1_cls.cache_clear()
    try:
        code_softmax._softmax_schedule = lambda _: ("register", threads)
        forced_cls = code_softmax._softmax_v1_cls(length, False)
    finally:
        code_softmax._softmax_schedule = original_schedule
        code_softmax._softmax_v1_cls.cache_clear()

    rng = np.random.default_rng(2026071121 + length + threads)
    x = jt.array(rng.standard_normal((rows, length)).astype("float32"))
    g = jt.array(rng.standard_normal((rows, length)).astype("float32"))
    forced = lambda z: forced_cls()(z)

    def forced_pair(z, go):
        value = forced(z)
        return value, jt.grad((value * go).sum(), z)

    suffix = f"{rows}x{length}_t{threads}"
    profile(f"softmax_forced_register_{suffix}", forced, (x,), warmup, rerun)
    profile(f"softmax_forced_register_pair_{suffix}", forced_pair,
            (x, g), warmup, rerun)
    wall(f"softmax_forced_register_{suffix}", forced, (rows, length),
         slots, 2026071122 + length + threads)


def layernorm_reference(x: np.ndarray, g: np.ndarray, w: np.ndarray,
                        b: np.ndarray, axes: tuple[int, ...],
                        eps: float) -> tuple[np.ndarray, ...]:
    x64 = x.astype(np.float64)
    g64 = g.astype(np.float64)
    w64 = w.astype(np.float64)
    b64 = b.astype(np.float64)
    mean = x64.mean(axis=axes, keepdims=True)
    var = ((x64 - mean) ** 2).mean(axis=axes, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = (x64 - mean) * rstd
    y = xhat * w64 + b64
    grad_norm = g64 * w64
    dx = rstd * (grad_norm - grad_norm.mean(axis=axes, keepdims=True) -
                  xhat * (grad_norm * xhat).mean(axis=axes, keepdims=True))
    reduce_axes = tuple(i for i in range(x64.ndim) if i not in axes)
    dw = (g64 * xhat).sum(axis=reduce_axes)
    db = g64.sum(axis=reduce_axes)
    return y, dx, dw, db


def run_layernorm_correctness(scale: float, eps: float,
                              multidim: bool, seed: int) -> None:
    rng = np.random.default_rng(seed)
    shape = (4, 2, 3, 4) if multidim else (8, 768)
    normalized_shape = (3, 4) if multidim else (768,)
    axes = (-2, -1) if multidim else (-1,)
    affine_shape = (1, 1, 3, 4) if multidim else (768,)
    x_np = (3.0 + scale * rng.standard_normal(shape)).astype("float32")
    g_np = rng.standard_normal(shape).astype("float32")
    w_base = rng.standard_normal(normalized_shape).astype("float32")
    b_base = rng.standard_normal(normalized_shape).astype("float32")
    w_ref = w_base.reshape(affine_shape)
    b_ref = b_base.reshape(affine_shape)
    refs = layernorm_reference(x_np, g_np, w_ref, b_ref, axes, eps)
    x = jt.array(x_np)
    g = jt.array(g_np)
    w = jt.array(w_base)
    b = jt.array(b_base)
    y = jt.nn.layer_norm(x, normalized_shape, w, b, eps)
    dx, dw, db = jt.grad((y * g).sum(), [x, w, b])
    got = jt.fetch_sync([y, dx, dw, db])
    emit(kind="layernorm_correctness", shape=list(shape),
         normalized_shape=list(normalized_shape), scale=scale, eps=eps,
         forward=error_stats(got[0], refs[0]),
         dx=error_stats(got[1], refs[1]),
         dw=error_stats(got[2], refs[2].reshape(w_base.shape)),
         db=error_stats(got[3], refs[3].reshape(b_base.shape)))


def run_cache_probe(calls: int) -> None:
    from jittor import nn
    from jittor.other import code_softmax
    nn._ln_function_cls.cache_clear()
    code_softmax._softmax_v1_cls.cache_clear()
    x = jt.ones((2, 768), dtype="float32")
    w = jt.ones((768,), dtype="float32")
    b = jt.zeros((768,), dtype="float32")
    start = time.perf_counter()
    outputs = [nn.layer_norm(x + np.float32(i * 1e-6), (768,), w, b, 1e-5)
               for i in range(calls)]
    ln_build_us = (time.perf_counter() - start) * 1e6 / calls
    jt.sync(outputs)
    sx = jt.ones((2, 1024), dtype="float32")
    start = time.perf_counter()
    outputs = [nn.softmax(sx + np.float32(i * 1e-6), -1)
               for i in range(calls)]
    softmax_build_us = (time.perf_counter() - start) * 1e6 / calls
    jt.sync(outputs)
    emit(kind="cache_probe", calls=calls, ln_build_us=ln_build_us,
         softmax_build_us=softmax_build_us,
         ln_cache=str(nn._ln_function_cls.cache_info()),
         softmax_cache=str(code_softmax._softmax_v1_cls.cache_info()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("softmax-case", "softmax-special",
                                            "softmax-forced-schedule",
                                            "layernorm-correctness", "profile",
                                            "cache-probe"), required=True)
    parser.add_argument("--length", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"),
                        default="float32")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--multidim", action="store_true")
    parser.add_argument("--op", choices=("softmax", "gelu", "layernorm"),
                        default="softmax")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rerun", type=int, default=20)
    parser.add_argument("--slots", type=int, default=6)
    parser.add_argument("--threads", type=int, default=500)
    parser.add_argument("--calls", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    emit(kind="environment", task=args.task, gpu=jt.get_device_count(),
         jittor=jt.__version__, cuda_archs=list(jt.flags.cuda_archs),
         cache_name=str(getattr(jt.flags, "cache_path", "")))
    if args.task == "softmax-case":
        run_softmax_case(args.length, args.dtype, args.rows, args.seed)
    elif args.task == "softmax-special":
        run_softmax_special(args.dtype)
    elif args.task == "layernorm-correctness":
        run_layernorm_correctness(args.scale, args.eps, args.multidim, args.seed)
    elif args.task == "profile":
        run_profile(args.op, args.length, args.rows, args.warmup,
                    args.rerun, args.slots)
    elif args.task == "softmax-forced-schedule":
        run_softmax_forced_schedule(args.length, args.threads, args.rows,
                                    args.warmup, args.rerun, args.slots)
    else:
        run_cache_probe(args.calls)
    jt.sync_all(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
