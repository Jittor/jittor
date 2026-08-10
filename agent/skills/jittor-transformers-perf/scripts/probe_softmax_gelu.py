#!/usr/bin/env python3
"""CUDA-only micro probes for Jittor softmax and GELU code generation.

Run through ``run_perf_env.sh`` so all JIT/cache artifacts stay in this project.
The experimental kernels are intentionally local to this probe; they do not
change Jittor's implementation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Callable

import numpy as np

import jittor as jt


from _paths import WORK_ROOT as WORKDIR


def _profile(label: str, fn: Callable[[jt.Var], jt.Var], x: jt.Var,
             warmup: int, rerun: int) -> dict:
    # Compile once before enabling the profiler so compilation is not measured.
    y = fn(x)
    y.sync()
    del y

    with jt.profile_scope(warmup, rerun, profiler_hide_relay=1) as report:
        y = fn(x)
        y.sync()

    header = report[0]
    rows = [dict(zip(header, row)) for row in report[1:]]
    source_files = []
    launch_sites = 0
    launch_lines = []
    float64_tokens = 0
    for row in rows:
        source = pathlib.Path(row["FileName"])
        source_files.append(str(source))
        if source.is_file():
            text = source.read_text(encoding="utf-8", errors="replace")
            launch_sites += text.count("<<<")
            float64_tokens += text.count("float64")
            launch_lines.extend(
                line.strip() for line in text.splitlines() if "<<<" in line
            )

    avg_us = sum(float(row["AvgTime"]) for row in rows) / 1000.0
    return {
        "kind": "profile",
        "label": label,
        "shape": list(x.shape),
        "profile_rows": len(rows),
        "avg_us": avg_us,
        "rows": rows,
        "source_files": source_files,
        "source_launch_sites": launch_sites,
        "source_launch_lines": launch_lines,
        "source_float64_tokens": float64_tokens,
    }


def _wall_bench(label: str, fn: Callable[[jt.Var], jt.Var], pool: list[jt.Var],
                warmup: int) -> dict:
    # Touch every slot before timing. Jittor's dual allocator can otherwise make
    # the first measured case pay host-to-device page migration for cold slots.
    warm_outputs = [
        fn(pool[i % len(pool)]) for i in range(max(warmup, len(pool)))
    ]
    jt.sync_all(True)
    del warm_outputs

    start = time.perf_counter()
    outputs = [fn(x) for x in pool]
    built = time.perf_counter()
    jt.sync_all(True)
    finished = time.perf_counter()
    elapsed_ms = (finished - start) * 1000.0 / len(pool)
    del outputs
    return {
        "kind": "wall_time",
        "label": label,
        "shape": list(pool[0].shape),
        "slots": len(pool),
        "latency_ms": elapsed_ms,
        "build_ms": (built - start) * 1000.0 / len(pool),
        "sync_ms": (finished - built) * 1000.0 / len(pool),
    }


def _make_pool(shape: tuple[int, ...], slots: int, seed: int) -> list[jt.Var]:
    rng = np.random.default_rng(seed)
    pool = [jt.array(rng.standard_normal(shape).astype("float32")) for _ in range(slots)]
    jt.sync_all(True)
    return pool


def _wall_bench_pairs(label: str, fn: Callable[[jt.Var, jt.Var], tuple[jt.Var, jt.Var]],
                      xs: list[jt.Var], grad_ys: list[jt.Var], warmup: int) -> dict:
    slots = len(xs)
    warm_outputs = [
        fn(xs[i % slots], grad_ys[i % slots])
        for i in range(max(warmup, slots))
    ]
    jt.sync_all(True)
    del warm_outputs

    start = time.perf_counter()
    outputs = [fn(x, grad_y) for x, grad_y in zip(xs, grad_ys)]
    built = time.perf_counter()
    jt.sync_all(True)
    finished = time.perf_counter()
    del outputs
    return {
        "kind": "training_wall_time",
        "label": label,
        "shape": list(xs[0].shape),
        "slots": slots,
        "latency_ms": (finished - start) * 1000.0 / slots,
        "build_ms": (built - start) * 1000.0 / slots,
        "sync_ms": (finished - built) * 1000.0 / slots,
    }


def _gelu_exact_fp32(x: jt.Var) -> jt.Var:
    half = np.float32(0.5)
    one = np.float32(1.0)
    inv_sqrt2 = np.float32(0.7071067811865476)
    return half * x * (one + jt.erf(x * inv_sqrt2))


def _gelu_tanh_fp32(x: jt.Var) -> jt.Var:
    half = np.float32(0.5)
    one = np.float32(1.0)
    sqrt_2_over_pi = np.float32(0.7978845608028654)
    cubic = np.float32(0.044715)
    return half * x * (one + jt.tanh(sqrt_2_over_pi * (x + cubic * x * x * x)))


def _gelu_exact_code(x: jt.Var, threads: int = 256) -> jt.Var:
    return jt.code(
        x.shape,
        x.dtype,
        [x],
        cuda_header="#include <math.h>\n",
        cuda_src=f"""
__global__ void gelu_exact_probe(in0_type* x, out0_type* y, int n) {{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {{
        float v = float(x[i]);
        y[i] = out0_type(0.5f * v * (1.0f + erff(v * 0.7071067811865476f)));
    }}
}}
int n = in0->num;
int blocks = (n + {threads} - 1) / {threads};
if (blocks > 4096) blocks = 4096;
gelu_exact_probe<<<blocks, {threads}>>>(in0_p, out0_p, n);
CHECK(0 == cudaGetLastError());
""",
    )


def _softmax_code(x: jt.Var, threads: int) -> jt.Var:
    """Clone the native one-block-per-row algorithm with tunable block size."""
    length = x.shape[-1]
    items = (length - 1) // threads + 1
    ilp = 1
    for candidate in (8, 4, 2):
        if length % threads == 0 and items % candidate == 0:
            ilp = candidate
            items //= candidate
            break

    loop = f"""
    #pragma unroll
    for (int i = 0; i < {items}; ++i)
"""
    if length % threads:
        loop += f"if ((i * {threads} + threadIdx.x) * {ilp} < len)\n"

    return jt.code(
        x.shape,
        x.dtype,
        [x],
        cuda_header=f"""
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
""",
        cuda_src=f"""
__global__ void softmax_probe(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {threads}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    int id = blockIdx.x * len;
    in0_type v[{items}][{ilp}];
    {loop}
        vload<sizeof(in0_type) * {ilp}>(v[i], &x[id + (i * {threads} + threadIdx.x) * {ilp}]);

    float local = -1e30f;
    {loop}
        #pragma unroll
        for (int j = 0; j < {ilp}; ++j)
            local = ::max(local, float(v[i][j]));

    __shared__ float vmax;
    float reduced = BlockReduce(temp_storage).Reduce(local, cub::Max());
    if (threadIdx.x == 0) vmax = reduced;
    __syncthreads();

    local = 0.0f;
    {loop}
        #pragma unroll
        for (int j = 0; j < {ilp}; ++j) {{
            v[i][j] = expf(float(v[i][j]) - vmax);
            local += float(v[i][j]);
        }}

    reduced = BlockReduce(temp_storage).Sum(local);
    __shared__ float vsum;
    if (threadIdx.x == 0) vsum = reduced;
    __syncthreads();

    {loop}
        #pragma unroll
        for (int j = 0; j < {ilp}; ++j)
            v[i][j] = float(v[i][j]) / vsum;
    {loop}
        vload<sizeof(in0_type) * {ilp}>(&y[id + (i * {threads} + threadIdx.x) * {ilp}], v[i]);
}}
int len = in0->shape[in0->shape.size() - 1];
int rows = in0->numel() / len;
softmax_probe<<<rows, {threads}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
""",
    )


class _HoistedSoftmax128(jt.Function):
    """Probe the cost of moving CodeSoftmax out of the per-call hot path."""

    def execute(self, x: jt.Var) -> jt.Var:
        self.y = _softmax_code(x, 128)
        return self.y

    def grad(self, grad_y: jt.Var) -> jt.Var:
        return self.y * (grad_y - (self.y * grad_y).sum(-1, keepdims=True))


def _softmax_hoisted_128(x: jt.Var) -> jt.Var:
    return _HoistedSoftmax128()(x)


class _SyncPair:
    def __init__(self, values: tuple[jt.Var, jt.Var]):
        self.values = values

    def sync(self) -> None:
        jt.sync(list(self.values))


def _max_abs(a: jt.Var, b: jt.Var) -> float:
    return float((a - b).abs().max().item())


def run_gelu(elements: int, warmup: int, rerun: int,
             wall_repeats: int, wall_warmup: int) -> None:
    cols = 1024
    rows = max(1, elements // cols)
    x = jt.array(np.random.default_rng(20260710).standard_normal((rows, cols)).astype("float32"))
    x.sync()

    native_exact = lambda z: jt.nn.gelu(z, approximate="none")
    native_tanh = lambda z: jt.nn.gelu(z, approximate="tanh")
    cases = [
        ("gelu_native_exact", native_exact),
        ("gelu_exact_forced_fp32", _gelu_exact_fp32),
        ("gelu_exact_code_256", _gelu_exact_code),
        ("gelu_native_tanh", native_tanh),
        ("gelu_tanh_forced_fp32", _gelu_tanh_fp32),
    ]
    for label, fn in cases:
        print(json.dumps(_profile(label, fn, x, warmup, rerun), sort_keys=True), flush=True)

    pool = _make_pool((rows, cols), wall_repeats, 20260711)
    for label, fn in cases:
        print(json.dumps(_wall_bench(label, fn, pool, wall_warmup), sort_keys=True), flush=True)
    del pool

    check = x[: min(rows, 256)]
    exact_ref = native_exact(check)
    tanh_ref = native_tanh(check)
    for label, fn, ref in [
        ("gelu_exact_forced_fp32", _gelu_exact_fp32, exact_ref),
        ("gelu_exact_code_256", _gelu_exact_code, exact_ref),
        ("gelu_tanh_forced_fp32", _gelu_tanh_fp32, tanh_ref),
    ]:
        print(json.dumps({
            "kind": "correctness",
            "label": label,
            "max_abs_vs_native": _max_abs(fn(check), ref),
        }, sort_keys=True), flush=True)


def run_softmax(elements: int, warmup: int, rerun: int,
                wall_repeats: int, wall_warmup: int) -> None:
    length = 1024
    rows = max(1, elements // length)
    x = jt.array(np.random.default_rng(20260710).standard_normal((rows, length)).astype("float32"))
    x.sync()
    native = lambda z: jt.nn.softmax(z, dim=-1)
    cases = [
        ("softmax_native", native),
        ("softmax_hoisted_t128", _softmax_hoisted_128),
    ] + [
        (f"softmax_probe_t{threads}", lambda z, t=threads: _softmax_code(z, t))
        for threads in (128, 256, 512)
    ]
    for label, fn in cases:
        print(json.dumps(_profile(label, fn, x, warmup, rerun), sort_keys=True), flush=True)

    pool = _make_pool((rows, length), wall_repeats, 20260712)
    for label, fn in cases:
        print(json.dumps(_wall_bench(label, fn, pool, wall_warmup), sort_keys=True), flush=True)
    del pool

    check = x[: min(rows, 256)]
    ref = native(check)
    for threads in (128, 256, 512):
        print(json.dumps({
            "kind": "correctness",
            "label": f"softmax_probe_t{threads}",
            "max_abs_vs_native": _max_abs(_softmax_code(check, threads), ref),
        }, sort_keys=True), flush=True)


def run_vocab_softmax(warmup: int, rerun: int,
                      wall_repeats: int, wall_warmup: int) -> None:
    # Keep roughly one million elements while crossing the native fast-path
    # cutoff at last-dimension length 10,000.
    shapes = ((128, 8192), (104, 10001), (21, 50257))
    for index, shape in enumerate(shapes):
        x = jt.array(
            np.random.default_rng(20260720 + index)
            .standard_normal(shape)
            .astype("float32")
        )
        x.sync()
        native = lambda z: jt.nn.softmax(z, dim=-1)
        suffix = f"{shape[0]}x{shape[1]}"
        thread_options = (256,) if shape[1] > 32768 else (256, 512)
        cases = [(f"softmax_native_{suffix}", native)] + [
            (f"softmax_probe_t{threads}_{suffix}",
             lambda z, t=threads: _softmax_code(z, t))
            for threads in thread_options
        ]
        for label, fn in cases:
            print(json.dumps(_profile(label, fn, x, warmup, rerun), sort_keys=True), flush=True)

        pool = _make_pool(shape, wall_repeats, 20260730 + index)
        for label, fn in cases:
            print(json.dumps(_wall_bench(label, fn, pool, wall_warmup), sort_keys=True), flush=True)
        del pool

        ref = native(x)
        for label, fn in cases:
            value = fn(x)
            sums = value.sum(-1)
            print(json.dumps({
                "kind": "correctness",
                "label": label,
                "max_row_sum_error": float((sums - 1.0).abs().max().item()),
                "max_abs_vs_native": _max_abs(value, ref),
            }, sort_keys=True), flush=True)


def run_training(elements: int, warmup: int, rerun: int,
                 wall_repeats: int, wall_warmup: int) -> None:
    cols = 1024
    rows = max(1, elements // cols)
    rng = np.random.default_rng(20260713)
    x = jt.array(rng.standard_normal((rows, cols)).astype("float32"))
    grad_y = jt.array(rng.standard_normal((rows, cols)).astype("float32"))
    x.sync()
    grad_y.sync()

    def grad_of(op: Callable[[jt.Var], jt.Var]) -> Callable[[jt.Var], jt.Var]:
        def fn(z: jt.Var) -> jt.Var:
            return jt.grad((op(z) * grad_y).sum(), z)
        return fn

    cases = [
        ("gelu_train_native_exact", grad_of(lambda z: jt.nn.gelu(z, approximate="none"))),
        ("gelu_train_forced_fp32", grad_of(_gelu_exact_fp32)),
        ("gelu_train_native_tanh", grad_of(lambda z: jt.nn.gelu(z, approximate="tanh"))),
        ("gelu_train_tanh_fp32", grad_of(_gelu_tanh_fp32)),
        ("softmax_train_native", grad_of(lambda z: jt.nn.softmax(z, dim=-1))),
        ("softmax_train_hoisted_t128", grad_of(_softmax_hoisted_128)),
    ]
    for label, fn in cases:
        print(json.dumps(_profile(label, fn, x, warmup, rerun), sort_keys=True), flush=True)

    def train_pair(op: Callable[[jt.Var], jt.Var]) -> Callable[[jt.Var, jt.Var], tuple[jt.Var, jt.Var]]:
        def fn(z: jt.Var, upstream: jt.Var) -> tuple[jt.Var, jt.Var]:
            y = op(z)
            dx = jt.grad((y * upstream).sum(), z)
            return y, dx
        return fn

    xs = _make_pool((rows, cols), wall_repeats, 20260714)
    grad_ys = _make_pool((rows, cols), wall_repeats, 20260715)
    train_cases = [
        ("gelu_train_native_exact", train_pair(lambda z: jt.nn.gelu(z, approximate="none"))),
        ("gelu_train_forced_fp32", train_pair(_gelu_exact_fp32)),
        ("gelu_train_native_tanh", train_pair(lambda z: jt.nn.gelu(z, approximate="tanh"))),
        ("softmax_train_native", train_pair(lambda z: jt.nn.softmax(z, dim=-1))),
        ("softmax_train_hoisted_t128", train_pair(_softmax_hoisted_128)),
    ]
    for label, fn in train_cases:
        print(json.dumps(
            _wall_bench_pairs(label, fn, xs, grad_ys, wall_warmup),
            sort_keys=True,
        ), flush=True)
    del xs, grad_ys

    for label, fn in train_cases:
        wrapped = lambda z, pair_fn=fn: _SyncPair(pair_fn(z, grad_y))
        print(json.dumps(
            _profile(label + "_materialized_pair", wrapped, x, warmup, rerun),
            sort_keys=True,
        ), flush=True)

    check = x[: min(rows, 256)]
    check_grad = grad_y[: min(rows, 256)]

    def check_grad_of(op: Callable[[jt.Var], jt.Var]) -> jt.Var:
        return jt.grad((op(check) * check_grad).sum(), check)

    exact_native = check_grad_of(lambda z: jt.nn.gelu(z, approximate="none"))
    tanh_native = check_grad_of(lambda z: jt.nn.gelu(z, approximate="tanh"))
    softmax_native = check_grad_of(lambda z: jt.nn.softmax(z, dim=-1))
    for label, value, ref in [
        ("gelu_train_forced_fp32", check_grad_of(_gelu_exact_fp32), exact_native),
        ("gelu_train_tanh_fp32", check_grad_of(_gelu_tanh_fp32), tanh_native),
        ("softmax_train_hoisted_t128", check_grad_of(_softmax_hoisted_128), softmax_native),
    ]:
        print(json.dumps({
            "kind": "gradient_correctness",
            "label": label,
            "max_abs_vs_native": _max_abs(value, ref),
        }, sort_keys=True), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--section",
        choices=("all", "gelu", "softmax", "softmax_vocab", "training"),
        default="all",
    )
    parser.add_argument("--elements", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rerun", type=int, default=30)
    parser.add_argument("--wall-repeats", type=int, default=8)
    parser.add_argument("--wall-warmup", type=int, default=3)
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    print(json.dumps({
        "kind": "environment",
        "jittor": jt.__version__,
        "cuda_archs": list(jt.flags.cuda_archs),
        "elements": args.elements,
    }, sort_keys=True), flush=True)
    with jt.no_grad():
        if args.section in ("all", "gelu"):
            run_gelu(args.elements, args.warmup, args.rerun,
                     args.wall_repeats, args.wall_warmup)
        if args.section in ("all", "softmax"):
            run_softmax(args.elements, args.warmup, args.rerun,
                        args.wall_repeats, args.wall_warmup)
        if args.section in ("all", "softmax_vocab"):
            run_vocab_softmax(args.warmup, args.rerun,
                              args.wall_repeats, args.wall_warmup)
    if args.section in ("all", "training"):
        run_training(args.elements, args.warmup, args.rerun,
                     args.wall_repeats, args.wall_warmup)
    jt.sync_all(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
