#!/usr/bin/env python3
"""Probe register-bounded streaming CUDA softmax for large vocabularies."""

from __future__ import annotations

import json
import time

import numpy as np
import jittor as jt


def streaming_softmax(x: jt.Var, threads: int) -> jt.Var:
    return jt.code(
        x.shape,
        x.dtype,
        [x],
        cuda_header=f"""
#include <{jt.compile_extern.cub_home}cub/cub.cuh>
#include <type/fp16_compute.h>
""",
        cuda_src=f"""
__global__ void large_softmax(in0_type* x, out0_type* y, int len) {{
    typedef cub::BlockReduce<float, {threads}> BlockReduce;
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ float row_max;
    __shared__ float row_sum;
    int base = blockIdx.x * len;

    float local = -INFINITY;
    for (int col = threadIdx.x; col < len; col += blockDim.x)
        local = ::max(local, float(x[base + col]));
    float reduced = BlockReduce(storage).Reduce(local, cub::Max());
    if (threadIdx.x == 0) row_max = reduced;
    __syncthreads();

    local = 0.0f;
    for (int col = threadIdx.x; col < len; col += blockDim.x)
        local += expf(float(x[base + col]) - row_max);
    reduced = BlockReduce(storage).Sum(local);
    if (threadIdx.x == 0) row_sum = reduced;
    __syncthreads();

    for (int col = threadIdx.x; col < len; col += blockDim.x)
        y[base + col] = out0_type(expf(float(x[base + col]) - row_max) / row_sum);
}}
int len = in0->shape[in0->shape.size() - 1];
int rows = in0->numel() / len;
large_softmax<<<rows, {threads}>>>(in0_p, out0_p, len);
CHECK(0 == cudaGetLastError());
""",
    )


def bench(rows: int, cols: int, threads: int, slots: int = 12) -> dict:
    rng = np.random.default_rng(20260710 + cols + threads)
    pool = [jt.array(rng.standard_normal((rows, cols)).astype("float32"))
            for _ in range(slots)]
    warm = [streaming_softmax(x, threads) for x in pool]
    jt.sync_all(True)
    del warm

    start = time.perf_counter()
    outputs = [streaming_softmax(x, threads) for x in pool]
    built = time.perf_counter()
    jt.sync_all(True)
    finished = time.perf_counter()

    check = pool[0][: min(rows, 8)]
    ref = jt.nn.softmax(check, dim=-1)
    got = streaming_softmax(check, threads)
    max_abs = float((got - ref).abs().max().item())
    row_error = float((got.sum(-1) - 1.0).abs().max().item())
    return {
        "rows": rows,
        "cols": cols,
        "threads": threads,
        "latency_ms": (finished - start) * 1000.0 / slots,
        "build_ms": (built - start) * 1000.0 / slots,
        "sync_ms": (finished - built) * 1000.0 / slots,
        "max_abs_vs_native": max_abs,
        "max_row_sum_error": row_error,
    }


def main() -> int:
    jt.flags.use_cuda = 1
    for rows, cols in ((104, 10001), (21, 50257)):
        for threads in (128, 256, 512):
            print(json.dumps(bench(rows, cols, threads), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
