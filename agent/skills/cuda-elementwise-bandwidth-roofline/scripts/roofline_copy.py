"""Measure the achievable DRAM bandwidth of the current CUDA device.

A pure grid-stride float4 copy is the empirical denominator for every
"this elementwise kernel reaches X GB/s" claim.  The vendor's nominal figure
is never reachable, so a roofline built on it understates how much headroom a
kernel really has; measure it once per machine and reuse the number.

Run with an explicit PYTHONPATH pointing at the worktree under test::

    PYTHONPATH=<worktree>/python JITTOR_HOME=... TMPDIR=... \
    CUDA_VISIBLE_DEVICES=... nvcc_path=/usr/local/cuda/bin/nvcc \
    python roofline_copy.py --mb 512
"""

import argparse
import json
import os

import numpy as np

import jittor as jt

COPY_SRC = r"""
__global__ void jt_roofline_copy(const float4* __restrict__ src,
                                 float4* __restrict__ dst, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        dst[i] = src[i];
}
int n4 = in0->num / 4;
int block = 256;
int grid = (n4 + block - 1) / block;
if (grid > 65535) grid = 65535;
jt_roofline_copy<<<grid, block>>>((const float4*)in0_p, (float4*)out0_p, n4);
"""

SCALE_SRC = r"""
__global__ void jt_roofline_scale(const float4* __restrict__ src,
                                  float4* __restrict__ dst, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float4 v = src[i];
        v.x *= 2.0f; v.y *= 2.0f; v.z *= 2.0f; v.w *= 2.0f;
        dst[i] = v;
    }
}
int n4 = in0->num / 4;
int block = 256;
int grid = (n4 + block - 1) / block;
if (grid > 65535) grid = 65535;
jt_roofline_scale<<<grid, block>>>((const float4*)in0_p, (float4*)out0_p, n4);
"""


def _bandwidth(name, src, elems, rerun, warmup):
    a = jt.random((elems,), "float32")
    a.sync()
    jt.code(a.shape, a.dtype, [a], cuda_src=src).sync()
    jt.sync_all(True)
    with jt.profile_scope(warmup=warmup, rerun=rerun) as rep:
        jt.code(a.shape, a.dtype, [a], cuda_src=src).sync()
        jt.sync_all(True)
    header, rows = rep[0], rep[1:]
    ci, ti = header.index("Count"), header.index("TotalTime")
    row = max(rows, key=lambda r: float(r[ti]))
    per_call_ns = float(row[ti]) / int(row[ci])
    moved = elems * 4 * 2  # one read plus one write per element
    return {"name": name, "bytes": moved, "ns": per_call_ns,
            "gbps": moved / per_call_ns}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mb", type=int, default=512,
                        help="size of the copied buffer in MiB")
    parser.add_argument("--rerun", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--out", default="")
    options = parser.parse_args()

    jt.flags.use_cuda = 1
    elems = options.mb * 1024 * 1024 // 4
    results = [
        _bandwidth("copy_float4", COPY_SRC, elems, options.rerun, options.warmup),
        _bandwidth("scale_float4", SCALE_SRC, elems, options.rerun, options.warmup),
    ]
    # The scale kernel must actually have run: a kernel that silently did
    # nothing would report an excellent bandwidth.
    a = jt.random((1024,), "float32")
    got = jt.code(a.shape, a.dtype, [a], cuda_src=SCALE_SRC).numpy()
    assert np.allclose(got, a.numpy() * 2.0), "roofline scale kernel is wrong"

    achievable = max(r["gbps"] for r in results)
    for r in results:
        print("%-14s %8.1f GB/s  (%.1f us for %.0f MiB moved)"
              % (r["name"], r["gbps"], r["ns"] / 1e3, r["bytes"] / 1048576.0))
    print("achievable_gbps %.1f" % achievable)
    if options.out:
        with open(options.out, "w") as handle:
            json.dump({"buffer_mib": options.mb, "runs": results,
                       "achievable_gbps": achievable}, handle, indent=2)


if __name__ == "__main__":
    main()
