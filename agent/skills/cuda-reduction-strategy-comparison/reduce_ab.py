"""A/B the CUDA reduction strategies on a set of shapes, in one process.

Produces the two things a reduction change has to be judged on together: the
device time of each strategy, and each strategy's error against a float64
reference.  Reduction changes alter the summation order, so a timing table on
its own is worthless -- a kernel that drops terms is usually the fastest one.

    PYTHONPATH=<worktree>/python JITTOR_HOME=... TMPDIR=... \
        CUDA_VISIBLE_DEVICES=<card> nvcc_path=/usr/local/cuda/bin/nvcc \
        PATH=/usr/local/cuda/bin:$PATH taskset -c <cores> \
        python reduce_ab.py --shapes unet

``--shapes unet`` is every reduction the code generator emits for one step of
``large_diffusers_unet2d``; ``--shapes representative`` is the four synthetic
shapes the skill's older table uses.  The two answer differently, which is why
both are here.

Do not try to judge a reduction strategy from a whole-network gradient diff:
the default warp path finishes with one atomic per warp, so its own run-to-run
spread on a UNet step is larger than the difference between the strategies
(measured: worst per-tensor max|d|/max|g| 1.8 between two runs of the *same*
strategy, 1.6 between the two strategies).  Per-shape against a higher-precision
reference is the only oracle that resolves anything.
"""

import argparse
import os

import numpy as np

import jittor as jt

# (shape, reduced dims, op) -- read off the profiler with
# profiler_record_shape=1, see the caliber section of the SKILL.
UNET = (
    ((4, 256, 384), (0, 1), "add"),        # linear bias gradients, 24 per step
    ((4, 128, 64, 64), (2, 3), "add"),     # time-embedding broadcast gradients
    ((4, 384, 16, 16), (2, 3), "add"),
    ((4, 256, 32, 32), (2, 3), "add"),
    ((4, 384), (0,), "add"),               # time-embedding linear bias gradient
    ((4, 384, 256), (0, 2), "add"),
    ((4, 32, 12, 256), (2, 3), "mean"),    # the six attention GroupNorms that
                                           # fall back to the code generator
)

REPRESENTATIVE = (
    ((8, 384, 32, 32), (0, 2, 3), "add"),
    ((8, 128, 64, 64), (0, 2, 3), "add"),
    ((16, 192, 32, 32), (0, 2, 3), "add"),
    ((32, 64, 56, 56), (0, 2, 3), "add"),
)

SETS = {"unet": UNET, "representative": REPRESENTATIVE}


def run_one(value, shape, dims, op, level, tag, repeats):
    """Return (device us per call, relative error, generated source)."""
    jt.flags.para_opt_level = level
    x = jt.array(value)
    x.sync()
    # para_opt_level does not enter the jit key: without a distinct compile
    # option the second strategy silently reuses the first one's kernel
    options = {"reduce_ab": tag}
    jt.reduce(x, op, dims).sync()          # compile outside the measurement
    jt.sync_all(True)
    with jt.profile_scope(rerun=0, compile_options=options) as report:
        for _ in range(repeats):
            jt.reduce(x, op, dims).sync()
        jt.sync_all(True)
    header, rows = report[0], report[1:]
    name, fastest, path = (header.index(c) for c in ("Name", "MinTime", "FileName"))
    micros, source = 0.0, ""
    for row in rows:
        if "reduce" not in row[name]:
            continue
        # best of `repeats`, not the mean: with a dozen agents on the box the
        # mean is dominated by whichever iteration lost the GPU, and the first
        # profiled iteration is regularly 20x the rest
        micros += float(row[fastest]) / 1e3
        source = open(row[path]).read()
    got = jt.reduce(jt.array(value), op, dims).numpy()
    reference = (value.astype("float64").sum(axis=tuple(dims)) if op == "add"
                 else value.astype("float64").mean(axis=tuple(dims)))
    scale = max(float(np.abs(reference).max()), 1e-30)
    error = float(np.abs(got.reshape(reference.shape) - reference).max()) / scale
    return micros, error, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", choices=sorted(SETS), default="unet")
    parser.add_argument("--repeats", type=int, default=30)
    options = parser.parse_args()

    print("tree", os.path.dirname(jt.__file__))
    jt.flags.use_cuda = 1

    print("%-22s %6s %10s %10s %8s %10s %10s"
          % ("shape", "dims", "warp_us", "block_us", "block/warp",
             "warp_err", "block_err"))
    totals = [0.0, 0.0]
    for index, (shape, dims, op) in enumerate(SETS[options.shapes]):
        value = np.random.RandomState(index).randn(*shape).astype("float32")
        warp_us, warp_err, warp_src = run_one(
            value, shape, dims, op, 3, 2 * index, options.repeats)
        block_us, block_err, block_src = run_one(
            value, shape, dims, op, 4, 2 * index + 1, options.repeats)
        # judge from the generated source, not from the flag
        assert "_wr_mask" in warp_src and "shared_reduce<" not in warp_src
        assert "shared_reduce<" in block_src and "_wr_mask" not in block_src
        totals[0] += warp_us
        totals[1] += block_us
        print("%-22s %6s %10.2f %10.2f %8.3f %10.2g %10.2g"
              % (str(shape), "".join(str(d) for d in dims), warp_us, block_us,
                 block_us / warp_us, warp_err, block_err))
    print("%-22s %6s %10.2f %10.2f %8.3f"
          % ("TOTAL", "", totals[0], totals[1], totals[1] / totals[0]))


if __name__ == "__main__":
    main()
