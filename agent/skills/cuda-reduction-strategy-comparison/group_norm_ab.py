"""Weighted before/after measurement of the hand-written CUDA GroupNorm.

Why not a single synthetic shape: the family costs what it costs on the shape
mix the workload actually uses.  ``SHAPES`` below is the exact mix
``large_diffusers_unet2d`` submits in one step (12 distinct shapes, 35 calls),
read off a ``profiler_record_shape=1`` report -- see the skill's "weighted"
section.  Timing one 4x128x64x64 call and multiplying by 35 gives a different
answer.

Three kinds of evidence, all in one run, because a change to a hand-written
kernel can only be accepted or rejected on all three:

``--what time``    device time of the three GroupNorm kernels, weighted by the
                   per-step call counts, from ``jt.profile_scope``.  Prints the
                   call count next to it: a call count that moved between two
                   revisions means the two runs are not the same work.
``--what memory``  bytes the allocator was *asked* for over one weighted step
                   (``use_stat_allocator=2``, which sits above the caching
                   allocator and therefore counts every var, not just the
                   device-level cache misses).  This is the measure of "one
                   fewer full-size intermediate": it is exact and does not move
                   between runs.
``--what accuracy`` float32 forward/backward against a float64 NumPy reference.

The fast path is asserted, not assumed: ``_group_norm_cuda`` returns ``None``
when ``_supports_group_norm`` rejects the arguments, and a run that silently
fell back to the code generator would report a perfectly plausible number for
a kernel that never ran.  Same for the profiler: if no record carries a
GroupNorm symbol the script fails instead of reporting 0.0 us.
"""

import argparse
import os
import sys

import numpy as np


def _report_rows(rep):
    header, rows = rep[0], rep[1:]
    index = {name: i for i, name in enumerate(header)}
    return index, rows


# (batch, channels, height, width, num_groups) -> calls per UNet step.
SHAPES = [
    ((4, 128, 64, 64), 32, 8),
    ((4, 384, 64, 64), 32, 1),
    ((4, 256, 64, 64), 32, 2),
    ((4, 256, 32, 32), 32, 6),
    ((4, 384, 16, 16), 32, 10),
    ((4, 640, 32, 32), 32, 1),
    ((4, 512, 32, 32), 32, 1),
    ((4, 768, 16, 16), 32, 2),
    ((4, 384, 32, 32), 32, 1),
    ((4, 640, 16, 16), 32, 1),
    ((4, 128, 32, 32), 32, 1),
    ((4, 256, 16, 16), 32, 1),
]


def build(jt, shape, seed):
    rng = np.random.RandomState(seed)
    x = jt.array(rng.randn(*shape).astype("float32"))
    weight = jt.array(rng.randn(shape[1]).astype("float32"))
    bias = jt.array(rng.randn(shape[1]).astype("float32"))
    cotangent = jt.array(rng.randn(*shape).astype("float32"))
    return x, weight, bias, cotangent


def one_step(jt, kernel, tensors, groups):
    """Forward plus backward for one shape, synced."""
    x, weight, bias, cotangent = tensors
    y = kernel(x, groups, weight, bias, 1e-5)
    assert y is not None, "the hand-written CUDA fast path refused this shape"
    grads = jt.grad((y * cotangent).sum(), [x, weight, bias])
    jt.sync([y] + list(grads), device_sync=True)
    return y, grads


def measure_time(jt, kernel, repeats):
    prepared = [(shape, groups, calls, build(jt, shape, 7 + i))
                for i, (shape, groups, calls) in enumerate(SHAPES)]
    for shape, groups, _, tensors in prepared:      # compile outside the timing
        one_step(jt, kernel, tensors, groups)
    jt.sync_all(True)

    with jt.profile_scope(warmup=0, rerun=0) as rep:
        for _ in range(repeats):
            for shape, groups, calls, tensors in prepared:
                for _ in range(calls):
                    one_step(jt, kernel, tensors, groups)
        jt.sync_all(True)

    index, rows = _report_rows(rep)
    totals = {}
    for row in rows:
        key = row[index["Name"]]
        for symbol in ("group_norm_forward", "group_norm_backward"):
            if symbol in key:
                bucket = totals.setdefault(symbol, [0, 0.0])
                bucket[0] += int(row[index["Count"]])
                bucket[1] += float(row[index["TotalTime"]])
                break
    assert totals, (
        "no profiler record carries a GroupNorm symbol -- the fast path did "
        "not run, or the kernels were renamed; refusing to report 0.0 us")
    print("%-24s %8s %12s" % ("kernel", "calls", "step_us"))
    grand = 0.0
    for symbol in sorted(totals):
        count, total_ns = totals[symbol]
        print("%-24s %8.0f %12.1f"
              % (symbol, count / repeats, total_ns / 1e3 / repeats))
        grand += total_ns / 1e3 / repeats
    print("%-24s %8.0f %12.1f"
          % ("GROUPNORM TOTAL",
             sum(c for c, _ in totals.values()) / repeats, grand))


def measure_memory(jt):
    """Bytes requested from the allocator by the GroupNorm operators alone.

    ``use_stat_allocator=2`` puts the counter *above* the caching allocator, so
    it counts what the graph asked for rather than what missed the device
    cache; that is the quantity "one fewer full-size intermediate" is a
    statement about, and unlike a device-level figure it is bit-stable.

    The two ``jt.code`` operators are driven directly through the ``Function``
    rather than through ``jt.grad`` of a loss, because a loss such as
    ``(y * cotangent).sum()`` allocates full-size intermediates of its own and
    would bury the very quantity being measured.
    """
    from jittor.backends.cuda.kernels.nn.group_norm_cuda import (
        _group_norm_cuda_cls)

    def run(function, tensors):
        x, weight, bias, cotangent = tensors
        y = function.execute(x, weight, bias)
        grads = function.grad(cotangent)
        jt.sync([y] + list(grads), device_sync=True)

    prepared = []
    for i, (shape, groups, calls) in enumerate(SHAPES):
        tensors = build(jt, shape, 7 + i)
        cls = _group_norm_cuda_cls(shape, groups, 1e-5)
        run(cls(), tensors)                          # compile outside the count
        prepared.append((shape, calls, cls, tensors))
    jt.sync_all(True)

    element_bytes = 4 * sum(int(np.prod(shape)) * calls
                            for shape, _, calls in SHAPES)
    jt.flags.use_stat_allocator = 2                  # enabling resets counters
    for shape, calls, cls, tensors in prepared:
        for _ in range(calls):
            run(cls(), tensors)
    jt.sync_all(True)
    allocated = int(jt.flags.stat_allocator_total_alloc_byte)
    calls_made = int(jt.flags.stat_allocator_total_alloc_call)
    jt.flags.use_stat_allocator = 0

    print("one full-size copy of the step's GroupNorm tensors %12d B  %7.1f MiB"
          % (element_bytes, element_bytes / 1048576.0))
    print("allocator bytes the GroupNorm ops requested        %12d B  %7.1f MiB"
          % (allocated, allocated / 1048576.0))
    print("allocator calls                                    %12d" % calls_made)
    print("full-size copies allocated per step                %12.3f"
          % (allocated / float(element_bytes)))


def measure_accuracy(jt, kernel):
    """float32 kernel against a float64 NumPy forward and backward."""
    worst = 0.0
    for shape, groups, _ in SHAPES[:4]:
        rng = np.random.RandomState(hash(shape) & 0xFFFF)
        x_np = rng.randn(*shape).astype("float32")
        w_np = rng.randn(shape[1]).astype("float32")
        b_np = rng.randn(shape[1]).astype("float32")
        cot_np = rng.randn(*shape).astype("float32")

        x = jt.array(x_np)
        weight = jt.array(w_np)
        bias = jt.array(b_np)
        y = kernel(x, groups, weight, bias, 1e-5)
        assert y is not None
        grads = jt.grad((y * jt.array(cot_np)).sum(), [x, weight, bias])
        got = jt.fetch_sync([y] + list(grads))

        for name, value, reference in zip(
                ("y", "dx", "dweight", "dbias"), got,
                reference_float64(x_np, groups, w_np, b_np, cot_np, 1e-5)):
            scale = max(float(np.abs(reference).max()), 1e-30)
            error = float(np.abs(value - reference).max()) / scale
            worst = max(worst, error)
            print("%-18s %-8s relerr %.3e" % (str(shape), name, error))
    print("worst relative error %.3e" % worst)


def reference_float64(x_np, groups, w_np, b_np, cot_np, eps):
    batch, channels, height, width = x_np.shape
    x = x_np.astype("float64").reshape(batch, groups, -1)
    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + eps)
    xhat = ((x - mean) * rstd).reshape(x_np.shape)
    weight = w_np.astype("float64").reshape(1, channels, 1, 1)
    bias = b_np.astype("float64").reshape(1, channels, 1, 1)
    y = xhat * weight + bias

    cot = cot_np.astype("float64")
    dweight = (cot * xhat).sum((0, 2, 3))
    dbias = cot.sum((0, 2, 3))
    g = (cot * weight).reshape(batch, groups, -1)
    xhat_g = xhat.reshape(batch, groups, -1)
    mean_g = g.mean(-1, keepdims=True)
    mean_gx = (g * xhat_g).mean(-1, keepdims=True)
    dx = (rstd * (g - mean_g - xhat_g * mean_gx)).reshape(x_np.shape)
    return y, dx, dweight, dbias


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=("time", "memory", "accuracy"),
                        default="time")
    parser.add_argument("--repeats", type=int, default=3)
    options = parser.parse_args()

    import jittor as jt
    print("jittor from %s" % os.path.dirname(jt.__file__))
    assert jt.has_cuda, "this measures a CUDA kernel; nvcc_path must be set"
    jt.flags.use_cuda = 1
    from jittor.backends.cuda.kernels.nn.group_norm_cuda import _group_norm_cuda

    if options.what == "time":
        measure_time(jt, _group_norm_cuda, options.repeats)
    elif options.what == "memory":
        measure_memory(jt)
    else:
        measure_accuracy(jt, _group_norm_cuda)


if __name__ == "__main__":
    sys.exit(main())
