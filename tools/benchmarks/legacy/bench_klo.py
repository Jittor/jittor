"""Measure CUDA kernel-launch overhead by launching an empty kernel in a loop.

Legacy standalone script, kept for the one number it produces. Like
``inference_perf.py`` next to it, it does not import Jittor until ``main()``
runs, so importing this module costs nothing.

The launches happen inside a single ``jt.code`` op and the op is ``sync()``ed
inside a ``profile_scope``, so the lazy graph cannot elide the work being
timed -- which is the usual way a hand-written Jittor micro-benchmark ends up
measuring nothing.

    python tools/benchmarks/legacy/bench_klo.py [launches]
"""

import sys


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    n = int(argv[0]) if argv else 100000

    import jittor as jt

    with jt.flag_scope(use_device=1):
        with jt.profile_scope(10, 10) as rep:
            jt.code([2], "float32", [],
                    cuda_header='''__global__ void kernel(float* a) {}''',
                    cuda_src=f'''
    for (int i=0; i<{n}; i++) kernel<<<1,32>>>(out0_p);
    ''').sync()

    avg_ns = float(rep[1][4]) / n
    print("kernel launch overhead(ns):", avg_ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
