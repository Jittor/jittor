#!/usr/bin/env python3
"""Where the time of building one forward graph goes, split by phase (task 3.21).

Only the wall time is measurable without a special build; the split needs the
core's phase probes, which are compiled in by ``JT_GRAPH_BUILD_PROFILE=1``
(see ``src/utils/graph_build_profile.h``).  Without them this still prints the
totals and says the split is unavailable, rather than printing zeros.

    cd <repo>
    JT_GRAPH_BUILD_PROFILE=1 JITTOR_HOME=... TMPDIR=... CUDA_VISIBLE_DEVICES=N \
    nvcc_path=/usr/local/cuda/bin/nvcc PATH=/usr/local/cuda/bin:$PATH \
    PYTHONPATH=<repo>/python taskset -c ... python \
        tools/benchmarks/graph_build_phases.py --case unet2d --device cuda

What "building" means here: the Python calls that create the operators, with
nothing executed.  ``--flush 0`` turns off ``auto_flush_ops`` so that no
segment is submitted while the graph is being built; leaving it at the default
measures the pipelined regime instead, in which the build window also contains
execution and the split cannot be read as build cost.

The phases are exclusive (self time), so they partition the time spent inside
the core, and

    wall time - sum(phases)

is the share that never entered the core -- the framework's own Python.  That
term is printed like any other: a report that listed only the phases the plan
named would suggest they were the whole cost.
"""
import argparse
import os
import statistics
import sys
import time


def _percentiles(samples):
    return (min(samples), statistics.median(samples), max(samples))


class Probe:
    """Phase counters, summed over the timed windows only.

    Reset before each window and read straight after it: draining the graph
    between repetitions runs the executor, which assembles jit keys and calls
    into the core, and folding that into the totals is exactly the mistake
    that would make the build look like it costs what a whole step costs.
    """

    def __init__(self, core):
        self.core = core
        self.enabled = bool(core.graph_build_profile_enabled())
        self.names = list(core.graph_build_profile_phases())
        # A build without the probes must not look like a measured zero.
        if self.enabled == (not self.names):
            raise SystemExit(
                "graph_build_profile_enabled() and the phase list disagree: "
                f"enabled={self.enabled} phases={self.names}")
        self.counts = [0] * len(self.names)
        self.nanoseconds = [0] * len(self.names)

    def start(self):
        if self.enabled:
            self.core.graph_build_profile_reset()

    def stop(self):
        if not self.enabled:
            return
        # Nanoseconds first: reading the counts is itself a call into the core,
        # and its scope would land in the second reader's numbers.
        nanoseconds = self.core.graph_build_profile_nanoseconds()
        counts = self.core.graph_build_profile_counts()
        for i in range(len(self.names)):
            self.nanoseconds[i] += nanoseconds[i]
            self.counts[i] += counts[i]

    def rows(self):
        return list(zip(self.names, self.counts, self.nanoseconds))


def case_unet2d(torch, jt, device):
    """The `large_diffusers_unet2d` case of tests/compat/torch/_ecosystem_speed.

    This is the model the 9 ms in the plan came from
    (docs/architecture/pipelined-execution.md).
    """
    sys.path.insert(0, os.path.join(_repo_root(), "tests"))
    from compat.torch import _ecosystem_speed

    builder, _ = _ecosystem_speed.CASES["large_diffusers_unet2d"]
    model, spec = builder(torch)
    if device == "cuda":
        model = model.cuda()
    inputs = {}
    for name, (dtype, shape, high) in spec.items():
        value = (torch.randint(0, high, shape) if dtype == "int64"
                 else torch.randn(*shape))
        inputs[name] = value.cuda() if device == "cuda" else value
    return lambda: model(**inputs).sample


def case_parity_unet(torch, jt, device):
    """The small diffusion UNet of tests/models/_parity_networks."""
    sys.path.insert(0, os.path.join(_repo_root(), "tests"))
    from models._parity_networks import _jittor_unet

    net = _jittor_unet(base=16, groups=4)
    x = jt.random((2, 3, 32, 32))
    t = jt.array([1, 2])
    return lambda: net(x, t)


def case_fanin(torch, jt, device, ops=200, fanin=2):
    """A fixed number of operators whose input-edge count is a knob.

    The control for the edge-table phase: `ops` is held constant while `fanin`
    multiplies only the number of edges, so a probe that attributes correctly
    has to move edge_table and leave the operator-count phases alone.
    """
    sources = [jt.random((4,)) for _ in range(fanin)]

    def build():
        out = []
        for _ in range(ops):
            out.append(jt.code((4,), "float32", sources,
                               cpu_src="@out0(0) = 0;"))
        return out

    return build


def case_chain(torch, jt, device, ops=200):
    """A chain of unary operators: one long fused segment, so one long jit key.

    The control for the jit-key phase, and the reason it needs `--execute`:
    building the graph does not assemble a key at all.
    """
    x = jt.random((4,))

    def build():
        value = x
        for _ in range(ops):
            value = value.abs() + 1.0
        return value

    return build


CASES = {
    "unet2d": case_unet2d,
    "parity-unet": case_parity_unet,
    "fanin": case_fanin,
    "chain": case_chain,
}


def _repo_root():
    # tools/benchmarks/<this file>
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(here))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", default="unet2d", choices=sorted(CASES))
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--reps", type=int, default=10,
                    help="timed repetitions; three is the minimum that gives a range")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--flush", type=int, default=0,
                    help="jt.flags.auto_flush_ops during the timed window")
    ap.add_argument("--ops", type=int, default=200, help="synthetic cases only")
    ap.add_argument("--fanin", type=int, default=2, help="--case fanin only")
    ap.add_argument("--execute", action="store_true",
                    help="also sync inside the timed window, i.e. measure a "
                         "whole step rather than the build")
    options = ap.parse_args()

    import torch  # torch_shim -> jittor, for the diffusers/transformers cases
    import jittor as jt
    print("jittor:", os.path.dirname(jt.__file__))

    if options.device == "cuda":
        jt.flags.use_cuda = 1
    print("has_cuda:", jt.has_cuda, "use_cuda:", jt.flags.use_cuda)

    factory = CASES[options.case]
    kwargs = {}
    if options.case in ("fanin", "chain"):
        kwargs["ops"] = options.ops
    if options.case == "fanin":
        kwargs["fanin"] = options.fanin
    build = factory(torch, jt, options.device, **kwargs)
    jt.sync_all(True)

    # Warm up under the default flags so every kernel the case needs is
    # compiled and cached: a first-time JIT compile inside the timed window
    # would be most of it.
    for _ in range(options.warmup):
        out = build()
        jt.sync_all(True)
        del out
    jt.sync_all(True)

    probe = Probe(jt.core)
    jt.flags.auto_flush_ops = options.flush
    print("auto_flush_ops:", jt.flags.auto_flush_ops,
          "lazy_execution:", jt.flags.lazy_execution,
          "probes:", "on" if probe.enabled else "off")

    wall = []
    executor_runs = 0
    for i in range(options.reps):
        exec_called_before = jt.flags.exec_called
        probe.start()
        started = time.perf_counter()
        out = build()
        if options.execute:
            jt.sync_all(True)
        elapsed = time.perf_counter() - started
        executor_runs += jt.flags.exec_called - exec_called_before
        probe.stop()
        wall.append(elapsed * 1e3)
        if not options.execute:
            # Drain outside the timed window, so each repetition starts from
            # the same state a real step would.
            jt.sync_all(True)
        del out

    low, mid, high = _percentiles(wall)
    print()
    print(f"case={options.case} device={options.device} reps={options.reps}"
          f" execute={options.execute}")
    print("wall: min %.3f  median %.3f  max %.3f ms" % (low, mid, high))
    print("executor runs inside the timed window:", executor_runs,
          "" if options.execute or executor_runs == 0 else
          "  <-- NOT a build-only measurement")
    if not probe.enabled:
        print("\nno phase split: rebuild with JT_GRAPH_BUILD_PROFILE=1")
        return

    total_ns = sum(wall) * 1e6
    print("\n%-16s %12s %12s %9s %10s" % ("phase", "count", "us/rep", "share", "us/call"))
    accounted = 0
    for name, count, nanoseconds in probe.rows():
        per_rep_us = nanoseconds / 1e3 / options.reps
        share = nanoseconds / total_ns * 100 if total_ns else 0.0
        per_call = nanoseconds / 1e3 / count if count else 0.0
        accounted += nanoseconds
        if nanoseconds:
            print("%-16s %12d %12.1f %8.1f%% %10.3f"
                  % (name, count, per_rep_us, share, per_call))
        else:
            print("%-16s %12d %12s %9s %10s"
                  % (name, count, "-", "-", "-"))
    rest = total_ns - accounted
    print("%-16s %12s %12.1f %8.1f%%"
          % ("core total", "", accounted / 1e3 / options.reps,
             accounted / total_ns * 100))
    print("%-16s %12s %12.1f %8.1f%%   <-- never entered the core"
          % ("python side", "", rest / 1e3 / options.reps,
             rest / total_ns * 100))


if __name__ == "__main__":
    main()
