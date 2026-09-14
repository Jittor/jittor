"""Reduction throughput across working-set sizes, with an addition control.

A NaN-propagating ``max`` was implemented, measured, and backed out once: the
bit test it needed was not a reduction pattern the compiler recognises, so the
loop stopped vectorising and it was quoted at 7.1-7.5x in-tree. That
measurement was made by hand because the suite had no reduction benchmark, and
when this harness was used to re-take it the figure came out at 1.9-2.0x -- the
baseline had moved to ``-O3`` and the expression had become a single branchless
select. The NaN-correct ``max`` is now shipped at that price, and KI-OPS-006 is
what remains of the cost.

So this file is the reason a claimed 7x turned out to be a 2x. Three things
make it answerable rather than merely a timing:

* **sizes that cross the cache.** The regression was 10-28x in cache and
  7.4-8.2x out of it; one working set would have reported whichever number the
  machine happened to give.
* **an addition control.** ``sum`` reduces the same bytes with a pattern that
  was never in question, so a slowdown that shows up in both is the machine,
  not the change. The hand measurement leaned on exactly this to trust its 7x.
* **bytes per second, not seconds.** A reduction is memory-bound at these
  sizes, so throughput is the quantity that stays comparable when the sizes or
  the machine change.
"""

from __future__ import annotations

import numpy as np

from ._shared import (
    as_numpy,
    backend_tensor,
    cleanup_backend,
    load_backend,
    synchronize,
)


#: In cache, around it, and well past it on any current card or host.
ELEMENTS = {"1M": 1 << 20, "16M": 1 << 24, "64M": 1 << 26}


class ReductionBenchmarks:
    params = (
        ["jittor", "torch"],
        ["cpu", "cuda"],
        ["max", "min", "sum"],
        ["1M", "16M", "64M"],
    )
    param_names = ["backend", "device", "reduction", "size"]
    number = 1
    repeat = (3, 7, 30.0)
    rounds = 1
    timeout = 300

    def setup(self, backend_name, device, reduction, size):
        self.backend_name = backend_name
        self.device = device
        self.reduction = reduction
        self.nbytes = ELEMENTS[size] * 4
        self.backend = load_backend(backend_name, device)
        rng = np.random.default_rng(20260909)
        host = rng.standard_normal(ELEMENTS[size]).astype("float32")
        self.x = backend_tensor(backend_name, self.backend, host, device)
        synchronize(backend_name, self.backend, device)

        # A reduction that returned the identity element would time beautifully.
        value = float(as_numpy(backend_name, self._run()))
        expected = {"max": host.max(), "min": host.min(), "sum": host.sum()}[reduction]
        tolerance = abs(float(expected)) * 1e-3 + 1e-2
        if not np.isfinite(value) or abs(value - float(expected)) > tolerance:
            raise RuntimeError(
                "%s returned %r, not the reduction of its input (%r)"
                % (reduction, value, float(expected))
            )
        self._assert_iterations_are_not_elided(backend_name, device)

    def _assert_iterations_are_not_elided(self, backend_name, device):
        """Ten reductions must take about ten times as long as one.

        The lazy graph makes this checkable and necessary: a loop whose results
        are each overwritten and which synchronises only at the end executes one
        reduction, not ten, and the throughput is then multiplied by ten. That
        draft read 3598 GB/s from a card whose memory tops out near 1008. The
        wrong number is not obviously wrong unless someone knows the hardware,
        so the harness checks its own linearity instead of relying on that.
        """
        import time

        def elapsed(rounds):
            self._keep = self._run()
            synchronize(backend_name, self.backend, device)
            start = time.perf_counter()
            for _ in range(rounds):
                self._keep = self._run()
                synchronize(backend_name, self.backend, device)
            return time.perf_counter() - start

        one = elapsed(1)
        ten = elapsed(10)
        if one <= 0:
            raise RuntimeError("a single reduction timed as instantaneous")
        ratio = ten / one
        if not 4.0 <= ratio <= 25.0:
            raise RuntimeError(
                "ten reductions took %.2fx one, not about ten: the timed loop is "
                "not executing every iteration, so any throughput it reports is "
                "multiplied by iterations that never ran" % ratio)

    def _run(self):
        backend = self.backend
        if self.backend_name == "torch":
            with backend.no_grad():
                return getattr(self.x, self.reduction)()
        return getattr(backend, self.reduction)(self.x)

    def time_reduce(self, backend_name, device, reduction, size):
        self._keep = self._run()
        synchronize(backend_name, self.backend, device)

    def track_bytes_per_second(self, backend_name, device, reduction, size):
        """Throughput, so the number survives a change of size or machine.

        Timed here rather than derived from ``time_reduce`` because ASV's timing
        and tracking runs are separate; a ratio built from two different runs
        would drift for reasons that have nothing to do with the kernel.
        """
        import time

        rounds = 5
        self._keep = self._run()
        synchronize(backend_name, self.backend, device)
        start = time.perf_counter()
        for _ in range(rounds):
            self._keep = self._run()
            # Synchronised every iteration, not once at the end. Jittor is lazy and
            # the previous result was just overwritten, so a loop that syncs
            # only afterwards executes *one* reduction and is then divided by
            # five: the first draft reported 3598 GB/s on a card whose memory
            # tops out near 1008. A benchmark that proves throughput the
            # hardware cannot deliver is worse than no benchmark.
            synchronize(backend_name, self.backend, device)
        elapsed = time.perf_counter() - start
        if elapsed <= 0:
            raise RuntimeError("the reduction timed as instantaneous")
        return float(self.nbytes * rounds) / elapsed

    track_bytes_per_second.unit = "bytes/s"

    def teardown(self, backend_name, device, reduction, size):
        backend = getattr(self, "backend", None)
        for name in ("x", "_keep"):
            if hasattr(self, name):
                delattr(self, name)
        if backend is not None:
            cleanup_backend(backend_name, backend)
