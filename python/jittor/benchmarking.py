"""Reliable timing for Jittor's lazy execution model."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import statistics
import time
from typing import Callable, Tuple


_timer = time.perf_counter


@dataclass(frozen=True)
class BenchmarkResult:
    """Immutable timing samples returned by :func:`benchmark`.

    All durations and derived statistics are measured in seconds.
    """

    samples: Tuple[float, ...]
    warmup: int
    input_count: int

    @property
    def repeat(self):
        return len(self.samples)

    @property
    def median(self):
        return statistics.median(self.samples)

    @property
    def mean(self):
        return sum(self.samples) / len(self.samples)

    @property
    def minimum(self):
        return min(self.samples)

    @property
    def maximum(self):
        return max(self.samples)

    @property
    def total(self):
        return sum(self.samples)

    @property
    def unit(self):
        return "seconds"


def _positive_count(name, value):
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("%s must be an integer" % name)
    if value < 1:
        raise ValueError("%s must be at least 1" % name)
    return value


def _collect_vars(value):
    # Import lazily: benchmarking is published while jittor itself is still
    # initializing, after the native Var type has become available.
    from . import Var

    variables = []
    seen_vars = set()
    seen_containers = set()

    def visit(item):
        if isinstance(item, Var):
            identity = id(item)
            if identity not in seen_vars:
                seen_vars.add(identity)
                variables.append(item)
            return

        if isinstance(item, Mapping):
            identity = id(item)
            if identity in seen_containers:
                return
            seen_containers.add(identity)
            for child in item.values():
                visit(child)
            return

        if isinstance(item, (tuple, list)):
            identity = id(item)
            if identity in seen_containers:
                return
            seen_containers.add(identity)
            for child in item:
                visit(child)

    visit(value)
    return variables


def _sync_vars(variables):
    if not variables:
        return
    from . import sync

    sync(variables, device_sync=True, weak_sync=False)


def _run_and_materialize(function, slot):
    output = function(slot)
    variables = _collect_vars(output)
    if not variables:
        raise TypeError(
            "benchmark callable must return at least one jittor.Var; "
            "return every output that the measurement must materialize"
        )
    _sync_vars(variables)


def benchmark(function: Callable, input_pool: Sequence, *, warmup=1, repeat=10):
    """Measure fully materialized invocations of ``function``.

    ``input_pool`` must be a non-empty concrete sequence. It is snapshotted
    before execution, materialized outside the timed region, and then selected
    round-robin. ``function`` receives one pool entry and must return a Var or a
    nested tuple/list/dict containing every Var that needs to execute.

    At least one warmup invocation is required so first-use compilation is not
    included. Each timed invocation ends with a target-specific device sync;
    the returned samples therefore include Python graph construction and actual
    execution rather than only lazy graph submission.
    """

    if not callable(function):
        raise TypeError("function must be callable")
    if isinstance(input_pool, (str, bytes)) or not isinstance(input_pool, Sequence):
        raise TypeError("input_pool must be a concrete sequence")

    pool = tuple(input_pool)
    if not pool:
        raise ValueError("input_pool must not be empty")

    warmup = _positive_count("warmup", warmup)
    repeat = _positive_count("repeat", repeat)

    # Input construction, host-to-device copies, and their first materialization
    # are setup costs, not work performed by the callable under measurement.
    input_vars = _collect_vars(pool)
    _sync_vars(input_vars)

    for index in range(warmup):
        _run_and_materialize(function, pool[index % len(pool)])

    samples = []
    for index in range(repeat):
        started = _timer()
        _run_and_materialize(function, pool[(warmup + index) % len(pool)])
        samples.append(_timer() - started)

    return BenchmarkResult(tuple(samples), warmup, len(pool))


__all__ = ["BenchmarkResult", "benchmark"]
