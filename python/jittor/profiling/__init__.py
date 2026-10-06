"""``jt.profile``: where one step's time and memory go.

One entry point for the questions a performance investigation has to answer
about a region of a program -- usually one training or inference step::

    with jt.profile() as prof:
        train_step()
    print(prof.summary())                 # host split, device split, tables, memory
    prof.export_chrome_trace("step.json") # chrome://tracing or Perfetto

What it measures, and how:

* **Host**: the wall time of the region split into Python + graph
  construction, executor planning, JIT compilation, operator launch, waiting
  on the device, and device-graph launches, from timestamps the executor
  records (``src/runtime/profiler/step_trace.h``). The parts sum to the wall
  time. With ``python=True`` a sampler also charges host time to Python call
  sites.
* **Device**: every kernel, copy and memset with its device timestamps, from
  CUPTI, each attributed to the operator that launched it. Nothing is
  synchronized or re-run per operator, so the step keeps its normal overlap
  of host and device work. Without CUPTI the report says device time is
  unavailable and falls back to the host's device-wait share for the
  host-/device-bound verdict, labelled as an estimate.
* **Memory** (``memory=True``): the allocator's exact high-water mark and the
  set of allocations live at that moment, grouped by producing operator,
  Python call site and tensor; pool reservation, caching and fragmentation;
  and what NVML says the process holds.
* **Replay**: device-graph launches (``jt.graph_replay`` recordings) appear as
  launches in the trace and are counted, and registered replay objects report
  their counter deltas and refusal reasons.

By default the region is bracketed by ``jt.sync_all(True)``, so work queued
before it does not leak in and work built inside it is finished (and timed)
before the report is made. Pass ``sync=False`` to leave both edges alone.

The older tools remain: ``jt.profile_scope``/``jt.profiler`` is a per-operator
kernel micro-benchmark (it synchronizes and optionally re-runs each operator),
and ``profile_memory_enable`` with ``jt.get_max_memory_treemap`` a
``trace_py_var``-based memory tree. See ``docs/notes/profiling.md``.
"""

import time
import weakref

import jittor_core as _core

from . import _cupti
from ._memory import MemoryReport
from ._report import ProfileResult
from ._sampler import NvmlSampler, PythonSampler

__all__ = ["profile", "Profile", "ProfileResult", "MemoryReport", "record_function",
           "register_replay_source"]

_ST_OPS, _ST_MEMORY, _ST_SHAPES = 1, 2, 4
_replay_sources = weakref.WeakSet()
_active = []


def register_replay_source(obj):
    """Report ``obj.stats`` (a dict of counters) and its refusal reason in
    every profile it is alive for. ``jt.graph_replay`` wrappers register
    themselves; other capture/replay mechanisms can call this."""
    _replay_sources.add(obj)
    return obj


def _layout():
    fields = {}
    for part in _core.step_trace_layout().split(";"):
        kind, _, names = part.partition(":")
        fields[kind] = names.split(",")
    return fields


def _records(flat, names):
    n = len(names)
    return [dict(zip(names, flat[i:i + n])) for i in range(0, len(flat), n)]


def _refusal(obj):
    for attr in ("refused", "_graph_refused", "refusal"):
        value = getattr(obj, attr, None)
        if value:
            return value
    return None


class Profile:
    """A profiling session; use :func:`profile` to make one."""

    def __init__(self, device="auto", memory=True, python=False, shapes=True, sync=True,
                 sample_interval=0.001, nvml=True):
        if device not in ("auto", True, False):
            raise ValueError("device must be 'auto', True or False, not %r" % (device,))
        self.want_device = device
        self.want_memory = bool(memory)
        self.want_python = bool(python)
        self.want_shapes = bool(shapes)
        self.sync = bool(sync)
        self.sample_interval = sample_interval
        self.want_nvml = bool(nvml)
        self.result = None
        self._steps = []

    # -- lifecycle ------------------------------------------------------------
    def __enter__(self):
        import jittor as jt
        if _core.step_trace_active():
            raise RuntimeError("a jt.profile region is already open; profiles do not nest")
        if self.sync:
            jt.sync_all(True)
        self._device_index = int(jt.core.current_device()) if jt.flags.use_cuda else -1
        self._cupti = None
        self._device_note = None
        if self.want_device is False:
            self._device_note = "device timing disabled (device=False)"
        elif self._device_index < 0:
            self._device_note = "the region runs on the host (jt.flags.use_cuda is 0)"
        else:
            self._cupti = _cupti.load()
            if self._cupti is None:
                if self.want_device is True:
                    raise RuntimeError(_cupti.why_unavailable())
                self._device_note = _cupti.why_unavailable()
        self._sources = [(obj, dict(getattr(obj, "stats", {}) or {})) for obj in list(_replay_sources)]
        self._graph_launches = _core.graph_launch_count()
        self._nvml = None
        if self.want_memory:
            _core.reset_device_memory_peak(self._device_index)
            self._mem_start = list(_core.device_pool_stats(self._device_index))
            if self.want_nvml and self._device_index >= 0:
                self._nvml = NvmlSampler(self._device_index).start()
        if self._cupti is not None:
            self._cupti.start()
            _core.step_trace_set_correlation(self._cupti.push, self._cupti.pop)
            self._offset = self._clock_offset()
        mode = _ST_OPS | (_ST_MEMORY if self.want_memory else 0) | (_ST_SHAPES if self.want_shapes else 0)
        _core.step_trace_start(mode)
        self._sampler = PythonSampler(self.sample_interval).start() if self.want_python else None
        _active.append(self)
        self._t0 = _core.step_trace_now()
        if self._sampler is not None:
            self._sampler.armed = True
        return self

    def __exit__(self, *exc):
        import jittor as jt
        try:
            if self.sync and exc[0] is None:
                jt.sync_all(True)
        finally:
            t1 = _core.step_trace_now()
            _active.remove(self)
            if self._sampler is not None:
                self._sampler.armed = False
                self._sampler.stop()
            _core.step_trace_stop()
            if self._cupti is not None:
                _core.step_trace_set_correlation(0, 0)
                kernels, copies, correlation = self._cupti.stop()
            else:
                kernels, copies, correlation = [], [], {}
            if self._nvml is not None:
                self._nvml.stop()
        if exc[0] is None:
            self.result = self._build(t1, kernels, copies, correlation)
        return False

    def step(self):
        """Mark a step boundary (shown in the trace, counted in the summary)."""
        self._steps.append(_core.step_trace_now())

    def _clock_offset(self):
        best = None
        for _ in range(5):
            a = _core.step_trace_now()
            c = self._cupti.timestamp()
            b = _core.step_trace_now()
            if best is None or b - a < best[0]:
                best = (b - a, (a + b) // 2 - c)
        return best[1]

    # -- results ----------------------------------------------------------------
    def _build(self, t1, kernels, copies, correlation):
        layout = _layout()
        strings = list(_core.step_trace_strings())
        ops = _records(_core.step_trace_ops(), layout["ops"])
        batches = _records(_core.step_trace_batches(), layout["batches"])
        waits = _records(_core.step_trace_waits(), layout["waits"])
        truncated = layout.get("truncated") == ["1"]
        t0 = self._t0
        off = getattr(self, "_offset", 0)
        kernel_records = [
            {"start": s + off, "end": e + off, "device": d, "stream": st, "correlation": c,
             "name": n, "graph": g}
            for s, e, d, st, c, n, g in kernels if e + off >= t0 and s + off <= t1]
        copy_records = [
            {"start": s + off, "end": e + off, "device": d, "stream": st, "correlation": c,
             "name": n, "bytes": nb}
            for s, e, d, st, c, n, nb in copies if e + off >= t0 and s + off <= t1]
        memory = None
        if self.want_memory:
            events = _records(_core.step_trace_memory(), layout["memory"])
            nv = {"error": None}
            if self._nvml is None:
                nv["error"] = "not requested" if not self.want_nvml else "host region"
            elif self._nvml.error:
                nv["error"] = self._nvml.error
            else:
                nv.update(start=self._nvml.start_bytes, end=self._nvml.end_bytes,
                          peak=self._nvml.peak_bytes, samples=self._nvml.samples)
            memory = MemoryReport(
                self._device_index, events, strings, self._mem_start,
                list(_core.device_pool_stats(self._device_index)),
                (_core.device_memory_peak(self._device_index),
                 _core.device_memory_reserved_peak(self._device_index)), nv, t0)
        sources = []
        for obj, before in self._sources:
            after = dict(getattr(obj, "stats", {}) or {})
            delta = {k: after.get(k, 0) - before.get(k, 0) for k in after
                     if isinstance(after.get(k), (int, float))}
            if any(delta.values()):
                sources.append(("%s@%x" % (type(obj).__name__, id(obj)), delta, _refusal(obj)))
        replay = {"graph_launches": _core.graph_launch_count() - self._graph_launches,
                  "sources": sources}
        return ProfileResult(
            t0=t0, t1=t1, strings=strings, ops=ops, batches=batches, waits=waits,
            kernels=kernel_records, copies=copy_records, correlation=correlation,
            device_note=self._device_note, memory=memory, python=self._sampler,
            replay=replay, steps=list(self._steps), truncated=truncated,
            on_host=self._device_index < 0)

    def _require(self):
        if self.result is None:
            raise RuntimeError("the profile has no result yet: leave the `with jt.profile()` block first")
        return self.result

    def summary(self, row_limit=12):
        return self._require().summary(row_limit)

    def table(self, sort_by="device_time", row_limit=20, group_by_shapes=False):
        return self._require().table(sort_by, row_limit, group_by_shapes)

    def export_chrome_trace(self, path):
        return self._require().export_chrome_trace(path)

    def __getattr__(self, name):
        # host, device, memory, bound, op_stats, kernel_table, ... of the result
        if name.startswith("_") or name == "result":
            raise AttributeError(name)
        return getattr(self._require(), name)


def profile(device="auto", memory=True, python=False, shapes=True, sync=True,
            sample_interval=0.001, nvml=True):
    """Profile a region: ``with jt.profile() as prof: ...; print(prof.summary())``.

    Args:
        device: ``"auto"`` records device activity with CUPTI when it can be
            loaded, ``True`` requires it, ``False`` skips it.
        memory: trace allocator events and report the peak and what was live.
        python: sample host time per Python call site (statistical; lowers
            the interpreter switch interval while running).
        shapes: record the input shapes of every launched operator.
        sync: bracket the region with ``jt.sync_all(True)``.
        sample_interval: seconds between Python samples.
        nvml: also read the process's device memory from NVML.
    """
    return Profile(device=device, memory=memory, python=python, shapes=shapes, sync=sync,
                   sample_interval=sample_interval, nvml=nvml)


class record_function:
    """A named host range in the current profile (a no-op outside one).

    Usable as a context manager or decorator. Device work launched *while the
    range is open* -- eager launches, device-graph launches -- is attributed
    to it; lazily built operators run where the graph is executed, and their
    rows carry the Python call site that built them instead.
    """

    def __init__(self, name):
        self.name = str(name)
        self._id = -1

    def __enter__(self):
        self._id = _core.step_trace_range_begin(self.name)
        return self

    def __exit__(self, *exc):
        _core.step_trace_range_end(self._id)
        self._id = -1
        return False

    def __call__(self, fn):
        import functools

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with record_function(self.name):
                return fn(*args, **kwargs)
        return wrapper
