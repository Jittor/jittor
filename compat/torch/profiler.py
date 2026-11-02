"""``torch.profiler`` and the recording half of ``torch.autograd.profiler``.

Both used to be inert: ``profile`` recorded nothing, ``key_averages()`` did
not exist (or returned an empty list) and ``export_chrome_trace`` wrote no
file, so a script that profiled itself reported success with nothing in hand.
They now run :func:`jittor.profiling.profile` underneath, and the events,
averages and ``table()`` follow PyTorch's (``torch/autograd/profiler_util.py``):
the same attributes, the same self/total arithmetic, the same columns and
formatting.

What the events are:

* one CPU event per launched jittor operator (a fused kernel group is one
  operator) -- its host time is the executor's time to launch it; Python
  graph construction is not charged to operators (``jt.profile``'s host split
  has it) -- and one per ``record_function`` range, which contains the
  operators launched inside it, so "self" excludes them as in PyTorch;
* one CUDA event per kernel and copy CUPTI saw, its time also charged to the
  operator that launched it as that operator's self device time;
* ``profile_memory``: an event's memory is what the pools handed out minus
  what they took back while it ran (self), plus its children's (total);
* ``record_shapes``: input shapes as lists of dimensions;
* ``with_stack``: the Python stack that built the operator, as
  ``"file(line): function"`` entries innermost first, without jittor's own
  frames; ``export_stacks`` writes them for flame graphs;
* ``export_memory_timeline``: the device's allocated bytes over the region,
  all in PyTorch's "Unknown" category -- jittor does not classify
  allocations into parameters, gradients and activations;
* ``schedule``/``on_trace_ready``/``step()`` follow PyTorch's state machine;
  ``with_flops``/``with_modules`` are accepted and change nothing.
"""

from enum import Enum as _Enum
import json
import os
import socket
import time

from jittor import profiling as _profiling

from .profiler_events import (DeviceType, EventList, FunctionEvent, FunctionEventAvg,  # noqa: F401
                              Interval, _build_events)
from .profiler_exports import _kineto_trace, _memory_timeline, _timeline_html


class ProfilerActivity(_Enum):
    CPU = 0
    XPU = 1
    MTIA = 4
    CUDA = 2
    HPU = 3
    PrivateUse1 = 5


class ProfilerAction(_Enum):
    NONE = 0
    WARMUP = 1
    RECORD = 2
    RECORD_AND_SAVE = 3


_RECORDING = (ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE)


def schedule(*, wait, warmup, active, repeat=0, skip_first=0, skip_first_wait=0):
    """PyTorch's cyclic wait/warmup/active schedule."""
    if wait < 0 or warmup < 0 or active <= 0 or repeat < 0 or skip_first < 0:
        raise ValueError("invalid profiler schedule arguments")

    def action(step):
        if step < skip_first:
            return ProfilerAction.NONE
        step -= skip_first
        span = wait + warmup + active
        if skip_first_wait:
            step += wait
        if repeat > 0 and step // span >= repeat:
            return ProfilerAction.NONE
        mod = step % span
        if mod < wait:
            return ProfilerAction.NONE
        if mod < wait + warmup:
            return ProfilerAction.WARMUP
        return ProfilerAction.RECORD if mod < span - 1 else ProfilerAction.RECORD_AND_SAVE
    return action


def tensorboard_trace_handler(dir_name, worker_name=None, use_gzip=False):
    """Write each finished recording as a Chrome trace into ``dir_name``."""
    def handler(prof):
        os.makedirs(dir_name, exist_ok=True)
        name = worker_name or "%s_%d" % (socket.gethostname(), os.getpid())
        path = os.path.join(dir_name, "%s.%d.pt.trace.json" % (name, time.time_ns()))
        prof.export_chrome_trace(path)
        if use_gzip:
            import gzip
            import shutil
            with open(path, "rb") as src, gzip.open(path + ".gz", "wb") as dst:
                shutil.copyfileobj(src, dst)
            os.remove(path)
    return handler


def kineto_available():
    return False


class profile:
    """``torch.profiler.profile`` on :func:`jittor.profiling.profile`."""

    _event_list = EventList

    def __init__(self, activities=None, schedule=None, on_trace_ready=None, record_shapes=False,
                 profile_memory=False, with_stack=False, with_flops=False, with_modules=False,
                 experimental_config=None, execution_trace_observer=None, acc_events=False,
                 use_cuda=None, enabled=True, **_ignored):
        acts = set(activities) if activities is not None else {ProfilerActivity.CPU, ProfilerActivity.CUDA}
        if use_cuda:
            acts.add(ProfilerActivity.CUDA)
        self.activities = acts
        self.schedule = schedule
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.enabled = enabled
        self.step_num = 0
        self.current_action = ProfilerAction.RECORD if schedule is None else schedule(0)
        self._session = None
        self._result = None
        self._events = None

    # -- recording ----------------------------------------------------------------
    def _begin(self):
        device = "auto" if ProfilerActivity.CUDA in self.activities else False
        self._session = _profiling.profile(device=device, memory=bool(self.profile_memory),
                                           shapes=bool(self.record_shapes), sync=True,
                                           stacks=bool(self.with_stack))
        self._session.__enter__()

    def _end(self):
        session, self._session = self._session, None
        session.__exit__(None, None, None)
        self._result = session.result
        self._events = None

    def start(self):
        if self.enabled and self.current_action in _RECORDING:
            self._begin()
        return self

    def stop(self):
        if self._session is not None:
            self._end()
            if self.on_trace_ready is not None and (
                    self.schedule is None or self.current_action == ProfilerAction.RECORD_AND_SAVE):
                self.on_trace_ready(self)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def step(self):
        if not self.enabled or self.schedule is None:
            self.step_num += 1
            if self._session is not None:
                self._session.step()
            return
        previous = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)
        if previous in _RECORDING and (previous == ProfilerAction.RECORD_AND_SAVE
                                       or self.current_action not in _RECORDING):
            self._end()
            if previous == ProfilerAction.RECORD_AND_SAVE and self.on_trace_ready is not None:
                self.on_trace_ready(self)
        elif self._session is not None:
            self._session.step()
        if self.current_action in _RECORDING and self._session is None:
            self._begin()

    # -- results ------------------------------------------------------------------
    @property
    def profiler(self):
        return self

    @property
    def result(self):
        if self._result is None:
            raise RuntimeError("no finished recording yet")
        return self._result

    @property
    def memory(self):
        return self.result.memory

    def _use_device(self):
        return "cuda" if ProfilerActivity.CUDA in self.activities and not self.result.on_host else None

    def events(self):
        if self._events is None:
            self._events = self._event_list(
                _build_events(self.result, self._use_device(), self.record_shapes, self.with_stack,
                              self.profile_memory),
                use_device=self._use_device(), profile_memory=self.profile_memory,
                with_flops=self.with_flops, profiler=self)
        return self._events

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0, group_by_overload_name=False):
        return self.events().key_averages(group_by_input_shape, group_by_stack_n, group_by_overload_name)

    def export_chrome_trace(self, path):
        """The recording as PyTorch's Kineto writes it (``.json`` or ``.json.gz``)."""
        trace = _kineto_trace(self)
        trace["traceName"] = path
        if path.endswith(".gz"):
            import gzip
            with gzip.open(path, "wt") as f:
                json.dump(trace, f)
        else:
            with open(path, "w") as f:
                json.dump(trace, f)
        return path

    def export_stacks(self, path, metric="self_cpu_time_total"):
        return self.events().export_stacks(path, metric)

    def export_memory_timeline(self, path, device=None):
        """PyTorch's memory timeline (``.json``, ``.json.gz`` or ``.html``) of the
        device's allocated bytes; jittor does not classify them, so all are "Unknown"."""
        timeline = _memory_timeline(self.result)
        index = self.result.memory.device
        if device is not None and ":" in str(device):
            index = int(str(device).rsplit(":", 1)[1])
        if path.endswith(".html"):
            html = _timeline_html(timeline, index)
            if html is not None:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
        elif path.endswith(".gz"):
            import gzip
            with gzip.open(path, "wt") as f:
                json.dump(timeline, f)
        else:
            with open(path, "w") as f:
                json.dump(timeline, f)

    def summary(self):
        return self.result.summary()


class record_function(_profiling.record_function):
    """``torch.profiler.record_function(name)``: a named range in the trace."""

    def __init__(self, name, args=None):
        super().__init__(name)
