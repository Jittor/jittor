"""``torch.profiler`` and the recording half of ``torch.autograd.profiler``.

Both used to be inert: ``profile`` recorded nothing, ``key_averages()`` did
not exist (or returned an empty list) and ``export_chrome_trace`` wrote no
file, so a script that profiled itself reported success with nothing in hand.
They now run :func:`jittor.profiling.profile` underneath.

What maps and what does not:

* one row per launched jittor operator (a fused kernel group is one
  operator); ``cpu_time`` is the executor's host time to launch it (Python
  graph construction is not charged to operators -- see ``jt.profile``'s host
  split for it), ``cuda_time``/``device_time`` the CUPTI time of the kernels
  it launched, 0 when CUPTI is unavailable;
* "self" and "total" times are equal: operators do not nest;
* ``record_shapes`` records input shapes; ``profile_memory`` turns on the
  allocator trace (``prof.memory`` is jittor's :class:`MemoryReport`);
* ``schedule``/``on_trace_ready``/``step()`` follow PyTorch's state machine;
* ``with_stack``/``with_flops``/``with_modules`` are accepted and ignored.
"""

import os
import socket
import time

from jittor import profiling as _profiling


class ProfilerActivity:
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"
    HPU = "hpu"
    MTIA = "mtia"
    # vLLM's profiler wrapper indexes its activity table by this at import.
    PrivateUse1 = "privateuse1"


class ProfilerAction:
    NONE = "none"
    WARMUP = "warmup"
    RECORD = "record"
    RECORD_AND_SAVE = "record_and_save"


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


class FunctionEventAvg:
    """One ``key_averages()`` row; times in microseconds like PyTorch."""

    def __init__(self, stat, device_timed):
        self.key = stat.name
        self.count = stat.calls
        self.cpu_time_total = stat.host_ns / 1e3
        self.self_cpu_time_total = self.cpu_time_total
        self.device_time_total = stat.device_ns / 1e3
        self.self_device_time_total = self.device_time_total
        self.cuda_time_total = self.self_cuda_time_total = self.device_time_total
        self.cpu_time = self.cpu_time_total / self.count if self.count else 0.0
        self.device_time = self.cuda_time = self.device_time_total / self.count if self.count else 0.0
        self.input_shapes = stat.shapes
        self.device_type = "cuda" if device_timed else "cpu"
        self.cpu_memory_usage = self.self_cpu_memory_usage = 0
        self.device_memory_usage = self.cuda_memory_usage = 0
        self.is_legacy = False
        self.node_id = -1
        self.stack = []

    def __repr__(self):
        return "<FunctionEventAvg key=%s self_cpu_time=%.3fus device_time=%.3fus count=%d>" % (
            self.key, self.self_cpu_time_total, self.device_time_total, self.count)


class FunctionEvent:
    """One launched operator; times in microseconds."""

    def __init__(self, record, strings, device_ns, t0):
        self.id = record["seq"]
        self.name = self.key = strings[record["name"]]
        self.time_range = ((record["t_start"] - t0) / 1e3, (record["t_end"] - t0) / 1e3)
        self.cpu_time_total = self.self_cpu_time_total = (record["t_end"] - record["t_start"]) / 1e3
        self.device_time_total = self.cuda_time_total = device_ns / 1e3
        self.self_device_time_total = self.self_cuda_time_total = self.device_time_total
        self.input_shapes = strings[record["shapes"]] if record["shapes"] > 0 else ""
        self.count = 1

    def __repr__(self):
        return "<FunctionEvent id=%d name=%s cpu=%.3fus device=%.3fus>" % (
            self.id, self.name, self.cpu_time_total, self.device_time_total)


_SORT_KEYS = ("cpu_time", "cuda_time", "device_time", "cpu_time_total", "cuda_time_total",
              "device_time_total", "self_cpu_time_total", "self_cuda_time_total",
              "self_device_time_total", "count")


class EventList(list):
    def __init__(self, items=(), profiler=None):
        super().__init__(items)
        self._profiler = profiler

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, max_name_column_width=55,
              max_shapes_column_width=80, header=None, top_level_events_only=False, **_):
        rows = list(self)
        if sort_by is not None:
            if sort_by not in _SORT_KEYS:
                raise ValueError("sort_by must be one of %s" % (_SORT_KEYS,))
            rows.sort(key=lambda e: getattr(e, sort_by), reverse=True)
        if row_limit is not None and row_limit >= 0:
            rows = rows[:row_limit]
        total_cpu = sum(e.self_cpu_time_total for e in self) or 1.0
        total_dev = sum(e.self_device_time_total for e in self) or 1.0
        shapes = any(getattr(e, "input_shapes", "") for e in rows)
        w = max_name_column_width
        cols = ("%-*s %10s %12s %12s %12s %10s %12s %10s" % (
            w, "Name", "Self CPU %", "Self CPU", "CPU total", "CPU avg", "Self CUDA %",
            "Self CUDA", "# Calls"))
        if shapes:
            cols += "  Input Shapes"
        line = "-" * len(cols)
        out = [header] if header else []
        out += [line, cols, line]
        for e in rows:
            name = e.key if len(e.key) <= w else e.key[:w - 3] + "..."
            text = "%-*s %9.2f%% %12s %12s %12s %9.2f%% %12s %10d" % (
                w, name, 100 * e.self_cpu_time_total / total_cpu, _t(e.self_cpu_time_total),
                _t(e.cpu_time_total), _t(e.cpu_time_total / max(e.count, 1)),
                100 * e.self_device_time_total / total_dev, _t(e.self_device_time_total), e.count)
            if shapes:
                text += "  " + (e.input_shapes or "")[:max_shapes_column_width]
            out.append(text)
        out.append(line)
        out.append("Self CPU time total: %s" % _t(total_cpu))
        out.append("Self CUDA time total: %s" % _t(sum(e.self_device_time_total for e in self)))
        return "\n".join(out)

    def export_chrome_trace(self, path):
        if self._profiler is None:
            raise RuntimeError("this EventList has no profile to export")
        return self._profiler.export_chrome_trace(path)

    def total_average(self):
        class _Total:
            pass
        total = _Total()
        total.key = "Total"
        for attr in ("cpu_time_total", "self_cpu_time_total", "device_time_total",
                     "self_device_time_total", "cuda_time_total", "self_cuda_time_total"):
            setattr(total, attr, sum(getattr(e, attr) for e in self))
        total.count = sum(e.count for e in self)
        return total

    def key_averages(self, group_by_input_shapes=False):
        return self


def _t(us):
    if us >= 1e6:
        return "%.3fs" % (us / 1e6)
    if us >= 1e3:
        return "%.3fms" % (us / 1e3)
    return "%.3fus" % us


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
        self.enabled = enabled
        self.step_num = 0
        self.current_action = ProfilerAction.RECORD if schedule is None else schedule(0)
        self._session = None
        self._result = None

    # -- recording ----------------------------------------------------------------
    def _begin(self):
        device = "auto" if ProfilerActivity.CUDA in self.activities else False
        self._session = _profiling.profile(device=device, memory=bool(self.profile_memory),
                                           shapes=bool(self.record_shapes), sync=True)
        self._session.__enter__()

    def _end(self):
        session, self._session = self._session, None
        session.__exit__(None, None, None)
        self._result = session.result

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

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0, group_by_overload_name=False):
        result = self.result
        timed = result.device is not None
        stats = [s for s in result.op_stats(group_by_shapes=group_by_input_shape) if s.calls]
        return self._event_list([FunctionEventAvg(s, timed) for s in stats], profiler=self)

    def events(self):
        result = self.result
        device = {}
        for r in result.kernel_records + result.copy_records:
            device[r["seq"]] = device.get(r["seq"], 0) + r["end"] - r["start"]
        return self._event_list([FunctionEvent(o, result.strings, device.get(o["seq"], 0), result.t0)
                                 for o in result.op_records], profiler=self)

    def export_chrome_trace(self, path):
        return self.result.export_chrome_trace(path)

    def summary(self):
        return self.result.summary()


class record_function(_profiling.record_function):
    """``torch.profiler.record_function(name)``: a named range in the trace."""

    def __init__(self, name, args=None):
        super().__init__(name)
