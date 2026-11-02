"""PyTorch's profiler events, averages and table, over a jittor profile.

The classes and ``_build_table`` follow ``torch/autograd/profiler_util.py``
attribute for attribute; ``_build_events`` turns a
:class:`jittor.profiling.ProfileResult` into them. See ``profiler.py``.
"""

from collections import defaultdict
from enum import Enum as _Enum
import os


class DeviceType(_Enum):
    """The device an event ran on, as ``torch.autograd.DeviceType`` names it."""
    CPU = 0
    CUDA = 1

    def __str__(self):
        return "DeviceType." + self.name


# -- formatting, as torch/autograd/profiler_util.py formats -------------------

def _format_time(time_us):
    if time_us >= 1e6:
        return "%.3fs" % (time_us / 1e6)
    if time_us >= 1e3:
        return "%.3fms" % (time_us / 1e3)
    return "%.3fus" % time_us


def _format_time_share(time_us, total_time_us):
    if total_time_us == 0:
        return "NaN"
    return "%.2f%%" % (time_us * 100.0 / total_time_us)


def _format_memory(nbytes):
    kb = 1024
    mb = 1024 * kb
    gb = 1024 * mb
    if abs(nbytes) >= gb:
        return "%.2f GB" % (nbytes * 1.0 / gb)
    if abs(nbytes) >= mb:
        return "%.2f MB" % (nbytes * 1.0 / mb)
    if abs(nbytes) >= kb:
        return "%.2f KB" % (nbytes * 1.0 / kb)
    return str(nbytes) + " B"


class _Times:
    """The ``*_str`` and ``cuda_*`` spellings PyTorch's events carry."""

    @property
    def cpu_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.cpu_time_total / self.count

    @property
    def device_time(self):
        return 0.0 if self.count == 0 else 1.0 * self.device_time_total / self.count

    cuda_time = device_time

    @property
    def cuda_time_total(self):
        return self.device_time_total

    @property
    def self_cuda_time_total(self):
        return self.self_device_time_total

    @property
    def cuda_memory_usage(self):
        return self.device_memory_usage

    @property
    def self_cuda_memory_usage(self):
        return self.self_device_memory_usage


for _attr in ("cpu_time", "device_time", "cpu_time_total", "device_time_total",
              "self_cpu_time_total", "self_device_time_total"):
    setattr(_Times, _attr + "_str", property(lambda self, _a=_attr: _format_time(getattr(self, _a))))


class Interval:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def elapsed_us(self):
        return self.end - self.start


class FunctionEvent(_Times):
    """One operator, ``record_function`` range, kernel or copy; times in us."""

    def __init__(self, id, name, start_us, end_us, device_type, input_shapes=None, stack=None,
                 use_device=None, is_user_annotation=False):
        self.id = id
        self.name = name
        self.trace_name = name
        self.time_range = Interval(start_us, end_us)
        self.device_type = device_type
        self.device_index = 0
        self.thread = 0
        self.fwd_thread = None
        self.input_shapes = input_shapes if input_shapes is not None else []
        self.concrete_inputs = []
        self.kwinputs = {}
        self.stack = stack if stack is not None else []
        self.scope = 0
        self.use_device = use_device
        self.count = 1
        self.cpu_children = []
        self.cpu_parent = None
        self.kernels = []
        self.is_async = False
        self.is_remote = False
        self.is_legacy = False
        self.is_user_annotation = is_user_annotation
        self.sequence_nr = -1
        self.node_id = -1
        self.overload_name = ""
        self.flops = 0
        self.metadata_json = None
        # Own device time (kernels it launched) and own memory; the totals
        # add the children's.
        self._own_device_us = 0.0
        self._own_cpu_memory = 0
        self._own_device_memory = 0

    @property
    def key(self):
        return self.name

    # A child normally runs inside its parent, and then these are PyTorch's
    # elapsed and elapsed-minus-children. An operator built inside a
    # record_function range runs when the graph executes, often after the
    # range closed; its time is still the range's, so the total is the range's
    # own time plus its children's wherever they ran.
    def _inside(self, child):
        return child.time_range.start >= self.time_range.start and child.time_range.end <= self.time_range.end

    @property
    def self_cpu_time_total(self):
        if self.device_type != DeviceType.CPU:
            return 0.0
        return self.time_range.elapsed_us() - sum(child.cpu_time_total for child in self.cpu_children
                                                  if self._inside(child))

    @property
    def cpu_time_total(self):
        if self.device_type != DeviceType.CPU:
            return 0.0
        return self.self_cpu_time_total + sum(child.cpu_time_total for child in self.cpu_children)

    @property
    def self_device_time_total(self):
        if self.device_type == DeviceType.CPU:
            return self._own_device_us
        return self.time_range.elapsed_us()

    @property
    def device_time_total(self):
        if self.device_type != DeviceType.CPU:
            return self.time_range.elapsed_us()
        return self._own_device_us + sum(child.device_time_total for child in self.cpu_children)

    @property
    def self_cpu_memory_usage(self):
        return self._own_cpu_memory

    @property
    def cpu_memory_usage(self):
        return self._own_cpu_memory + sum(child.cpu_memory_usage for child in self.cpu_children)

    @property
    def self_device_memory_usage(self):
        return self._own_device_memory

    @property
    def device_memory_usage(self):
        return self._own_device_memory + sum(child.device_memory_usage for child in self.cpu_children)

    def append_cpu_child(self, child):
        self.cpu_children.append(child)

    def set_cpu_parent(self, parent):
        self.cpu_parent = parent

    def __repr__(self):
        return ("<FunctionEvent id=%d name=%s device_type=%s cpu_time=%s start_us=%.3f end_us=%.3f "
                "cpu_children=%s cuda_time=%s input_shapes=%s>" % (
                    self.id, self.name, self.device_type, self.cpu_time_str, self.time_range.start,
                    self.time_range.end, [c.id for c in self.cpu_children], self.device_time_str,
                    self.input_shapes))


class FunctionEventAvg(_Times):
    """Events with the same key, summed; :meth:`EventList.key_averages` rows."""

    def __init__(self):
        self.key = None
        self.count = 0
        self.node_id = 0
        self.is_async = False
        self.is_remote = False
        self.use_device = None
        self.cpu_time_total = 0
        self.device_time_total = 0
        self.self_cpu_time_total = 0
        self.self_device_time_total = 0
        self.input_shapes = None
        self.overload_name = None
        self.stack = None
        self.scope = None
        self.cpu_memory_usage = 0
        self.device_memory_usage = 0
        self.self_cpu_memory_usage = 0
        self.self_device_memory_usage = 0
        self.cpu_children = None
        self.cpu_parent = None
        self.device_type = DeviceType.CPU
        self.is_legacy = False
        self.is_user_annotation = False
        self.flops = 0

    def add(self, other):
        if self.key is None:
            self.key = other.key
            self.node_id = other.node_id
            self.is_async = other.is_async
            self.is_remote = other.is_remote
            self.cpu_parent = other.cpu_parent
            self.cpu_children = other.cpu_children
            self.overload_name = other.overload_name
            self.input_shapes = other.input_shapes
            self.stack = other.stack
            self.scope = other.scope
            self.device_type = other.device_type
            self.is_legacy = other.is_legacy
            self.use_device = other.use_device
            self.is_user_annotation = other.is_user_annotation
        if other.key != self.key:
            raise AssertionError("Expected keys to match, but got %s vs %s" % (other.key, self.key))
        self.cpu_time_total += other.cpu_time_total
        self.device_time_total += other.device_time_total
        self.self_cpu_time_total += other.self_cpu_time_total
        self.self_device_time_total += other.self_device_time_total
        self.cpu_memory_usage += other.cpu_memory_usage
        self.device_memory_usage += other.device_memory_usage
        self.self_cpu_memory_usage += other.self_cpu_memory_usage
        self.self_device_memory_usage += other.self_device_memory_usage
        self.count += other.count
        self.flops += other.flops
        return self

    __iadd__ = add

    def __repr__(self):
        device_name = "cuda" if self.use_device else "None"
        return ("<FunctionEventAvg key=%s self_cpu_time=%s cpu_time=%s  self_%s_time=%s %s_time=%s "
                "input_shapes=%s cpu_memory_usage=%d %s_memory_usage=%d>" % (
                    self.key, self.self_cpu_time_total_str, self.cpu_time_str, device_name,
                    self.self_device_time_total_str, device_name, self.device_time_str,
                    self.input_shapes, self.cpu_memory_usage, device_name, self.device_memory_usage))


_SORT_KEYS = ("cpu_time", "cuda_time", "device_time", "cpu_time_total", "cuda_time_total",
              "device_time_total", "self_cpu_time_total", "self_cuda_time_total",
              "self_device_time_total", "cpu_memory_usage", "cuda_memory_usage",
              "device_memory_usage", "self_cpu_memory_usage", "self_cuda_memory_usage",
              "self_device_memory_usage", "count")


def _sort_attr(sort_by):
    return sort_by.replace("cuda", "device").replace("xpu", "device").replace("privateuse1", "device")


class EventList(list):
    """A list of events with PyTorch's table, averages and exports."""

    def __init__(self, *args, use_device=None, profile_memory=False, with_flops=False,
                 profiler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_device = use_device
        self._profile_memory = profile_memory
        self._with_flops = with_flops
        self._profiler = profiler

    @property
    def self_cpu_time_total(self):
        return sum(event.self_cpu_time_total for event in self)

    def table(self, sort_by=None, row_limit=100, max_src_column_width=75, max_name_column_width=55,
              max_shapes_column_width=80, header=None, top_level_events_only=False,
              time_unit=None):
        """PyTorch's ``EventList.table``: the same columns, widths and units."""
        if sort_by is not None and _sort_attr(sort_by) not in [_sort_attr(k) for k in _SORT_KEYS]:
            raise AttributeError("unknown sort key %r" % (sort_by,))
        return _build_table(self, sort_by=sort_by, header=header, row_limit=row_limit,
                            max_src_column_width=max_src_column_width,
                            max_name_column_width=max_name_column_width,
                            max_shapes_column_width=max_shapes_column_width,
                            profile_memory=self._profile_memory,
                            top_level_events_only=top_level_events_only, time_unit=time_unit)

    def export_chrome_trace(self, path):
        if self._profiler is None:
            raise RuntimeError("this EventList has no profile to export")
        return self._profiler.export_chrome_trace(path)

    def supported_export_stacks_metrics(self):
        return ["self_cpu_time_total", "self_cuda_time_total", "self_xpu_time_total",
                "self_privateuse1_time_total"]

    def export_stacks(self, path, metric):
        """One ``outer;...;inner value`` line per event with a stack, for flame graphs."""
        if metric not in self.supported_export_stacks_metrics():
            raise ValueError("metric should be one of: " + str(self.supported_export_stacks_metrics()))
        translate = str.maketrans(" ;\t\n", "____")
        with open(path, "w") as f:
            for event in self:
                if event.stack:
                    value = getattr(event, _sort_attr(metric))
                    if int(value) > 0:
                        frames = ";".join(entry.translate(translate) for entry in reversed(event.stack))
                        f.write("%s %d\n" % (frames, int(value)))

    def key_averages(self, group_by_input_shapes=False, group_by_stack_n=0,
                     group_by_overload_name=False):
        """Events summed by key (and input shapes / the innermost stack entries)."""
        stats = defaultdict(FunctionEventAvg)
        for event in self:
            key = [str(event.key), str(event.node_id), str(event.device_type),
                   str(event.is_legacy), str(event.is_user_annotation)]
            if group_by_overload_name:
                key.append(event.overload_name)
            if group_by_input_shapes:
                key.append(str(event.input_shapes))
            if group_by_stack_n > 0:
                key += event.stack[:group_by_stack_n]
            stats[tuple(key)].add(event)
        averages = type(self)(stats.values(), use_device=self._use_device,
                             profile_memory=self._profile_memory, with_flops=self._with_flops,
                             profiler=self._profiler)
        for event in averages:
            event.stack = event.stack[:group_by_stack_n]
            if not group_by_input_shapes:
                event.input_shapes = ""
            if not group_by_overload_name:
                event.overload_name = ""
        return averages

    def total_average(self):
        total = FunctionEventAvg()
        for event in self:
            total += event
            total.key = None
        total.key = "Total"
        return total


def _build_table(events, sort_by=None, header=None, row_limit=100, max_src_column_width=75,
                 max_name_column_width=55, max_shapes_column_width=80, profile_memory=False,
                 top_level_events_only=False, time_unit=None):
    """The text ``torch.autograd.profiler_util._build_table`` lays out."""
    if len(events) == 0:
        return ""
    has_device_time = any(event.self_device_time_total > 0 for event in events)
    has_device_mem = any(event.self_device_memory_usage > 0 for event in events)
    use_device = events[0].use_device
    if not use_device and has_device_time:
        raise RuntimeError("use_device is None, but there is device performance data.")
    has_input_shapes = any(event.input_shapes is not None and len(event.input_shapes) > 0
                           for event in events)
    if sort_by is not None:
        events = sorted(events, key=lambda event: getattr(event, _sort_attr(sort_by)), reverse=True)

    name_column_width = max(len(event.key) for event in events) + 4
    if max_name_column_width is not None:
        name_column_width = min(name_column_width, max_name_column_width)
    shapes_column_width = max(len(str(event.input_shapes)) for event in events) + 4
    if max_shapes_column_width is not None:
        shapes_column_width = min(shapes_column_width, max_shapes_column_width)
    stacks = [event.stack for event in events if event.stack is not None and len(event.stack) > 0]
    has_stack = len(stacks) > 0
    src_column_width = None
    if has_stack:
        src_column_width = max(max(len(entry) for entry in stack) for stack in stacks) + 4
        if max_src_column_width is not None:
            src_column_width = min(src_column_width, max_src_column_width)

    device_name = use_device.upper() if use_device is not None else "None"
    headers = ["Name", "Self CPU %", "Self CPU", "CPU total %", "CPU total", "CPU time avg"]
    if has_device_time:
        headers += ["Self %s" % device_name, "Self %s %%" % device_name, "%s total" % device_name,
                    "%s time avg" % device_name]
    if profile_memory:
        headers += ["CPU Mem", "Self CPU Mem"]
        if use_device and has_device_mem:
            headers += ["%s Mem" % device_name, "Self %s Mem" % device_name]
    headers.append("# of Calls")

    spacing = 2
    formats, seps, widths = [], [], []

    def add_column(width, align=">"):
        formats.append("{: " + align + str(width) + "}" + " " * spacing)
        seps.append("-" * width + " " * spacing)
        widths.append(width + spacing)

    add_column(name_column_width)
    for _ in headers[1:]:
        add_column(12)
    if has_input_shapes:
        headers.append("Input Shapes")
        add_column(shapes_column_width)
    if has_stack:
        headers.append("Source Location")
        add_column(src_column_width, align="<")
    row_format = "".join(formats)
    header_sep = "".join(seps)
    line_length = sum(widths) - spacing

    def override_time_unit(time_us, default):
        if time_unit == "s":
            return "%.3fs" % (time_us / 1e6)
        if time_unit == "ms":
            return "%.3fms" % (time_us / 1e3)
        if time_unit == "us":
            return "%.3fus" % time_us
        return default

    def trim_path(path):
        if len(path) > src_column_width:
            path = path[len(path) - src_column_width:]
            if len(path) > 3:
                path = "..." + path[3:]
        return path

    sum_self_cpu = sum(event.self_cpu_time_total for event in events)
    sum_self_device = sum(event.self_device_time_total for event in events
                          if event.device_type == DeviceType.CUDA and not event.is_user_annotation)
    result = []

    def append(line):
        result.append(line)
        result.append("\n")

    if header is not None:
        append("=" * line_length)
        append(header)
    if top_level_events_only:
        append("=" * line_length)
        append("This report only display top-level ops statistics")
    append(header_sep)
    append(row_format.format(*headers))
    append(header_sep)

    shown = 0
    for event in events:
        if shown == row_limit:
            break
        if top_level_events_only and event.cpu_parent is not None:
            continue
        shown += 1
        name = event.key
        if max_name_column_width is not None and len(name) >= max_name_column_width - 3:
            name = name[:max_name_column_width - 3] + "..."
        row = [name,
               _format_time_share(event.self_cpu_time_total, sum_self_cpu),
               override_time_unit(event.self_cpu_time_total, event.self_cpu_time_total_str),
               _format_time_share(event.cpu_time_total, sum_self_cpu) if not event.is_async else 0,
               override_time_unit(event.cpu_time_total, event.cpu_time_total_str),
               override_time_unit(event.cpu_time, event.cpu_time_str)]
        if has_device_time:
            row += [override_time_unit(event.self_device_time_total, event.self_device_time_total_str),
                    _format_time_share(event.self_device_time_total, sum_self_device),
                    override_time_unit(event.device_time_total, event.device_time_total_str),
                    override_time_unit(event.device_time, event.device_time_str)]
        if profile_memory:
            row += [_format_memory(event.cpu_memory_usage), _format_memory(event.self_cpu_memory_usage)]
            if use_device and has_device_mem:
                row += [_format_memory(event.device_memory_usage),
                        _format_memory(event.self_device_memory_usage)]
        row.append(event.count)
        if has_input_shapes:
            row.append(str(event.input_shapes)[:shapes_column_width])
        if has_stack:
            row.append(trim_path(event.stack[0]) if len(event.stack) > 0 else "")
        append(row_format.format(*row))
        if has_stack:
            empty = [""] * (len(headers) - 1)
            for entry in event.stack[1:]:
                append(row_format.format(*(empty + [trim_path(entry)])))
            append(row_format.format(*(empty + [""])))
    append(header_sep)
    append("Self CPU time total: %s" % override_time_unit(sum_self_cpu, _format_time(sum_self_cpu)))
    if has_device_time:
        append("Self %s time total: %s" % (device_name, override_time_unit(
            sum_self_device, _format_time(sum_self_device))))
    return "".join(result)


# -- events from a jittor profile ---------------------------------------------

def _input_shapes(text):
    """'float32[64,256], int32[]' -> [[64, 256], []]."""
    shapes = []
    for part in text.split("], "):
        if "[" not in part:
            continue
        dims = part[part.index("[") + 1:].rstrip("]")
        shapes.append([int(d) for d in dims.split(",") if d.strip()])
    return shapes


_JITTOR_ROOTS = []


def _jittor_frame(path):
    """Whether a frame is jittor's own: its package, the Torch layer and the
    backends, wherever the installation put them -- not their test suites."""
    if not _JITTOR_ROOTS:
        import jittor
        package = os.path.realpath(os.path.dirname(jittor.__file__))
        compat = os.path.realpath(os.path.join(package, "compat"))
        roots = {package, compat}
        checkout = os.path.dirname(os.path.dirname(package))
        if os.path.isdir(os.path.join(checkout, "backends")):
            roots.add(os.path.realpath(os.path.join(checkout, "backends")))
        _JITTOR_ROOTS.append([root + os.sep for root in roots])
        _JITTOR_ROOTS.append([os.path.join(compat, "tests") + os.sep])
    real = os.path.realpath(path)
    roots, tests = _JITTOR_ROOTS
    return any(real.startswith(r) for r in roots) and not any(real.startswith(t) for t in tests)


_demangled = {}


def _demangle(name):
    """A kernel's C++ name as PyTorch's profiler shows it ("void ns::kernel<...>(...)")."""
    if not name.startswith("_Z"):
        return name
    if name not in _demangled:
        import ctypes
        import ctypes.util
        text = name
        try:
            lib = ctypes.CDLL(ctypes.util.find_library("stdc++") or "libstdc++.so.6")
        except OSError:
            lib = None
        if lib is not None:
            demangle = lib.__cxa_demangle
            demangle.restype = ctypes.c_void_p
            status = ctypes.c_int()
            out = demangle(name.encode(), None, None, ctypes.byref(status))
            if out and status.value == 0:
                text = ctypes.string_at(out).decode("utf-8", "replace")
                ctypes.CDLL(None).free(ctypes.c_void_p(out))
        _demangled[name] = text
    return _demangled[name]


def _stack_entries(frames):
    """PyTorch's ``"file(line): function"`` entries, innermost first, without
    the frames of jittor itself (PyTorch's own C++ frames are not shown
    either)."""
    return ["%s(%d): %s" % (f["filename"], f["line"], f["name"])
            for f in frames if not _jittor_frame(f["filename"])]


def _build_events(result, use_device, record_shapes, with_stack, profile_memory):
    from jittor.profiling._memory import POOL, TEMP
    strings, t0 = result.strings, result.t0
    us = lambda ns: (ns - t0) / 1e3
    host = []
    for record in sorted(result.op_records, key=lambda r: (r["t_start"], -r["t_end"])):
        shapes = _input_shapes(strings[record["shapes"]]) if record_shapes and record["shapes"] > 0 else []
        stack = []
        if with_stack and result.stacks is not None and 0 < record.get("stack", 0) < len(result.stacks):
            stack = _stack_entries(result.stacks[record["stack"]])
        event = FunctionEvent(record["seq"], strings[record["name"]], us(record["t_start"]),
                              us(record["t_end"]), DeviceType.CPU, shapes, stack, use_device)
        event._built_in = record.get("range", -1)
        host.append(event)
    by_seq = {event.id: event for event in host}
    # Nesting: what ran inside a record_function range is its child, and so
    # is an operator built inside one, wherever it ran.
    open_events = []
    for event in host:
        while open_events and open_events[-1].time_range.end <= event.time_range.start:
            open_events.pop()
        parent = None
        if open_events and event.time_range.end <= open_events[-1].time_range.end:
            parent = open_events[-1]
        if parent is None and event._built_in in by_seq:
            parent = by_seq[event._built_in]
        if parent is not None:
            event.set_cpu_parent(parent)
            parent.append_cpu_child(event)
        open_events.append(event)
    device = []
    next_id = max([e.id for e in host] + [-1]) + 1
    for record in sorted(result.kernel_records + result.copy_records, key=lambda r: r["start"]):
        event = FunctionEvent(next_id, _demangle(record["name"]), us(record["start"]),
                              us(record["end"]), DeviceType.CUDA, use_device=use_device)
        next_id += 1
        owner = by_seq.get(record.get("seq", -1))
        if owner is not None:
            owner._own_device_us += event.time_range.elapsed_us()
            owner.kernels.append(event)
        device.append(event)
    if profile_memory and result.memory is not None:
        for e in result.memory.events:
            owner = by_seq.get(e["op"])
            if owner is None or e["kind"] not in (POOL, TEMP):
                continue
            if e["device"] >= 0:
                owner._own_device_memory += e["bytes"]
            else:
                owner._own_cpu_memory += e["bytes"]
    return sorted(host + device, key=lambda e: (e.time_range.start, -e.time_range.end))
