"""The files PyTorch's profiler exports, written from a jittor profile:
the Kineto Chrome trace and the memory timeline. See ``profiler.py``."""

import os
import time

from .profiler_events import _demangle, _input_shapes


def _kineto_trace(prof):
    """The trace PyTorch's Kineto writes: cpu_op/user_annotation events on the
    process, kernel/gpu_memcpy/gpu_memset events on "GPU n" per stream, ac2g
    flows from each operator to its kernels, [memory] instants, and the same
    top-level keys, so TensorBoard's profiler, HTA and Perfetto read it as a
    PyTorch trace."""
    from jittor.profiling._memory import RESERVE, EXISTING
    result = prof.result
    pid = os.getpid()
    tid = pid
    base_ns = time.time_ns() - time.perf_counter_ns()
    us = lambda ns: ns / 1e3
    strings = result.strings
    events = []
    index = 0

    def add(event):
        nonlocal index
        if event.get("ph") in ("X", "i") and "args" in event:
            event["args"]["Ev Idx"] = index
            index += 1
        events.append(event)

    by_seq = {}
    for record in result.op_records:
        is_range = record["kind"] == 2
        args = {"External id": record["seq"], "Record function id": 0}
        if prof.record_shapes and record["shapes"] > 0:
            shapes = _input_shapes(strings[record["shapes"]])
            args["Input Dims"] = shapes
            args["Input type"] = [part.split("[", 1)[0] for part in strings[record["shapes"]].split(", ")]
        event = {"ph": "X", "cat": "user_annotation" if is_range else "cpu_op",
                 "name": strings[record["name"]], "pid": pid, "tid": tid,
                 "ts": us(record["t_start"]), "dur": us(record["t_end"] - record["t_start"]),
                 "args": args}
        by_seq[record["seq"]] = event
        add(event)
    devices = set()
    for record in sorted(result.kernel_records + result.copy_records, key=lambda r: r["start"]):
        is_kernel = "bytes" not in record
        name = _demangle(record["name"]) if is_kernel else record["name"]
        cat = "kernel"
        if not is_kernel:
            cat = "gpu_memset" if name.startswith("memset") else "gpu_memcpy"
        args = {"External id": record["seq"], "device": record["device"], "context": 1,
                "stream": record["stream"], "correlation": record["correlation"]}
        if is_kernel:
            args.update({"registers per thread": record["registers"],
                         "shared memory": record["shared_memory"],
                         "grid": list(record["grid"]), "block": list(record["block"])})
        else:
            args["bytes"] = record["bytes"]
        devices.add((record["device"], record["stream"]))
        add({"ph": "X", "cat": cat, "name": name, "pid": record["device"], "tid": record["stream"],
             "ts": us(record["start"]), "dur": us(record["end"] - record["start"]), "args": args})
        owner = by_seq.get(record["seq"])
        if owner is not None:
            events.append({"ph": "s", "id": record["correlation"], "pid": pid, "tid": tid,
                           "ts": owner["ts"], "cat": "ac2g", "name": "ac2g"})
            events.append({"ph": "f", "id": record["correlation"], "pid": record["device"],
                           "tid": record["stream"], "ts": us(record["start"]), "cat": "ac2g",
                           "name": "ac2g", "bp": "e"})
    memory = result.memory
    if prof.profile_memory and memory is not None:
        allocated, reserved = memory.start[0], memory.start[1]
        for e in sorted((e for e in memory.events if e["kind"] != EXISTING and e["device"] == memory.device),
                        key=lambda e: e["t"]):
            if e["kind"] == RESERVE:
                reserved += e["bytes"]
                continue
            allocated += e["bytes"]
            add({"ph": "i", "cat": "cpu_instant_event", "s": "t", "name": "[memory]", "pid": pid,
                 "tid": tid, "ts": us(e["t"]),
                 "args": {"Total Reserved": reserved, "Total Allocated": allocated,
                          "Bytes": e["bytes"], "Device Id": e["device"],
                          "Device Type": 1 if e["device"] >= 0 else 0, "Addr": e["allocation"],
                          "finished": False}})
    t0 = us(result.t0)
    meta = [{"name": "process_name", "ph": "M", "ts": t0, "pid": pid, "tid": 0, "args": {"name": "python"}},
            {"name": "process_labels", "ph": "M", "ts": t0, "pid": pid, "tid": 0, "args": {"labels": "CPU"}},
            {"name": "process_sort_index", "ph": "M", "ts": t0, "pid": pid, "tid": 0, "args": {"sort_index": pid}},
            {"name": "thread_name", "ph": "M", "ts": t0, "pid": pid, "tid": tid,
             "args": {"name": "thread %d (python)" % tid}},
            {"name": "thread_sort_index", "ph": "M", "ts": t0, "pid": pid, "tid": tid, "args": {"sort_index": tid}}]
    for device in sorted({d for d, _ in devices}):
        meta += [{"name": "process_name", "ph": "M", "ts": t0, "pid": device, "tid": 0, "args": {"name": "python"}},
                 {"name": "process_labels", "ph": "M", "ts": t0, "pid": device, "tid": 0,
                  "args": {"labels": "GPU %d" % device}},
                 {"name": "process_sort_index", "ph": "M", "ts": t0, "pid": device, "tid": 0,
                  "args": {"sort_index": 5000000 + device}}]
    for device, stream in sorted(devices):
        meta += [{"name": "thread_name", "ph": "M", "ts": t0, "pid": device, "tid": stream,
                  "args": {"name": "stream %d " % stream}},
                 {"name": "thread_sort_index", "ph": "M", "ts": t0, "pid": device, "tid": stream,
                  "args": {"sort_index": stream}}]
    span = {"ph": "X", "cat": "Trace", "ts": t0, "dur": us(result.t1 - result.t0), "pid": "Spans",
            "tid": "Jittor Profiler", "name": "Jittor Profiler (0)", "args": {"Op count": 0}}
    trace = {"schemaVersion": 1, "deviceProperties": _device_properties(sorted({d for d, _ in devices})),
             "with_stack": int(bool(prof.with_stack)), "profile_memory": int(bool(prof.profile_memory)),
             "record_shapes": int(bool(prof.record_shapes)), "displayTimeUnit": "ms",
             "baseTimeNanoseconds": base_ns,
             "traceEvents": events + meta + [span, {"name": "Iteration Start: Jittor Profiler",
                                                  "ph": "i", "s": "g", "pid": "Traces",
                                                  "tid": "Trace Jittor Profiler", "ts": t0}]}
    return trace


def _device_properties(devices):
    try:
        import torch
    except ImportError:
        return []
    out = []
    for device in devices:
        props = torch.cuda.get_device_properties(device)
        out.append({"id": device, "name": props.name, "totalGlobalMem": props.total_memory,
                    "computeMajor": props.major, "computeMinor": props.minor,
                    "numSms": props.multi_processor_count})
    return out


# PyTorch's memory-timeline categories, in its column order; jittor puts every
# byte in the last one (None, shown as "Unknown").
_TIMELINE_CATEGORIES = (("PARAMETER", "darkgreen"), ("OPTIMIZER_STATE", "goldenrod"),
                        ("INPUT", "black"), ("TEMPORARY", "mediumpurple"),
                        ("ACTIVATION", "red"), ("GRADIENT", "mediumblue"),
                        ("AUTOGRAD_DETAIL", "royalblue"), (None, "grey"))


def _memory_timeline(result):
    """[times_us, sizes] as PyTorch's ``export_memory_timeline`` writes them:
    each row a leading 0 and the bytes of the eight categories."""
    memory = result.memory
    if memory is None:
        raise RuntimeError("export_memory_timeline needs profile_memory=True")
    # The trace clock is CLOCK_MONOTONIC (perf_counter); PyTorch's is epoch.
    offset_us = (time.time_ns() - time.perf_counter_ns()) / 1e3
    times, sizes = [], []
    for t, allocated, temp, _reserved in memory.timeline:
        times.append(int(offset_us + t / 1e3))
        sizes.append([0] * len(_TIMELINE_CATEGORIES) + [allocated + temp])
    return [times, sizes]


def _timeline_html(timeline, device_index, figsize=(20, 12), title=None):
    """PyTorch's ``export_memory_timeline_html``: the stacked plot as an embedded PNG."""
    import importlib.util
    if importlib.util.find_spec("matplotlib") is None:
        print("export_memory_timeline_html failed because matplotlib was not found.")
        return None
    from base64 import b64encode
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import jittor as jt

    times, sizes = np.array(timeline[0], dtype=np.float64), np.array(timeline[1], dtype=np.float64)
    times -= times.min()
    stacked = np.cumsum(sizes, axis=1) / 1024 ** 3
    fig = plt.figure(figsize=figsize, dpi=80)
    axes = fig.gca()
    for i, (_name, color) in enumerate(_TIMELINE_CATEGORIES):
        axes.fill_between(times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7)
    fig.legend(["Unknown" if name is None else name for name, _ in _TIMELINE_CATEGORIES])
    axes.set_xlabel("Time (ms)")
    axes.set_ylabel("Memory (GB)")
    peak = jt.core.device_memory_peak(device_index) / 1024 ** 3
    reserved = jt.core.device_memory_reserved_peak(device_index) / 1024 ** 3
    axes.set_title("\n\n".join(([title] if title else []) + [
        "Max memory allocated: %.2f GiB \nMax memory reserved: %.2f GiB" % (peak, reserved)]))
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    encoded = b64encode(buffer.getvalue()).decode("utf-8")
    return """<html>
<head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
<body>
  <img src='data:image/png;base64,%s'>
</body>
</html>""" % encoded
