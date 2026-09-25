"""Chrome trace (chrome://tracing, Perfetto) export of a ProfileResult."""

import json
import os

_HOST_PID, _DEVICE_PID = 1, 2


def _us(ns, t0):
    return (ns - t0) / 1e3


def write_chrome_trace(result, path):
    t0 = result.t0
    s = result.strings
    events = [
        {"ph": "M", "pid": _HOST_PID, "name": "process_name", "args": {"name": "host"}},
        {"ph": "M", "pid": _DEVICE_PID, "name": "process_name", "args": {"name": "device"}},
        {"ph": "M", "pid": _HOST_PID, "tid": 1, "name": "thread_name", "args": {"name": "executor"}},
        {"ph": "M", "pid": _HOST_PID, "tid": 2, "name": "thread_name", "args": {"name": "operators"}},
    ]
    for b in result.batches:
        events.append({"ph": "X", "pid": _HOST_PID, "tid": 1, "name": "executor batch %d" % b["batch"],
                       "ts": _us(b["t_begin"], t0), "dur": (b["t_end"] - b["t_begin"]) / 1e3,
                       "args": {"planning_us": (b["t_planned"] - b["t_begin"]) / 1e3 if b["t_planned"] else None,
                                "compile_us": (b["t_compiled"] - b["t_planned"]) / 1e3 if b["t_compiled"] else None,
                                "parent": b["parent"]}})
    for w in result.waits:
        events.append({"ph": "X", "pid": _HOST_PID, "tid": 1, "name": "device wait",
                       "ts": _us(w["t_begin"], t0), "dur": (w["t_end"] - w["t_begin"]) / 1e3})
    for o in result.op_records:
        args = {"seq": o["seq"], "alloc_us": (o["t_alloc"] - o["t_start"]) / 1e3}
        if o["shapes"] > 0:
            args["inputs"] = s[o["shapes"]]
        if o["site"] >= 0:
            args["site"] = s[o["site"]]
        if o["out_bytes"]:
            args["out_bytes"] = o["out_bytes"]
        events.append({"ph": "X", "pid": _HOST_PID, "tid": 2, "name": s[o["name"]],
                       "ts": _us(o["t_start"], t0), "dur": (o["t_end"] - o["t_start"]) / 1e3,
                       "args": args})
    for index, t in enumerate(result.steps):
        events.append({"ph": "i", "s": "g", "pid": _HOST_PID, "tid": 1, "name": "step %d" % index,
                       "ts": _us(t, t0)})
    streams = set()
    for r in result.kernel_records + result.copy_records:
        tid = 1000 + r["stream"]
        streams.add(tid)
        op = result._by_seq.get(r["seq"])
        args = {"correlation": r["correlation"]}
        if op is not None:
            args["operator"] = s[op["name"]]
            args["seq"] = op["seq"]
        if r.get("graph"):
            args["graph"] = r["graph"]
        if "bytes" in r:
            args["bytes"] = r["bytes"]
        events.append({"ph": "X", "pid": _DEVICE_PID, "tid": tid, "name": r["name"],
                       "ts": _us(r["start"], t0), "dur": (r["end"] - r["start"]) / 1e3, "args": args})
        if op is not None:
            events.append({"ph": "s", "id": r["correlation"], "pid": _HOST_PID, "tid": 2,
                           "ts": _us(op["t_start"], t0), "name": "launch", "cat": "launch"})
            events.append({"ph": "f", "bp": "e", "id": r["correlation"], "pid": _DEVICE_PID, "tid": tid,
                           "ts": _us(r["start"], t0), "name": "launch", "cat": "launch"})
    for tid in sorted(streams):
        events.append({"ph": "M", "pid": _DEVICE_PID, "tid": tid, "name": "thread_name",
                       "args": {"name": "stream %d" % (tid - 1000)}})
    memory = result.memory
    if memory is not None:
        timeline = memory.timeline
        step = max(1, len(timeline) // 20000)
        for t, allocated, temp, reserved in timeline[::step] + timeline[-1:]:
            events.append({"ph": "C", "pid": _HOST_PID, "name": "device memory (MiB)", "ts": _us(t, t0),
                           "args": {"allocated": allocated / 2**20, "workspace": temp / 2**20,
                                    "reserved": reserved / 2**20}})
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    with open(path, "w") as out:
        json.dump({"traceEvents": events, "displayTimeUnit": "ms"}, out)
