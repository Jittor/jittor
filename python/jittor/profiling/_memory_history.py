"""Device memory history and snapshots in PyTorch's format.

``record_memory_history`` / ``memory_snapshot`` / ``dump_memory_snapshot`` are
``torch.cuda.memory._record_memory_history`` / ``_snapshot`` /
``_dump_snapshot`` for Jittor's device pools: the dict and the pickle have the
keys, actions and nesting PyTorch's have, so the pickle opens in
https://pytorch.org/memory_viz and reads with ``torch.cuda._memory_viz``.

The records come from the allocator (``src/runtime/profiler/memory_history.h``).
What differs from PyTorch is what the allocator is:

* segments are Jittor's SFRL pool segments -- small ones are 1 MiB, not 2 MiB --
  and blocks are aligned to 512 bytes as there;
* a free never waits behind a stream, so ``free_requested`` and
  ``free_completed`` come in pairs with the same timestamp;
* ``stacks="all"`` records the same Python stacks as ``"python"``; there are
  no C++ frames;
* the stack of a tensor's block is the stack that built its operator (Jittor
  runs lazily, so the stack at allocation time is wherever execution was
  forced); a workspace is charged to the operator that needed it.
"""

import os
import pickle
import sys
import time

import jittor_core as _core

__all__ = ["record_memory_history", "memory_snapshot", "dump_memory_snapshot"]

_ON, _TRACE, _ALLOC_STACKS, _FREE_STACKS, _CLEAR_TRACE = 1, 2, 4, 8, 16
_ACTIONS = ("alloc", "free_requested", "free_completed", "segment_alloc",
            "segment_free", "oom", "snapshot")
# What the current recording asked for that the native side does not keep.
_settings = {"trace_stacks": False}
_annotations = []


def record_memory_history(enabled="all", context="all", stacks="all", max_entries=sys.maxsize,
                          device=None, clear_history=False):
    """Start or stop recording device memory history.

    Args:
        enabled: ``None`` stops (and forgets what was recorded), ``"state"``
            keeps what is allocated now and where, ``"all"`` also keeps the
            trace of every allocation and free.
        context: ``None`` records no stacks, ``"state"`` the stacks of live
            allocations, ``"alloc"`` also those of the trace's allocations,
            ``"all"`` also those of frees.
        stacks: ``"python"`` or ``"all"``; both record Python stacks.
        max_entries: the newest entries kept per device.
        device: accepted for PyTorch's signature; all devices are recorded.
        clear_history: drop the trace recorded so far.
    """
    if enabled not in (None, "state", "all"):
        raise ValueError("enabled must be None, 'state' or 'all', not %r" % (enabled,))
    if context not in (None, "state", "alloc", "all"):
        raise ValueError("context must be None, 'state', 'alloc' or 'all', not %r" % (context,))
    if stacks not in ("python", "all"):
        raise ValueError("stacks must be 'python' or 'all', not %r" % (stacks,))
    mode = 0
    if enabled is not None:
        mode = _ON | (_TRACE if enabled == "all" else 0)
        if context is not None:
            mode |= _ALLOC_STACKS
        if context == "all":
            mode |= _FREE_STACKS
        if clear_history:
            mode |= _CLEAR_TRACE
    else:
        del _annotations[:]
    _settings["trace_stacks"] = context in ("alloc", "all")
    if clear_history:
        del _annotations[:]
    entries = int(max_entries)
    _core.memory_history_start(mode, entries if 0 < entries < (1 << 62) else 0)


def record_memory_history_legacy(enabled, record_context=True, trace_alloc_max_entries=1,
                                 trace_alloc_record_context=False, device=None,
                                 record_context_cpp=False, clear_history=False):
    """PyTorch's older boolean form of :func:`record_memory_history`."""
    if not enabled:
        return record_memory_history(None)
    trace = trace_alloc_max_entries > 1
    if trace_alloc_record_context:
        context = "all"
    elif record_context:
        context = "state"
    else:
        context = None
    return record_memory_history("all" if trace else "state", context,
                                 max_entries=trace_alloc_max_entries if trace else 0,
                                 clear_history=clear_history)


def note_annotation(stage, name):
    """A ``record_function`` range edge, kept while the history is on."""
    if _core.memory_history_mode_now():
        import jittor as jt
        device = int(jt.core.current_device()) if jt.flags.use_cuda else 0
        _annotations.append({"stage": stage, "name": name, "device": device,
                             "time_us": time.time_ns() // 1000})


def _layout():
    fields = {}
    for part in _core.memory_history_layout().split(";"):
        kind, _, names = part.partition(":")
        fields[kind] = names.split(",")
    return fields


def _stack_table():
    flat = _core.memory_history_frames()
    frames = [{"name": flat[i + 2], "filename": flat[i], "line": int(flat[i + 1])}
              for i in range(0, len(flat), 3)]
    raw = _core.memory_history_stacks()
    stacks = [[]]
    i = 0
    while i < len(raw):
        n = raw[i]
        stacks.append([frames[f] for f in raw[i + 1:i + 1 + n]])
        i += 1 + n
    return stacks


def _allocator_settings():
    return {
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""),
        "max_split_size": -1,
        "garbage_collection_threshold": 0.0,
        "expandable_segments": False,
        "pinned_num_register_threads": 1,
        "release_lock_on_cudamalloc": False,
        "pinned_use_cuda_host_register": False,
        "graph_capture_record_stream_reuse": False,
        "roundup_power2_divisions": {str(1 << i): 0 for i in range(16)},
    }


def memory_snapshot():
    """The device pools now, and the recorded trace, as PyTorch's
    ``torch.cuda.memory._snapshot()`` dict."""
    layout = _layout()
    stacks = _stack_table()
    frames = lambda sid: list(stacks[sid]) if 0 < sid < len(stacks) else []
    seg_fields, blk_fields, tr_fields = layout["segment"], layout["block"], layout["trace"]
    ns, nb, nt = len(seg_fields), len(blk_fields), len(tr_fields)
    # Read the trace before the segments: reading the segments appends the
    # "snapshot" entry, which belongs to the next snapshot's trace.
    count = max([0] + [d + 1 for d in _core.memory_history_devices()])
    raw_traces = [_core.memory_history_trace(d) for d in range(count)]
    segments = []
    for device in range(count):
        flat = _core.memory_history_segments(device)
        i = 0
        while i < len(flat):
            seg = dict(zip(seg_fields, flat[i:i + ns]))
            i += ns
            blocks = []
            for _ in range(seg["blocks"]):
                blk = dict(zip(blk_fields, flat[i:i + nb]))
                i += nb
                active = blk["state"] == 1
                blocks.append({"address": blk["address"], "size": blk["size"],
                               "requested_size": blk["requested_size"] if active else 0,
                               "state": "active_allocated" if active else "inactive",
                               "frames": frames(blk["stack"]) if active else []})
            segments.append({
                "device": device, "address": seg["address"], "total_size": seg["total_size"],
                "allocated_size": seg["allocated_size"], "active_size": seg["allocated_size"],
                "requested_size": seg["requested_size"], "stream": seg["stream"],
                "segment_type": "small" if seg["is_small"] else "large",
                "segment_pool_id": (0, 0), "is_expandable": False, "frames": [],
                "blocks": blocks})
    trace_stacks = _settings["trace_stacks"]
    device_traces = []
    for flat in raw_traces:
        trace = []
        for i in range(0, len(flat), nt):
            e = dict(zip(tr_fields, flat[i:i + nt]))
            entry = {"action": _ACTIONS[e["action"]], "addr": e["addr"], "size": e["size"],
                     "stream": e["stream"], "time_us": e["time_us"], "compile_context": "N/A",
                     "user_metadata": "", "frames": frames(e["stack"]) if trace_stacks else []}
            if entry["action"] == "oom":
                entry["device_free"] = e["device_free"]
            trace.append(entry)
        device_traces.append(trace)
    return {"segments": segments, "device_traces": device_traces,
            "allocator_settings": _allocator_settings(),
            "external_annotations": list(_annotations)}


def dump_memory_snapshot(filename="dump_snapshot.pickle"):
    """Write :func:`memory_snapshot` as the pickle https://pytorch.org/memory_viz opens."""
    with open(filename, "wb") as f:
        pickle.dump(memory_snapshot(), f)
