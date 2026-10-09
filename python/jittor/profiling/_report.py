"""The result of one ``jt.profile`` region: host split, device split, tables."""

from collections import defaultdict

OP, GRAPH_LAUNCH, RANGE = 0, 1, 2


def _ms(ns):
    return "%.3f ms" % (ns / 1e6)


def _us(ns):
    return "%.1f" % (ns / 1e3)


def _union(intervals):
    total, end = 0, None
    for a, b in sorted(intervals):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def _short(text, width):
    return text if len(text) <= width else text[:width - 3] + "..."


def _short_left(text, width):
    """Keep the end: for a call site that is the file name and line."""
    return text if len(text) <= width else "..." + text[-(width - 3):]


def _site(path):
    import os
    try:
        rel = os.path.relpath(path)
    except ValueError:
        return path
    return rel if not rel.startswith("..") else path


class OpStat:
    """One row of the operator table (all times in ns)."""

    __slots__ = ("name", "shapes", "calls", "host_ns", "alloc_ns", "wait_ns", "device_ns",
                 "kernels", "sites", "out_bytes", "kind")

    def __init__(self, name, shapes, kind):
        self.name, self.shapes, self.kind = name, shapes, kind
        self.calls = self.host_ns = self.alloc_ns = self.wait_ns = self.device_ns = 0
        self.kernels = 0
        self.out_bytes = 0
        self.sites = defaultdict(int)

    @property
    def top_site(self):
        return max(self.sites.items(), key=lambda kv: kv[1])[0] if self.sites else ""


class ProfileResult:
    """Everything one profiled region measured.

    ``host``    -- the wall time split into Python/graph construction, executor
                   planning, compilation, operator launch, device waits and
                   device-graph launches (ns; these parts sum to ``wall_ns``);
    ``device``  -- kernel/copy time from CUPTI (None when CUPTI is missing);
    ``ops``     -- one record per launched operator;
    ``memory``  -- a :class:`MemoryReport` when memory tracing was on;
    ``python``  -- sampled host time per Python call site when requested;
    ``replay``  -- device-graph launches and graph-replay counters.
    """

    def __init__(self, *, t0, t1, strings, ops, batches, waits, kernels, copies,
                 correlation, device_note, memory, python, replay, steps, truncated,
                 on_host=False):
        self.t0, self.t1 = t0, t1
        self.wall_ns = t1 - t0
        self.strings = strings
        self.op_records = ops
        self.batches = batches
        self.waits = waits
        self.kernel_records = kernels
        self.copy_records = copies
        self.device_note = device_note
        self.memory = memory
        self.python = python
        self.replay = replay
        self.steps = steps
        self.truncated = truncated
        self.on_host = on_host
        self._by_seq = {op["seq"]: op for op in ops}
        self._op_wait = self._waits_inside_ops()
        for rec in kernels + copies:
            rec["seq"] = correlation.get(rec["correlation"], -1)
        self.host = self._host_split()
        self.device = self._device_split() if device_note is None else None

    def _waits_inside_ops(self):
        """Device waits that happened while an operator was being launched:
        a blocking copy or allocation inside the op, charged to it."""
        import bisect
        ops = sorted((o["t_start"], o["t_end"], o["seq"]) for o in self.op_records if o["kind"] == OP)
        starts = [o[0] for o in ops]
        out = defaultdict(int)
        for w in self.waits:
            i = bisect.bisect_right(starts, w["t_begin"]) - 1
            if i >= 0 and ops[i][1] >= w["t_end"]:
                out[ops[i][2]] += w["t_end"] - w["t_begin"]
        return out

    # -- host ---------------------------------------------------------------
    def _host_split(self):
        batches = self.batches
        by_id = {b["batch"]: b for b in batches}
        child_ns = defaultdict(int)
        for b in batches:
            if b["parent"] in by_id:
                child_ns[b["parent"]] += b["t_end"] - b["t_begin"]
        waits_in = defaultdict(int)
        outside_wait = 0
        for w in self.waits:
            if w["batch"] in by_id:
                waits_in[w["batch"]] += w["t_end"] - w["t_begin"]
            else:
                outside_wait += w["t_end"] - w["t_begin"]
        planning = compile_ = launch = wait = 0
        top = 0
        for b in batches:
            planning += max(b["t_planned"] - b["t_begin"] - child_ns[b["batch"]], 0) if b["t_planned"] else 0
            if b["t_compiled"]:
                compile_ += b["t_compiled"] - b["t_planned"]
                run = b["t_end"] - b["t_compiled"]
                wait += waits_in[b["batch"]]
                launch += max(run - waits_in[b["batch"]], 0)
            if b["parent"] not in by_id:
                top += b["t_end"] - b["t_begin"]
        graph = sum(o["t_end"] - o["t_start"] for o in self.op_records
                    if o["kind"] == GRAPH_LAUNCH and o["batch"] not in by_id)
        launched = [o for o in self.op_records if o["kind"] == OP]
        return {
            "wall": self.wall_ns,
            "python_and_graph_build": max(self.wall_ns - top - graph - outside_wait, 0),
            "executor_planning": planning,
            "executor_compile": compile_,
            "operator_launch": launch,
            "device_wait": wait + outside_wait,
            "graph_launch": graph,
            "batches": len(batches),
            "operators": len(launched),
            "operator_alloc": sum(o["t_alloc"] - o["t_start"] for o in launched),
        }

    # -- device ---------------------------------------------------------------
    def _device_split(self):
        records = self.kernel_records + self.copy_records
        busy = _union([(r["start"], r["end"]) for r in records])
        kernel_ns = sum(r["end"] - r["start"] for r in self.kernel_records)
        copy_ns = sum(r["end"] - r["start"] for r in self.copy_records)
        attributed = sum(r["end"] - r["start"] for r in records if r["seq"] in self._by_seq)
        return {"busy": busy, "kernel": kernel_ns, "copy": copy_ns,
                "kernels": len(self.kernel_records), "copies": len(self.copy_records),
                "attributed": attributed,
                "utilization": busy / float(self.wall_ns) if self.wall_ns else 0.0}

    @property
    def bound(self):
        """"device-bound", "host-bound" or "mixed", with the evidence."""
        wall = float(self.wall_ns) or 1.0
        wait = self.host["device_wait"] / wall
        if self.on_host:
            return "host (CPU) execution", "the region runs on the CPU; every part of it is host time"
        if self.device is not None:
            util = self.device["utilization"]
            if util >= 0.8:
                return "device-bound", "device busy %.0f%% of wall time" % (100 * util)
            if util < 0.5:
                h = self.host
                return "host-bound", (
                    "device idle %.0f%% of wall time; host: %.0f%% Python + graph construction, "
                    "%.0f%% executor planning/launch, %.0f%% waiting on the device" % (
                        100 * (1 - util), 100 * h["python_and_graph_build"] / wall,
                        100 * (h["executor_planning"] + h["executor_compile"] + h["operator_launch"]
                               + h["graph_launch"]) / wall, 100 * wait))
            return "mixed", "device busy %.0f%%, host waiting on device %.0f%%" % (100 * util, 100 * wait)
        if wait >= 0.5:
            return "device-bound (estimated)", "host waited on the device %.0f%% of wall time; no device timing" % (100 * wait)
        if wait < 0.1:
            return "host-bound (estimated)", "host waited on the device only %.0f%% of wall time; no device timing" % (100 * wait)
        return "mixed (estimated)", "host waited on the device %.0f%%; no device timing" % (100 * wait)

    # -- tables ---------------------------------------------------------------
    def op_stats(self, group_by_shapes=False):
        stats = {}
        device = defaultdict(int)
        kernels = defaultdict(int)
        for r in self.kernel_records + self.copy_records:
            device[r["seq"]] += r["end"] - r["start"]
            kernels[r["seq"]] += 1
        for o in self.op_records:
            name = self.strings[o["name"]]
            shapes = self.strings[o["shapes"]] if o["shapes"] > 0 else ""
            key = (name, shapes if group_by_shapes else "")
            st = stats.get(key)
            if st is None:
                st = stats[key] = OpStat(name, shapes, o["kind"])
            st.calls += 1
            wait = self._op_wait.get(o["seq"], 0)
            st.wait_ns += wait
            st.host_ns += o["t_end"] - o["t_start"] - wait
            st.alloc_ns += o["t_alloc"] - o["t_start"]
            st.device_ns += device.get(o["seq"], 0)
            st.kernels += kernels.get(o["seq"], 0)
            st.out_bytes += o["out_bytes"]
            if o["site"] >= 0:
                st.sites[_site(self.strings[o["site"]])] += 1
        if device.get(-1):
            st = stats[("(device work outside any operator)", "")] = OpStat(
                "(device work outside any operator)", "", -1)
            st.device_ns = device[-1]
            st.kernels = kernels[-1]
        return list(stats.values())

    def table(self, sort_by="device_time", row_limit=20, group_by_shapes=False, width=48):
        """The operator table, like ``torch.profiler``'s ``key_averages().table()``."""
        rows = self.op_stats(group_by_shapes)
        keys = {"device_time": lambda s: s.device_ns, "host_time": lambda s: s.host_ns,
                "calls": lambda s: s.calls, "name": lambda s: s.name}
        if sort_by not in keys:
            raise ValueError("sort_by must be one of %s" % sorted(keys))
        rows.sort(key=keys[sort_by], reverse=sort_by != "name")
        total_dev = float(sum(r.device_ns for r in rows)) or 1.0
        total_host = float(sum(r.host_ns for r in rows)) or 1.0
        header = "%-*s %6s %10s %6s %9s %10s %6s %8s %7s  %s" % (
            width, "operator", "calls", "host us", "host%", "wait us", "device us", "dev%",
            "dev avg", "kernels", "input shapes" if group_by_shapes else "call site")
        lines = [header, "-" * len(header)]
        for r in rows[:row_limit]:
            extra = _short(r.shapes, 70) if group_by_shapes else _short_left(r.top_site, 60)
            lines.append("%-*s %6d %10s %5.1f%% %9s %10s %5.1f%% %8s %7d  %s" % (
                width, _short(r.name, width), r.calls, _us(r.host_ns), 100 * r.host_ns / total_host,
                _us(r.wait_ns) if r.wait_ns else "",
                _us(r.device_ns) if self.device is not None else "n/a",
                100 * r.device_ns / total_dev,
                _us(r.device_ns / r.calls) if r.calls and self.device is not None else "",
                r.kernels, extra))
        if len(rows) > row_limit:
            lines.append("... %d more rows" % (len(rows) - row_limit))
        return "\n".join(lines)

    def kernel_stats(self):
        stats = defaultdict(lambda: [0, 0, set()])
        for r in self.kernel_records + self.copy_records:
            st = stats[r["name"]]
            st[0] += 1
            st[1] += r["end"] - r["start"]
            op = self._by_seq.get(r["seq"])
            st[2].add(self.strings[op["name"]] if op else "(outside any operator)")
        return sorted(((n, c, t, sorted(o)) for n, (c, t, o) in stats.items()), key=lambda r: -r[2])

    def kernel_table(self, row_limit=15, width=60):
        if self.device is None:
            return "device kernels: unavailable (%s)" % self.device_note
        lines = ["%-*s %6s %11s  %s" % (width, "kernel", "calls", "device us", "issued by")]
        for name, calls, total, ops in self.kernel_stats()[:row_limit]:
            lines.append("%-*s %6d %11s  %s" % (width, _short(name, width), calls, _us(total),
                                                 _short(", ".join(ops), 60)))
        return "\n".join(lines)

    def site_stats(self):
        """Per Python call site: host launch time and device time of the
        operators it built (and sampled host time, if sampling was on)."""
        sites = defaultdict(lambda: {"ops": 0, "launch_ns": 0, "device_ns": 0, "sampled_ns": 0})
        device = defaultdict(int)
        for r in self.kernel_records + self.copy_records:
            device[r["seq"]] += r["end"] - r["start"]
        for o in self.op_records:
            site = _site(self.strings[o["site"]]) if o["site"] >= 0 else "(no Python site)"
            s = sites[site]
            s["ops"] += 1
            s["launch_ns"] += o["t_end"] - o["t_start"]
            s["device_ns"] += device.get(o["seq"], 0)
        if self.python is not None:
            per = self.python.interval * 1e9
            for site, count in self.python.sites.items():
                sites[_site(site)]["sampled_ns"] += int(count * per)
        return dict(sites)

    def site_table(self, sort_by="sampled", row_limit=15):
        rows = list(self.site_stats().items())
        key = {"sampled": "sampled_ns", "device": "device_ns", "launch": "launch_ns"}[sort_by]
        if sort_by == "sampled" and self.python is None:
            key = "device_ns" if self.device is not None else "launch_ns"
        rows.sort(key=lambda kv: -kv[1][key])
        lines = ["%12s %12s %12s %6s  %s" % ("sampled us", "launch us", "device us", "ops", "Python call site")]
        for site, s in rows[:row_limit]:
            lines.append("%12s %12s %12s %6d  %s" % (
                _us(s["sampled_ns"]) if self.python is not None else "n/a", _us(s["launch_ns"]),
                _us(s["device_ns"]) if self.device is not None else "n/a", s["ops"], _short_left(site, 90)))
        return "\n".join(lines)

    def summary(self, row_limit=12):
        h = self.host
        wall = float(self.wall_ns) or 1.0
        lines = ["jt.profile: %s wall, %d executor batches, %d operators launched%s" % (
            _ms(self.wall_ns), h["batches"], h["operators"],
            ", %d steps" % len(self.steps) if self.steps else "")]
        if self.truncated:
            lines.append("  WARNING: the trace hit its record limit; later records are missing")
        lines.append("Host")
        for key, label in (("python_and_graph_build", "Python + graph construction"),
                           ("executor_planning", "executor planning"),
                           ("executor_compile", "JIT compile / kernel lookup"),
                           ("operator_launch", "operator launch"),
                           ("device_wait", "waiting on device"),
                           ("graph_launch", "device graph launch")):
            lines.append("  %-28s %12s %5.1f%%" % (label, _ms(h[key]), 100 * h[key] / wall))
        if h["operators"]:
            lines.append("  (launch: %.1f us per operator, of which allocation %.1f us)" % (
                h["operator_launch"] / 1e3 / h["operators"], h["operator_alloc"] / 1e3 / h["operators"]))
        lines.append("Device")
        if self.device is None:
            lines.append("  unavailable: %s" % self.device_note)
        else:
            d = self.device
            lines.append("  busy %s (%.1f%% of wall), kernels %s in %d launches, copies/memsets %s in %d" % (
                _ms(d["busy"]), 100 * d["utilization"], _ms(d["kernel"]), d["kernels"],
                _ms(d["copy"]), d["copies"]))
            if d["kernel"] + d["copy"]:
                lines.append("  attributed to operators: %.1f%%" % (
                    100.0 * d["attributed"] / (d["kernel"] + d["copy"])))
        verdict, why = self.bound
        lines.append("Verdict: %s (%s)" % (verdict, why))
        r = self.replay
        if r["graph_launches"] or r["sources"]:
            lines.append("Replay: %d device-graph launches in the region" % r["graph_launches"])
            for name, delta, refused in r["sources"]:
                lines.append("  %s: %s%s" % (name, delta, "; refused: %s" % refused if refused else ""))
        lines.append("")
        lines.append(self.table(row_limit=row_limit))
        if self.device is not None:
            lines.append("")
            lines.append(self.kernel_table(row_limit=min(row_limit, 10)))
        lines.append("")
        lines.append(self.site_table(row_limit=min(row_limit, 10)))
        if self.memory is not None:
            lines.append("")
            lines.append(self.memory.summary())
        return "\n".join(lines)

    __str__ = summary

    def export_chrome_trace(self, path):
        from ._trace import write_chrome_trace
        write_chrome_trace(self, path)
        return path
