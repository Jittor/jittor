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
                 on_host=False, stacks=None, dropped_device_records=0):
        self.t0, self.t1 = t0, t1
        self.wall_ns = max(t1 - t0, 0)
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
        self.dropped_device_records = int(dropped_device_records)
        # Stack id -> frames ({"name", "filename", "line"}, innermost first),
        # when the profile recorded stacks.
        self.stacks = stacks
        self._by_seq = {op["seq"]: op for op in ops}
        self._op_wait = self._waits_inside_ops()
        for rec in kernels + copies:
            rec["seq"] = correlation.get(rec["correlation"], -1)
        self.host = self._host_split()
        self.device = self._device_split() if device_note is None else None

    def _waits_inside_ops(self):
        """Device waits that happened while an operator was being launched:
        a blocking copy or allocation inside the op, charged to the innermost
        enclosing operator (handling nested batches/operators)."""
        ops = sorted(((o["t_start"], o["t_end"], o["seq"]) for o in self.op_records if o["kind"] == OP),
                     key=lambda x: (x[0], -x[1], x[2]))
        waits = sorted(self.waits, key=lambda w: w["t_begin"])
        out = defaultdict(int)
        active = []
        idx = 0
        n_ops = len(ops)
        for w in waits:
            wb, we = w["t_begin"], w["t_end"]
            while idx < n_ops and ops[idx][0] <= wb:
                while active and active[-1][1] <= ops[idx][0]:
                    active.pop()
                active.append(ops[idx])
                idx += 1
            while active and active[-1][1] <= wb:
                active.pop()
            for o_start, o_end, seq in reversed(active):
                if o_start <= wb and we <= o_end:
                    out[seq] += max(we - wb, 0)
                    break
        return out

    # -- host ---------------------------------------------------------------
    @staticmethod
    def _overlap(a0, a1, b0, b1):
        if a1 <= a0 or b1 <= b0:
            return 0
        lo = a0 if a0 > b0 else b0
        hi = a1 if a1 < b1 else b1
        return hi - lo if hi > lo else 0

    def _host_split(self):
        batches = self.batches
        by_id = {b["batch"]: b for b in batches}
        children_of = defaultdict(list)
        for b in batches:
            if b["parent"] in by_id:
                children_of[b["parent"]].append((b["t_begin"], b["t_end"]))
        waits_in = defaultdict(list)
        outside_waits = []
        for w in self.waits:
            if w["batch"] in by_id:
                waits_in[w["batch"]].append((w["t_begin"], w["t_end"]))
            else:
                outside_waits.append((w["t_begin"], w["t_end"]))

        planning = compile_ = launch = wait = 0
        top = 0
        for b in batches:
            t_begin = b["t_begin"]
            t_end = b["t_end"]
            if t_end <= t_begin:
                continue
            t_planned = b["t_planned"] if b["t_planned"] and b["t_planned"] >= t_begin else 0
            t_compiled = b["t_compiled"] if b["t_compiled"] and t_planned and b["t_compiled"] >= t_planned else 0
            # Partition [t_begin, t_end] into its active phases, handling
            # incomplete batches (e.g. aborted during planning or compile).
            p_end = min(t_planned, t_end) if t_planned else t_end
            c_start = p_end
            c_end = min(t_compiled, t_end) if t_compiled else (t_end if t_planned else p_end)
            r_start = c_end
            r_end = t_end if t_compiled else c_end

            child_p = child_c = child_r = 0
            for cb, ce in children_of.get(b["batch"], ()):
                child_p += self._overlap(t_begin, p_end, cb, ce)
                child_c += self._overlap(c_start, c_end, cb, ce)
                child_r += self._overlap(r_start, r_end, cb, ce)

            wait_p = wait_c = wait_r = 0
            for wb, we in waits_in.get(b["batch"], ()):
                wait_p += self._overlap(t_begin, p_end, wb, we)
                wait_c += self._overlap(c_start, c_end, wb, we)
                wait_r += self._overlap(r_start, r_end, wb, we)

            planning += max((p_end - t_begin) - child_p - wait_p, 0)
            compile_ += max((c_end - c_start) - child_c - wait_c, 0)
            launch += max((r_end - r_start) - child_r - wait_r, 0)
            wait += wait_p + wait_c + wait_r
            if b["parent"] not in by_id:
                top += t_end - t_begin

        graph_ops = [(o["t_start"], o["t_end"]) for o in self.op_records
                     if o["kind"] == GRAPH_LAUNCH and o["batch"] not in by_id]
        graph_raw = sum(max(ge - gs, 0) for gs, ge in graph_ops)
        outside_wait = sum(max(we - wb, 0) for wb, we in outside_waits)
        wait_in_graph = 0
        for wb, we in outside_waits:
            for gs, ge in graph_ops:
                wait_in_graph += self._overlap(gs, ge, wb, we)
        graph = max(graph_raw - wait_in_graph, 0)

        launched = [o for o in self.op_records if o["kind"] == OP]
        # An operator compiles its kernel the first time it runs; that is
        # compilation, not launching.
        in_ops = min(sum(max(o["compile_ns"], 0) for o in launched), launch)
        compile_ += in_ops
        launch -= in_ops
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
            "operator_alloc": sum(max(o["t_alloc"] - o["t_start"], 0) for o in launched),
        }

    # -- device ---------------------------------------------------------------
    def _device_split(self):
        records = self.kernel_records + self.copy_records
        clamped = [(max(r["start"], self.t0), min(r["end"], self.t1)) for r in records]
        clamped = [(a, b) for a, b in clamped if b > a]
        busy = _union(clamped)
        kernel_ns = sum(max(r["end"] - r["start"], 0) for r in self.kernel_records)
        copy_ns = sum(max(r["end"] - r["start"], 0) for r in self.copy_records)
        attributed = sum(max(r["end"] - r["start"], 0) for r in records if r["seq"] in self._by_seq)
        return {"busy": busy, "kernel": kernel_ns, "copy": copy_ns,
                "kernels": len(self.kernel_records), "copies": len(self.copy_records),
                "attributed": attributed,
                "utilization": min(busy / float(self.wall_ns), 1.0) if self.wall_ns else 0.0}

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
        unattributed_ns = 0
        unattributed_kernels = 0
        for r in self.kernel_records + self.copy_records:
            dur = max(r["end"] - r["start"], 0)
            seq = r["seq"]
            if seq in self._by_seq:
                device[seq] += dur
                kernels[seq] += 1
            else:
                unattributed_ns += dur
                unattributed_kernels += 1
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
            st.host_ns += max(o["t_end"] - o["t_start"] - wait - o["compile_ns"], 0)
            st.alloc_ns += max(o["t_alloc"] - o["t_start"], 0)
            st.device_ns += device.get(o["seq"], 0)
            st.kernels += kernels.get(o["seq"], 0)
            st.out_bytes += o["out_bytes"]
            if o["site"] >= 0:
                st.sites[_site(self.strings[o["site"]])] += 1
        if unattributed_ns or unattributed_kernels:
            st = stats[("(device work outside any operator)", "")] = OpStat(
                "(device work outside any operator)", "", -1)
            st.device_ns = unattributed_ns
            st.kernels = unattributed_kernels
        return list(stats.values())

    def table(self, sort_by="device_time", row_limit=20, group_by_shapes=False, width=48):
        """The operator table, like ``torch.profiler``'s ``key_averages().table()``."""
        rows = self.op_stats(group_by_shapes)
        keys = {"device_time": lambda s: (s.device_ns, s.host_ns),
                "host_time": lambda s: (s.host_ns, s.device_ns),
                "calls": lambda s: (s.calls, s.device_ns, s.host_ns),
                "name": lambda s: s.name}
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
            dev_pct = ("%5.1f%%" % (100 * r.device_ns / total_dev)) if self.device is not None else "   n/a"
            lines.append("%-*s %6d %10s %5.1f%% %9s %10s %6s %8s %7d  %s" % (
                width, _short(r.name, width), r.calls, _us(r.host_ns), 100 * r.host_ns / total_host,
                _us(r.wait_ns) if r.wait_ns else "",
                _us(r.device_ns) if self.device is not None else "n/a",
                dev_pct,
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
            s["launch_ns"] += o["t_end"] - o["t_start"] - o["compile_ns"]
            s["device_ns"] += device.get(o["seq"], 0)
        if self.python is not None:
            sites_ns = getattr(self.python, "sites_ns", None)
            if sites_ns:
                for site, ns in sites_ns.items():
                    sites[_site(site)]["sampled_ns"] += int(ns)
            else:
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
        if self.dropped_device_records:
            lines.append("  WARNING: CUPTI dropped %d device activity records" % self.dropped_device_records)
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
