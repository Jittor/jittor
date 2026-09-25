"""What a profiled region did to device memory, from the allocator's own log.

The step tracer records every pool allocation and free, every workspace
(TempAllocator) allocation and free, and every change in what the pools hold
from the driver, each with a timestamp and -- for a Var -- the Var's dtype and
shape, the operator producing it and the Python line that built that
operator. Allocations already live when the trace opened are synthesized at
start from the graph reachable from the held Vars.

Replaying that log gives the exact high-water mark (it must equal the pools'
own ``device_memory_peak``, and the report says so when it does not), the
moment it happened, and the set of allocations live at that moment -- which
is what "what was live at the peak" means. Nothing is sampled.
"""

from collections import defaultdict

MiB = float(1 << 20)

POOL, TEMP, RESERVE, EXISTING = 0, 1, 2, 3


def _fmt(nbytes):
    if nbytes is None:
        return "n/a"
    return "%.1f MiB" % (nbytes / MiB)


class MemoryReport:
    """Memory of one device over a profiled region. Byte counts are ints."""

    def __init__(self, device, events, strings, start, end, pool_peaks, nvml, t0):
        self.device = device
        self.t0 = t0
        self.start = start          # device_pool_stats at start
        self.end = end              # device_pool_stats at end
        self.pool_peak_allocated, self.pool_peak_reserved = pool_peaks
        self.nvml = nvml
        self._strings = strings
        self._analyse(events)

    # -- analysis ---------------------------------------------------------
    def _tag(self, event):
        s = self._strings
        desc = s[event["var_desc"]] if event["var_desc"] > 0 else ""
        producer = s[event["producer"]] if event["producer"] > 0 else ""
        site = s[event["site"]] if event["site"] >= 0 else ""
        if site:
            from ._report import _site
            site = _site(site)
        return (event["kind"], producer, desc, site, event["op"])

    def _analyse(self, events):
        events = [e for e in events if e["device"] == self.device]
        events.sort(key=lambda e: e["t"])
        existing = [e for e in events if e["kind"] == EXISTING]
        stream = [e for e in events if e["kind"] != EXISTING]
        self.existing_bytes = sum(e["bytes"] for e in existing)
        used0, reserved0, temp0 = self.start[0], self.start[1], self.start[6]
        self.untracked_at_start = used0 - self.existing_bytes

        allocated, temp, reserved = used0, temp0, reserved0
        peak = (allocated, -1)
        peak_total = (allocated + temp, -1)
        peak_reserved = reserved
        timeline = [(self.t0, allocated, temp, reserved)]
        self.alloc_count = self.free_count = 0
        self.allocated_total = 0
        for index, e in enumerate(stream):
            kind, nbytes = e["kind"], e["bytes"]
            if kind == POOL:
                allocated += nbytes
                if nbytes > 0:
                    self.alloc_count += 1
                    self.allocated_total += nbytes
                else:
                    self.free_count += 1
                if allocated > peak[0]:
                    peak = (allocated, index)
            elif kind == TEMP:
                temp += nbytes
            elif kind == RESERVE:
                reserved += nbytes
                peak_reserved = max(peak_reserved, reserved)
            if allocated + temp > peak_total[0]:
                peak_total = (allocated + temp, index)
            timeline.append((e["t"], allocated, temp, reserved))
        self.peak_allocated, peak_index = peak
        self.peak_with_workspace = peak_total[0]
        self.peak_reserved_traced = peak_reserved
        self.peak_time = stream[peak_index]["t"] if peak_index >= 0 else self.t0
        self.timeline = timeline

        # Second pass: the live set at the peak.
        live = {}
        for e in existing:
            live[(e["allocator"], e["allocation"])] = (e["bytes"], ("existing",) + self._tag(e)[1:])
        for e in stream[:peak_index + 1]:
            if e["kind"] != POOL:
                continue
            key = (e["allocator"], e["allocation"])
            if e["bytes"] > 0:
                live[key] = (e["bytes"], ("new",) + self._tag(e)[1:])
            else:
                live.pop(key, None)
        self.live_at_peak = list(live.values())
        self.unattributed_at_peak = self.peak_allocated - sum(b for b, _ in self.live_at_peak)

    # -- views ------------------------------------------------------------
    def live_by(self, key="op"):
        """Bytes live at the peak grouped by "op", "site", "tensor" or "origin"
        ("new" = allocated inside the region, "existing" = before it)."""
        groups = defaultdict(lambda: [0, 0])
        for nbytes, (origin, producer, desc, site, _op) in self.live_at_peak:
            if key == "op":
                label = producer or ("(allocated before profile)" if origin == "existing" else "(no producer)")
            elif key == "site":
                label = site or ("(allocated before profile)" if origin == "existing" else "(no Python site)")
            elif key == "tensor":
                label = desc or "?"
            elif key == "origin":
                label = "allocated before profile" if origin == "existing" else "allocated in profile"
            else:
                raise ValueError("group by 'op', 'site', 'tensor' or 'origin', not %r" % (key,))
            groups[label][0] += nbytes
            groups[label][1] += 1
        return sorted(((label, b, n) for label, (b, n) in groups.items()), key=lambda r: -r[1])

    @property
    def fragmentation(self):
        """1 - largest cached block / all cached bytes, at the end; 0 when
        nothing is cached."""
        cached = self.end[1] - self.end[0]
        if cached <= 0:
            return 0.0
        return 1.0 - self.end[5] / float(cached)

    def to_dict(self):
        return {
            "device": self.device,
            "peak_allocated": self.peak_allocated,
            "pool_peak_allocated": self.pool_peak_allocated,
            "peak_with_workspace": self.peak_with_workspace,
            "peak_reserved": self.pool_peak_reserved,
            "start": {"allocated": self.start[0], "reserved": self.start[1]},
            "end": {"allocated": self.end[0], "reserved": self.end[1],
                    "free_blocks": self.end[4], "largest_free_block": self.end[5],
                    "workspace_reserved": self.end[7]},
            "fragmentation": self.fragmentation,
            "nvml": dict(self.nvml),
            "live_at_peak_by_op": self.live_by("op"),
            "live_at_peak_by_site": self.live_by("site"),
            "live_at_peak_by_tensor": self.live_by("tensor"),
            "allocations": self.alloc_count,
            "frees": self.free_count,
            "allocated_total": self.allocated_total,
        }

    def summary(self, row_limit=8):
        name = "cuda:%d" % self.device if self.device >= 0 else "host"
        lines = ["Memory (%s)" % name]
        check = ""
        if self.pool_peak_allocated != self.peak_allocated:
            check = "  [pool counter says %s]" % _fmt(self.pool_peak_allocated)
        lines.append("  peak allocated (Vars)         %s at +%.3f ms%s" % (
            _fmt(self.peak_allocated), (self.peak_time - self.t0) / 1e6, check))
        lines.append("  peak incl. workspaces          %s" % _fmt(self.peak_with_workspace))
        lines.append("  peak reserved by pools         %s" % _fmt(self.pool_peak_reserved))
        nv = self.nvml
        if nv.get("error"):
            lines.append("  NVML                           unavailable: %s" % nv["error"])
        else:
            lines.append("  NVML process used              start %s, end %s, sampled peak %s (%d samples)" % (
                _fmt(nv.get("start")), _fmt(nv.get("end")), _fmt(nv.get("peak")), nv.get("samples", 0)))
            if nv.get("end") is not None:
                lines.append("    outside the pools at end     %s (context, modules, library handles)" % _fmt(
                    nv["end"] - self.end[1] - self.end[7]))
        lines.append("  start -> end allocated         %s -> %s; reserved %s -> %s" % (
            _fmt(self.start[0]), _fmt(self.end[0]), _fmt(self.start[1]), _fmt(self.end[1])))
        lines.append("  cached at end                  %s in %d free blocks, largest %s, fragmentation %.0f%%" % (
            _fmt(self.end[1] - self.end[0]), self.end[4], _fmt(self.end[5]), 100 * self.fragmentation))
        lines.append("  allocations / frees in region  %d / %d (%s allocated in total)" % (
            self.alloc_count, self.free_count, _fmt(self.allocated_total)))
        for key, title in (("origin", "live at peak"), ("op", "live at peak by producing op"),
                           ("site", "live at peak by Python call site"),
                           ("tensor", "live at peak by tensor")):
            rows = self.live_by(key)
            lines.append("  %s:" % title)
            for label, nbytes, count in rows[:row_limit]:
                lines.append("    %12s  %5d x  %s" % (_fmt(nbytes), count, label))
            if len(rows) > row_limit:
                rest = sum(r[1] for r in rows[row_limit:])
                lines.append("    %12s  (%d more groups)" % (_fmt(rest), len(rows) - row_limit))
        # Allocations that predate the profile are counted at their tensor size,
        # the pools' at their aligned block size; below a MiB that difference is
        # all this line would show.
        if abs(self.unattributed_at_peak) >= MiB:
            lines.append("  not attributed to a Var at peak %s" % _fmt(self.unattributed_at_peak))
        return "\n".join(lines)
