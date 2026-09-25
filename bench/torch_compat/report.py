#!/usr/bin/env python
"""Render or compare saved ``run.py`` results.

    python bench/torch_compat/report.py results.json            # table
    python bench/torch_compat/report.py --markdown results.json
    python bench/torch_compat/report.py old.json new.json       # Jittor over time

Ratio = Jittor time / PyTorch time. Below 1 means Jittor is faster.
"""

import argparse
import json
import sys


def _pairs(document):
    """{workload: {runtime: result}} in first-seen order."""
    rows = {}
    for result in document["results"]:
        rows.setdefault(result["workload"], {})[result["runtime"]] = result
    return rows


def _ok(result):
    return result is not None and result.get("status") == "ok"


def _ratio(pair):
    torch, jittor = pair.get("torch"), pair.get("jittor")
    if _ok(torch) and _ok(jittor):
        return jittor["median_s"] / torch["median_s"]
    return None


def _time(result):
    if result is None:
        return "-"
    if not _ok(result):
        return result.get("status", "?").upper()
    value = result["median_s"]
    return "%.1f ms" % (value * 1e3) if value < 10 else "%.2f s" % value


def _throughput(result):
    if not _ok(result):
        return "-"
    return "%.4g %s/s" % (result["throughput"], result.get("unit", ""))


def _memory(result):
    """Peak device memory the process held (NVML), shown for failures too.

    An OOM row keeps its number: how high it got before the card ran out is
    the comparison that matters there. Falls back to the runtime's own
    allocator counter only when NVML was unavailable.
    """
    if result is None:
        return "-"
    value = result.get("peak_device_bytes")
    if not value and _ok(result):
        value = result.get("peak_memory_bytes")
    if not value:
        return "-"
    return "%.1f GB" % (value / 2**30)


def _first(result):
    if not _ok(result):
        return "-"
    return "%.1f s" % result["first_step_s"]


def _rows(document):
    rows = []
    for workload, pair in _pairs(document).items():
        any_result = pair.get("torch") or pair.get("jittor")
        ratio = _ratio(pair)
        rows.append([
            workload,
            any_result.get("mode", ""),
            any_result.get("dtype", ""),
            _time(pair.get("torch")),
            _time(pair.get("jittor")),
            "%.2fx" % ratio if ratio is not None else "-",
            _throughput(pair.get("torch")),
            _throughput(pair.get("jittor")),
            _memory(pair.get("torch")),
            _memory(pair.get("jittor")),
            _first(pair.get("jittor")),
        ])
    return rows


HEADER = ["workload", "mode", "dtype", "torch", "jittor", "ratio",
          "torch thpt", "jittor thpt", "torch mem", "jittor mem",
          "jittor 1st step"]


def geomean(values):
    import math

    values = [v for v in values if v]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def summary(document):
    ratios = [_ratio(pair) for pair in _pairs(document).values()]
    ok = [r for r in ratios if r is not None]
    mean = geomean(ok)
    text = "%d/%d workloads compared" % (len(ok), len(ratios))
    if mean is not None:
        text += "; geometric-mean ratio %.2fx" % mean
    return text


def render_table(document):
    rows = [HEADER] + _rows(document)
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(HEADER))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(str(cell).ljust(widths[i])
                               for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    lines.append("")
    lines.append(summary(document))
    failures = [r for r in document["results"] if not _ok(r)]
    for result in failures:
        error = (result.get("error") or "").strip().splitlines()
        lines.append("%s/%s %s: %s" % (
            result["workload"], result["runtime"], result.get("status"),
            error[-1] if error else "see " + str(result.get("log"))))
    return "\n".join(lines)


def render_markdown(document):
    meta = document["meta"]
    lines = [
        "# Torch compat vs PyTorch",
        "",
        "- commit: `%s`%s" % ((meta.get("commit") or "?")[:12],
                             " (dirty)" if meta.get("dirty") else ""),
        "- device: %s (%s)" % (meta.get("device"), meta.get("gpu")),
        "- size: %s, batch override: %s, TF32: %s, cudnn.benchmark: %s" % (
            meta.get("size"), meta.get("batch"), meta.get("tf32"),
            meta.get("cudnn_benchmark")),
        "- started: %s" % meta.get("started"),
    ]
    versions = {}
    for result in document["results"]:
        if result.get("versions"):
            versions.setdefault(result["runtime"], result["versions"])
    for runtime, report in sorted(versions.items()):
        lines.append("- %s: %s" % (runtime, ", ".join(
            "%s %s" % (k, v) for k, v in report.items() if v is not None)))
    lines += ["", "Ratio = Jittor time / PyTorch time (median step); "
              "below 1 means Jittor is faster.", ""]
    lines.append("| " + " | ".join(HEADER) + " |")
    lines.append("|" + "---|" * len(HEADER))
    for row in _rows(document):
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    lines += ["", summary(document)]
    failures = [r for r in document["results"] if not _ok(r)]
    if failures:
        lines += ["", "## Failures", ""]
        for result in failures:
            error = (result.get("error") or "").strip().splitlines()
            lines.append("- `%s` / %s — %s: `%s`" % (
                result["workload"], result["runtime"], result.get("status"),
                error[-1] if error else result.get("log")))
    return "\n".join(lines) + "\n"


def compare(old, new):
    old_pairs, new_pairs = _pairs(old), _pairs(new)
    header = ["workload", "old ratio", "new ratio", "jittor old", "jittor new",
              "jittor speedup"]
    rows = [header]
    for workload, pair in new_pairs.items():
        before = old_pairs.get(workload, {})
        old_ratio, new_ratio = _ratio(before), _ratio(pair)
        jo, jn = before.get("jittor"), pair.get("jittor")
        speedup = (jo["median_s"] / jn["median_s"]
                   if _ok(jo) and _ok(jn) else None)
        rows.append([
            workload,
            "%.2fx" % old_ratio if old_ratio else "-",
            "%.2fx" % new_ratio if new_ratio else "-",
            _time(jo), _time(jn),
            "%.2fx" % speedup if speedup else "-",
        ])
    widths = [max(len(str(r[i])) for r in rows) for i in range(len(header))]
    lines = ["  ".join(str(c).ljust(widths[i]) for i, c in enumerate(r))
             for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append("")
    lines.append("old %s @ %s" % ((old["meta"].get("commit") or "?")[:10],
                                  old["meta"].get("started")))
    lines.append("new %s @ %s" % ((new["meta"].get("commit") or "?")[:10],
                                  new["meta"].get("started")))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("results", nargs="+")
    parser.add_argument("--markdown", action="store_true")
    options = parser.parse_args()
    documents = [json.load(open(path)) for path in options.results]
    if len(documents) == 2:
        print(compare(*documents))
    elif options.markdown:
        print(render_markdown(documents[0]), end="")
    else:
        print(render_table(documents[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
