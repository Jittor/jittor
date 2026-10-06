#!/usr/bin/env python
"""Render or compare saved ``run.py`` results.

    python bench/torch_compat/report.py results.json            # table
    python bench/torch_compat/report.py --markdown results.json
    python bench/torch_compat/report.py old.json new.json       # Jittor over time

Ratio = Jittor time / PyTorch time. Below 1 means Jittor is faster.

Rows are (workload, compile mode). ``agree`` is the largest relative
difference between the two runtimes' per-step values (loss, or the mean of an
output) over the steps both ran -- both start from the same NumPy weights and
inputs, so a large one means a runtime computes something else. A workload
that draws from the framework's generator (dropout) is ``stoch.`` instead.
``compiled`` says what the compiled step did: Jittor's replays and device-graph
launches or its refusal, PyTorch's graph breaks.
"""

import argparse
import json
import sys


def _pairs(document):
    """{(workload, compile mode): {runtime: result}} in first-seen order."""
    rows = {}
    for result in document["results"]:
        key = (result["workload"], result.get("compile") or "none")
        rows.setdefault(key, {})[result["runtime"]] = result
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


def _host(result):
    """Share of the step spent before it returned, i.e. on the host."""
    if not _ok(result) or not result.get("host_s"):
        return "-"
    return "%d%%" % round(100 * result["host_s"] / result["median_s"])


def _steady(result):
    if not _ok(result) or not result.get("steady_device_bytes"):
        return "-"
    return "%.1f GB" % (result["steady_device_bytes"] / 2**30)


def disagreement(a, b):
    """Largest difference between two value sequences, relative to their scale.

    Scaled by the reference's root mean square, not value by value: an
    inference output's mean can sit near zero, where a relative difference
    means nothing. Training is compared over its first two steps only --
    after that two correct runs drift apart by the chaos of training itself
    (a large learning rate on random weights diverges from the last bit).
    """
    if not (_ok(a) and _ok(b)):
        return None
    pairs = [(x, y) for x, y in zip(a.get("values") or [], b.get("values") or [])
             if x is not None and y is not None]
    if (a.get("mode") or b.get("mode")) == "train":
        pairs = pairs[:2]
    if not pairs:
        return None
    scale = max((sum(y * y for _, y in pairs) / len(pairs)) ** 0.5, 1e-12)
    return max(abs(x - y) for x, y in pairs) / scale


def _agree(a, b):
    if (a or {}).get("stochastic") or (b or {}).get("stochastic"):
        return "stoch."
    value = disagreement(a, b)
    if value is None:
        return "-"
    return "%.1e%s" % (value, " !" if value > 5e-2 else "")


def _compiled(result):
    report = (result or {}).get("compile_report")
    if not _ok(result) or not report:
        return "-"
    if report.get("kind") == "dynamo":
        return "breaks %s" % report.get("graph_breaks", "?")
    if report.get("refused"):
        return "refused: " + str(report["refused"])[:40]
    stats = report.get("stats") or {}
    text = "replayed %s, graph %s" % (stats.get("replayed", 0), stats.get("graph", 0))
    if report.get("graph_refused"):
        text += " (no graph: %s)" % str(report["graph_refused"])[:30]
    return text


def _rows(document):
    rows = []
    for (workload, mode), pair in _pairs(document).items():
        any_result = pair.get("torch") or pair.get("jittor")
        ratio = _ratio(pair)
        torch, jittor = pair.get("torch"), pair.get("jittor")
        rows.append([
            workload, mode,
            any_result.get("mode", ""),
            any_result.get("dtype", ""),
            _time(torch), _time(jittor),
            "%.2fx" % ratio if ratio is not None else "-",
            _host(torch), _host(jittor),
            _memory(torch), _memory(jittor), _steady(jittor),
            _first(jittor),
            _agree(jittor, torch),
            _compiled(jittor) if mode != "none" else "-",
            _compiled(torch) if mode != "none" else "-",
        ])
    return rows


HEADER = ["workload", "compile", "mode", "dtype", "torch", "jittor", "ratio",
          "torch host", "jittor host", "torch mem", "jittor mem",
          "jittor steady", "jittor 1st step", "agree", "jittor compiled",
          "torch compiled"]


def _compile_rows(document):
    """Each runtime compiled against itself eager, where both were run."""
    pairs = _pairs(document)
    rows = []
    for (workload, mode), pair in pairs.items():
        if mode == "none" or (workload, "none") not in pairs:
            continue
        eager = pairs[(workload, "none")]
        row = [workload, mode]
        for runtime in ("torch", "jittor"):
            before, after = eager.get(runtime), pair.get(runtime)
            if _ok(before) and _ok(after):
                row.append("%.2fx" % (before["median_s"] / after["median_s"]))
            else:
                row.append("-")
            row.append(_agree(after, before))
        rows.append(row)
    return rows


COMPILE_HEADER = ["workload", "compile", "torch speedup", "torch agree",
                  "jittor speedup", "jittor agree"]


def geomean(values):
    import math

    values = [v for v in values if v]
    if not values:
        return None
    return math.exp(sum(math.log(v) for v in values) / len(values))


def _plain_table(header, rows):
    rows = [header] + rows
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(header))]
    lines = []
    for index, row in enumerate(rows):
        lines.append("  ".join(str(cell).ljust(widths[i])
                               for i, cell in enumerate(row)).rstrip())
        if index == 0:
            lines.append("  ".join("-" * w for w in widths))
    return lines


def summary(document):
    ratios = [_ratio(pair) for pair in _pairs(document).values()]
    ok = [r for r in ratios if r is not None]
    mean = geomean(ok)
    text = "%d/%d workloads compared" % (len(ok), len(ratios))
    if mean is not None:
        text += "; geometric-mean ratio %.2fx" % mean
    return text


def render_table(document):
    lines = _plain_table(HEADER, _rows(document))
    compile_rows = _compile_rows(document)
    if compile_rows:
        lines += ["", "compiled against eager, per runtime (speedup = eager / compiled):"]
        lines += _plain_table(COMPILE_HEADER, compile_rows)
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
    compile_rows = _compile_rows(document)
    if compile_rows:
        lines += ["", "Compiled against eager, per runtime (speedup = eager / compiled):", ""]
        lines.append("| " + " | ".join(COMPILE_HEADER) + " |")
        lines.append("|" + "---|" * len(COMPILE_HEADER))
        for row in compile_rows:
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
    for key, pair in new_pairs.items():
        workload = key[0] if key[1] == "none" else "%s [%s]" % key
        before = old_pairs.get(key, {})
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
