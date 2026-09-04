"""Classify a kernel report into roles and apply the measured roofline.

Three input kinds:

    classify.py jittor <report.json> --achievable-gbps G
    classify.py nsys   <kern_sum.csv> --jittor <report.json> --steps N
                       --achievable-gbps G
    classify.py torch  <report.json>

Byte accounting for Jittor comes from the profiler's ``Input``/``Output``
columns, which count only the *external* inputs and outputs of each (fused)
operator: fused intermediates are excluded, so the sum is the DRAM traffic a
perfect implementation would move.  Divided by the machine's measured
achievable copy bandwidth it gives a per-kernel time floor.  A ratio above 1
means the kernel is not bandwidth bound; a ratio below 1 means part of its
traffic was served by L2 and the floor is not tight for it.
"""

import argparse
import csv
import json
import re

LIBRARY = ("cudnn", "cublas", "cutt", "cub", "curand", "cusparse", "cufft",
           "mkl", "nccl")
INDEXING = ("getitem", "setitem", "transpose")

TORCH_ELEMENTWISE = ("elementwise_kernel", "vectorized_elementwise",
                     "unrolled_elementwise", "CatArrayBatchedCopy",
                     "index_elementwise", "fill_", "upsample_nearest")
TORCH_REDUCE = ("reduce_kernel", "reduce", "batch_norm_collect", "welford",
                "softmax", "norm")
TORCH_GEMM = ("gemm", "cutlass", "sm80", "sm86", "sm89", "ampere", "cublas",
              "wgrad", "dgrad", "implicit", "conv", "cudnn", "xmma",
              "winograd", "nchw", "nhwc", "fmha")


def reps_per_step(rerun):
    """How many times the Jittor profiler reruns each operator per record.

    ``Profiler::record_and_run`` computes ``r = get_nbits(rerun+1) - 2`` and
    runs the operator ``1 << r`` times, and ``NanoVector::get_nbits(v)`` is
    ``65 - lzcnt64(v)``, i.e. ``v.bit_length() + 1``.  So ``r`` is
    ``bit_length(rerun+1) - 1``.  Getting this off by one scales every
    per-step figure by two and the report stays internally consistent, so it
    is invisible: cross-check per-step call counts against nsys instance
    counts before trusting them.
    """
    return 1 << max(max(rerun + 1, 1).bit_length() - 1, 0)


def op_names(key):
    body = key.split("\u00abJIT")[0]
    names = re.findall(r"opkey\d+:([a-zA-Z_0-9]+)", body)
    return names or [re.split(r"[\u00ab\[]", body)[0] or "?"]


def classify(key, names):
    head = re.split(r"[\u00ab\[]", key)[0]
    for library in LIBRARY:
        if library in head:
            return "library:" + library
    for name in INDEXING:
        if head.startswith(name):
            return "indexing"
    if names == ["code"] or head.startswith("code"):
        # jt.code kernels are hand-written CUDA from nn/backends, not the
        # code generator's output; keeping them separate stops them from
        # flattering or damning the codegen number.
        return "handwritten:code"
    if any(n.startswith("reduce") for n in names):
        return "reduce"
    return "elementwise"


def classify_torch(name):
    lowered = name.lower()
    for hint in TORCH_GEMM:
        if hint in lowered:
            return "conv/gemm"
    for hint in TORCH_REDUCE:
        if hint.lower() in lowered:
            return "reduce/norm"
    for hint in TORCH_ELEMENTWISE:
        if hint.lower() in lowered:
            return "elementwise"
    return "other"


def load_jittor(path, achievable):
    with open(path) as handle:
        payload = json.load(handle)
    index = {name: i for i, name in enumerate(payload["header"])}
    reps = reps_per_step(payload["rerun"])
    entries = []
    for row in payload["records"]:
        key = row[index["Name"]]
        count = int(row[index["Count"]])
        avg_ns = float(row[index["TotalTime"]]) / count
        rate = (float(row[index["Input"]]) + float(row[index["Output"]])) / 1e9
        per_call_bytes = rate * avg_ns
        calls = count // reps
        names = op_names(key)
        hashed = re.search(r"_hash_([0-9a-f]+)_op\.cc", row[index["FileName"]])
        entries.append({
            "hash": hashed.group(1) if hashed else "",
            "ops": names,
            "role": classify(key, names),
            "calls": calls,
            "avg_us": avg_ns / 1e3,
            "step_us": avg_ns * calls / 1e3,
            "step_mb": per_call_bytes * calls / 1048576.0,
            "bytes": per_call_bytes,
            "gbps": per_call_bytes / avg_ns if avg_ns else 0.0,
            "step_floor_us": per_call_bytes / achievable * calls / 1e3,
            "ratio": avg_ns / (per_call_bytes / achievable)
                     if per_call_bytes else 0.0,
        })
    return payload, entries


def load_nsys(csv_path, jittor_path, steps, achievable):
    _, jittor_entries = load_jittor(jittor_path, achievable)
    by_hash = {e["hash"]: e for e in jittor_entries if e["hash"]}
    entries = []
    with open(csv_path) as handle:
        for record in csv.DictReader(handle):
            name = record["Name"]
            step_us = float(record["Total Time (ns)"]) / 1e3 / steps
            calls = int(record["Instances"]) / steps
            match = re.match(r"func_([0-9a-f]+)_\d+", name)
            source = by_hash.get(match.group(1)) if match else None
            if source is None:
                # Jittor's own non-fused kernels keep their C++ symbol name;
                # everything else is a vendor library kernel.
                if any(k in name for k in INDEXING):
                    role = "indexing"
                elif "jittor::" in name:
                    # Jittor's own hand-written CUDA (nn/backends, jt.code):
                    # its symbols live in the jittor namespace, vendor kernels
                    # do not.
                    role = "handwritten:code"
                else:
                    role = "library:" + _torch_library(name)
                entries.append({
                    "ops": [name[:70]], "role": role,
                    "calls": calls, "step_us": step_us, "step_mb": 0.0,
                    "gbps": 0.0, "step_floor_us": 0.0, "ratio": 0.0})
                continue
            step_mb = source["bytes"] * calls / 1048576.0
            entries.append({
                "ops": source["ops"], "role": source["role"], "calls": calls,
                "step_us": step_us, "step_mb": step_mb,
                "gbps": step_mb * 1048576.0 / (step_us * 1e3) if step_us else 0.0,
                "step_floor_us": step_mb * 1048576.0 / achievable / 1e3,
                "ratio": 0.0})
    for entry in entries:
        if entry["step_floor_us"]:
            entry["ratio"] = entry["step_us"] / entry["step_floor_us"]
    return {"case": "nsys", "tag": csv_path}, entries


def _torch_library(name):
    lowered = name.lower()
    for library in LIBRARY:
        if library in lowered:
            return library
    if any(h in lowered for h in ("xmma", "dgrad", "wgrad", "implicit_gemm",
                                  "winograd", "nchw", "nhwc")):
        return "cudnn"
    if "gemm" in lowered or "cutlass" in lowered:
        return "cublas"
    return "other"


def report(entries, top, header):
    roles = {}
    for entry in entries:
        bucket = roles.setdefault(entry["role"], {"kinds": 0, "calls": 0.0,
                                                  "us": 0.0, "mb": 0.0,
                                                  "floor": 0.0})
        bucket["kinds"] += 1
        bucket["calls"] += entry["calls"]
        bucket["us"] += entry["step_us"]
        bucket["mb"] += entry["step_mb"]
        bucket["floor"] += entry["step_floor_us"]

    print(header)
    print("%-18s %6s %7s %10s %10s %9s %7s"
          % ("role", "kinds", "calls", "step_us", "floor_us", "GB/s", "x"))
    total = 0.0
    for role in sorted(roles, key=lambda r: -roles[r]["us"]):
        b = roles[role]
        gbps = b["mb"] * 1048576.0 / (b["us"] * 1e3) if b["us"] else 0.0
        print("%-18s %6d %7.0f %10.1f %10.1f %9.1f %7.2f"
              % (role, b["kinds"], b["calls"], b["us"], b["floor"], gbps,
                 b["us"] / b["floor"] if b["floor"] else 0.0))
        total += b["us"]
    print("%-18s %6s %7s %10.1f" % ("TOTAL", "", "", total))

    if not top:
        return
    element = [e for e in entries if e["role"] == "elementwise"]
    element.sort(key=lambda e: -(e["step_us"] - e["step_floor_us"]))
    print()
    print("elementwise kernels ranked by excess over the roofline floor")
    print("%4s %6s %8s %8s %8s %8s %7s  %s"
          % ("#", "calls", "step_us", "floor", "excess", "GB/s", "x", "ops"))
    for rank, e in enumerate(element[:top], 1):
        print("%4d %6.0f %8.1f %8.1f %8.1f %8.1f %7.2f  %s"
              % (rank, e["calls"], e["step_us"], e["step_floor_us"],
                 e["step_us"] - e["step_floor_us"], e["gbps"], e["ratio"],
                 ",".join(e["ops"][:7])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("jittor", "nsys", "torch"))
    parser.add_argument("report")
    parser.add_argument("--jittor", default="", help="nsys mode: the Jittor report")
    parser.add_argument("--steps", type=int, default=1, help="nsys mode")
    parser.add_argument("--achievable-gbps", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=20)
    options = parser.parse_args()

    if options.kind == "torch":
        with open(options.report) as handle:
            payload = json.load(handle)
        entries = [{"ops": [r["name"][:70]], "role": classify_torch(r["name"]),
                    "calls": r["calls"], "step_us": r["step_us"],
                    "step_mb": 0.0, "gbps": 0.0, "step_floor_us": 0.0,
                    "ratio": 0.0}
                   for r in payload["records"]]
        report(entries, 0, "== torch %s case %s" % (payload["torch"], payload["case"]))
        return

    assert options.achievable_gbps > 0, "--achievable-gbps is required"
    if options.kind == "jittor":
        payload, entries = load_jittor(options.report, options.achievable_gbps)
        label = "== jittor profiler (per-operator, isolated)  case=%s tag=%s" % (
            payload["case"], payload.get("tag", ""))
    else:
        assert options.jittor, "--jittor <report.json> is required for nsys"
        _, entries = load_nsys(options.report, options.jittor, options.steps,
                               options.achievable_gbps)
        label = "== nsys (real pipelined step, %d steps)" % options.steps
    report(entries, options.top, label)


if __name__ == "__main__":
    main()
