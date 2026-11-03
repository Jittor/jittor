#!/usr/bin/env python3
"""Ablation probe for Jittor torch-compat gradient clipping.

The probe is intentionally separate from the implementation under test. Run it
through run_perf_env.sh so CUDA/JIT artifacts stay under this project.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Callable

import numpy as np

import jittor as jt
from jittor.compat.torch import _clip_grad_norm_device


from _paths import WORK_ROOT


def emit(row: dict) -> None:
    print(json.dumps(row, sort_keys=True), flush=True)


def old_clip(grads: list[jt.Var], max_norm: float) -> jt.Var:
    total = jt.sqrt(jt.concat([
        g.cast("float32").sqr().reshape((-1,)) for g in grads
    ]).sum())
    coef = max_norm / (float(total.item()) + 1e-6)
    if coef < 1.0:
        for grad in grads:
            grad.update(grad * coef)
    return total


def current_norm(grads: list[jt.Var]) -> jt.Var:
    parts = [jt.sqrt((g.cast("float32") ** 2).sum()).reshape((1,))
             for g in grads]
    values = jt.concat(parts)
    return jt.sqrt((values * values).sum())


def flat_norm(grads: list[jt.Var]) -> jt.Var:
    flat = jt.concat([g.cast("float32").reshape((-1,)) for g in grads])
    return jt.sqrt((flat * flat).sum())


def sum_norm(grads: list[jt.Var]) -> jt.Var:
    # For p=2, sqrt(sum_i(sqrt(sum(g_i^2))^2)) is exactly
    # sqrt(sum_i(sum(g_i^2))). This preserves O(count) scalar temporaries but
    # removes one sqrt per gradient and the square of every partial norm.
    partials = [(g.cast("float32") ** 2).sum().reshape((1,))
                for g in grads]
    return jt.sqrt(jt.concat(partials).sum())


def device_scale(grads: list[jt.Var], coef: jt.Var) -> None:
    for grad in grads:
        grad.update(grad * coef.cast(str(grad.dtype)))


def host_scale(grads: list[jt.Var], coef: float) -> None:
    for grad in grads:
        grad.update(grad * coef)


def device_clip_with(norm_fn: Callable[[list[jt.Var]], jt.Var],
                     grads: list[jt.Var], max_norm: float) -> jt.Var:
    total = norm_fn(grads)
    if max_norm != float("inf"):
        raw = np.float32(max_norm) / (total + np.float32(1e-6))
        coef = jt.minimum(raw, np.float32(1.0))
        coef = jt.ternary(jt.isnan(raw), raw, coef)
        device_scale(grads, coef)
    return total


def run_mode(mode: str, grads: list[jt.Var], max_norm: float):
    if mode == "old":
        return old_clip(grads, max_norm)
    if mode == "current":
        return _clip_grad_norm_device(grads, max_norm, 2.0)
    if mode == "flat_device":
        return device_clip_with(flat_norm, grads, max_norm)
    if mode == "per_grad_device":
        return device_clip_with(current_norm, grads, max_norm)
    if mode == "sum_device":
        return device_clip_with(sum_norm, grads, max_norm)
    if mode == "norm_current":
        return current_norm(grads)
    if mode == "norm_flat":
        return flat_norm(grads)
    if mode == "norm_sum":
        return sum_norm(grads)
    if mode == "scale_device":
        coef = jt.array(np.array([0.5], dtype="float32"))
        device_scale(grads, coef)
        return coef
    if mode == "scale_host":
        host_scale(grads, 0.5)
        return jt.array(np.array([0.5], dtype="float32"))
    if mode == "noop":
        return grads[0]
    raise ValueError(mode)


def make_pool(count: int, total_elements: int, slots: int,
              seed: int) -> list[list[jt.Var]]:
    rng = np.random.default_rng(seed)
    base, remainder = divmod(total_elements, count)
    sizes = [base + (i < remainder) for i in range(count)]
    return [[jt.array(rng.standard_normal(size).astype("float32"))
             for size in sizes] for _ in range(slots)]


def make_resident(pool: list[list[jt.Var]]) -> None:
    # Optimizer gradients are already on device. Assign a trivial CUDA result
    # before timing so a candidate is not charged for first-use H2D migration.
    zero = np.float32(0.0)
    values = []
    for grads in pool:
        for grad in grads:
            grad.update(grad + zero)
            values.append(grad)
    jt.sync(values)


def sync_result(result, grads: list[jt.Var]) -> None:
    values = list(grads)
    if isinstance(result, jt.Var):
        values.append(result)
    jt.sync(values)


def benchmark(mode: str, count: int, total_elements: int, max_norm: float,
              warmup: int, repeats: int, seed: int) -> dict:
    warm = make_pool(count, total_elements, warmup, seed)
    timed = make_pool(count, total_elements, repeats, seed + 1)
    make_resident(warm)
    make_resident(timed)
    for grads in warm:
        result = run_mode(mode, grads, max_norm)
        sync_result(result, grads)
    build_s = 0.0
    sync_s = 0.0
    total_s = 0.0
    returned_norm = None
    for grads in timed:
        started = time.perf_counter()
        result = run_mode(mode, grads, max_norm)
        built = time.perf_counter()
        sync_result(result, grads)
        finished = time.perf_counter()
        build_s += built - started
        sync_s += finished - built
        total_s += finished - started
        if mode not in ("scale_device", "scale_host", "noop"):
            returned_norm = float(result.item())
    scale_expected = mode in ("old", "current", "flat_device",
                              "per_grad_device", "sum_device")
    theoretical_temp = 0
    if mode in ("old", "current", "flat_device", "norm_flat"):
        theoretical_temp = total_elements * 4
    elif mode in ("per_grad_device", "sum_device", "norm_current", "norm_sum"):
        theoretical_temp = count * 4
    return {
        "kind": "wall",
        "mode": mode,
        "count": count,
        "total_elements": total_elements,
        "elements_per_grad_min": total_elements // count,
        "max_norm": max_norm,
        "clip_expected": bool(scale_expected and max_norm < 1000.0),
        "warmup": warmup,
        "repeats": repeats,
        "total_ms": total_s * 1000.0 / repeats,
        # For old, build_ms includes the .item() execution and D2H sync inside
        # old_clip; for all device-only modes this is almost entirely graph build.
        "call_build_ms": build_s * 1000.0 / repeats,
        "final_sync_ms": sync_s * 1000.0 / repeats,
        "returned_norm": returned_norm,
        "theoretical_norm_temp_bytes": theoretical_temp,
    }


def source_summary(report) -> dict:
    if not report:
        return {"profile_rows": 0, "source_launch_sites": 0, "rows": []}
    header = report[0]
    rows = [dict(zip(header, row)) for row in report[1:]]
    launches = 0
    files = []
    compact = []
    for row in rows:
        path = pathlib.Path(row.get("FileName", ""))
        files.append(str(path))
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            launches += text.count("<<<")
        compact.append({key: row.get(key) for key in (
            "Name", "FileName", "Calls", "TotalTime", "AvgTime",
            "MinTime", "MaxTime", "Input", "Output", "Computation")
            if key in row})
    return {
        "profile_rows": len(rows),
        "source_launch_sites": launches,
        "avg_kernel_us_sum": sum(float(row["AvgTime"]) for row in rows) / 1000.0,
        "source_files": files,
        "rows": compact,
    }


def profile(mode: str, count: int, total_elements: int, max_norm: float,
            warmup: int, rerun: int, seed: int) -> dict:
    # Compile and execute an independent graph before enabling the profiler.
    first = make_pool(count, total_elements, 1, seed)[0]
    result = run_mode(mode, first, max_norm)
    sync_result(result, first)
    grads = make_pool(count, total_elements, 1, seed + 1)[0]
    with jt.profile_scope(warmup, rerun, profiler_hide_relay=1) as report:
        result = run_mode(mode, grads, max_norm)
        sync_result(result, grads)
    row = {
        "kind": "profile",
        "mode": mode,
        "count": count,
        "total_elements": total_elements,
        "max_norm": max_norm,
        "profiler_warmup": warmup,
        "profiler_rerun": rerun,
    }
    row.update(source_summary(report))
    return row


def correctness() -> list[dict]:
    original = [np.array([3.0, 4.0], dtype="float32"),
                np.array([0.0, -3.0], dtype="float32")]
    expected_norm = float(np.sqrt(34.0))
    expected_coef = 1.0 / (expected_norm + 1e-6)
    rows = []
    for mode in ("old", "current", "flat_device", "per_grad_device", "sum_device"):
        grads = [jt.array(value) for value in original]
        result = run_mode(mode, grads, 1.0)
        values = jt.fetch_sync([result] + grads)
        got = np.concatenate([np.asarray(value).reshape(-1)
                              for value in values[1:]])
        ref = np.concatenate(original) * expected_coef
        rows.append({
            "kind": "correctness",
            "mode": mode,
            "norm_abs_error": abs(float(values[0].reshape(-1)[0]) - expected_norm),
            "grads_max_abs_error": float(np.max(np.abs(got - ref))),
        })
    return rows


def parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def parse_csv_modes(value: str) -> list[str]:
    return [part for part in value.split(",") if part]


def write_row(path: str | None, row: dict) -> None:
    emit(row)
    if not path:
        return
    target = pathlib.Path(path)
    if not target.is_absolute():
        target = WORK_ROOT / target
    target = target.resolve()
    target.relative_to(WORK_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    modes = ("old", "current", "flat_device", "per_grad_device", "sum_device",
             "norm_current", "norm_flat", "norm_sum",
             "scale_device", "scale_host", "noop")
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("wall", "matrix", "profile", "correctness"),
                        required=True)
    parser.add_argument("--mode", choices=modes, default="current")
    parser.add_argument("--modes", default=",".join(modes))
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--counts", default="1,16,128,512")
    parser.add_argument("--total-elements", type=int, default=262144)
    parser.add_argument("--total-elements-list", default="262144,4194304")
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--max-norms", default="1,1000000000,inf")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--rerun", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--jsonl")
    args = parser.parse_args()
    jt.flags.use_cuda = 1
    write_row(args.jsonl, {
        "kind": "environment",
        "task": args.task,
        "jittor": jt.__version__,
        "cuda_archs": list(jt.flags.cuda_archs),
    })
    if args.task == "correctness":
        for row in correctness():
            write_row(args.jsonl, row)
    elif args.task == "wall":
        write_row(args.jsonl, benchmark(
            args.mode, args.count, args.total_elements, args.max_norm,
            args.warmup, args.repeats, args.seed))
    elif args.task == "profile":
        write_row(args.jsonl, profile(
            args.mode, args.count, args.total_elements, args.max_norm,
            args.warmup, args.rerun, args.seed))
    else:
        selected_modes = parse_csv_modes(args.modes)
        counts = parse_csv_ints(args.counts)
        totals = parse_csv_ints(args.total_elements_list)
        max_norms = [float(value) for value in args.max_norms.split(",")]
        clip_modes = {"old", "current", "flat_device", "per_grad_device", "sum_device"}
        for total in totals:
            for count in counts:
                for mode in selected_modes:
                    limits = max_norms if mode in clip_modes else [1.0]
                    for limit in limits:
                        write_row(args.jsonl, benchmark(
                            mode, count, total, limit, args.warmup,
                            args.repeats, args.seed + total + count))
    jt.sync_all(True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
