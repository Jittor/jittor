#!/usr/bin/env python3
"""Benchmark torch-compat gradient clipping and AMP overflow synchronization."""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

import jittor as jt
from jittor.compat.torch import _GradScaler, _clip_grad_norm_device


from _paths import WORK_ROOT


class _FakeOptimizer:
    def __init__(self, grads):
        self.param_groups = [{"grads": grads}]

    def step(self):
        return None


def _old_clip(grads, max_norm: float):
    total = jt.sqrt(jt.concat([
        g.cast("float32").sqr().reshape((-1,)) for g in grads
    ]).sum())
    coef = max_norm / (float(total.item()) + 1e-6)
    if coef < 1.0:
        for grad in grads:
            grad.update(grad * coef)
    return total


def _old_unscale(grads, scale: float):
    inv = 1.0 / scale
    found_inf = False
    for grad in grads:
        grad.update(grad * inv)
        value = float(grad.abs().max().item()) if grad.numel() else 0.0
        found_inf = found_inf or not np.isfinite(value)
    return found_inf


def _flat_unscale(grads, scale: float):
    inv = np.float32(1.0 / scale)
    parts = []
    for grad in grads:
        unscaled = grad * inv
        grad.update(unscaled)
        parts.append(unscaled.reshape((-1,)))
    flat = jt.concat(parts)
    return not bool(jt.isfinite(flat).all().item())


def _pool(count: int, total_elements: int, slots: int, seed: int):
    rng = np.random.default_rng(seed)
    size = max(1, total_elements // count)
    return [[jt.array(rng.standard_normal(size).astype("float32"))
             for _ in range(count)] for _ in range(slots)]


def _run_clip(mode: str, pool, max_norm: float) -> float:
    start = time.perf_counter()
    for grads in pool:
        if mode == "old":
            _old_clip(grads, max_norm)
        else:
            _clip_grad_norm_device(grads, max_norm, 2.0)
        jt.sync_all(True)
    return (time.perf_counter() - start) * 1000.0 / len(pool)


def _run_scaler(mode: str, pool) -> float:
    start = time.perf_counter()
    for grads in pool:
        if mode == "old":
            _old_unscale(grads, 2.0)
        elif mode == "flat":
            _flat_unscale(grads, 2.0)
        else:
            scaler = _GradScaler(init_scale=2.0)
            scaler.unscale_(_FakeOptimizer(grads))
        jt.sync_all(True)
    return (time.perf_counter() - start) * 1000.0 / len(pool)


def _output(path: str, row: dict) -> None:
    target = pathlib.Path(path)
    if not target.is_absolute():
        target = WORK_ROOT / target
    target = target.resolve()
    target.relative_to(WORK_ROOT)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=("clip", "scaler"), required=True)
    parser.add_argument("--mode", choices=("old", "new", "flat"), required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--total-elements", type=int, default=1 << 20)
    parser.add_argument("--max-norm", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--jsonl", default="results/grad_management.jsonl")
    args = parser.parse_args()

    jt.flags.use_cuda = 1
    warm = _pool(args.count, args.total_elements, args.warmup, 20260710)
    timed = _pool(args.count, args.total_elements, args.repeats, 20260711)
    runner = _run_clip if args.operation == "clip" else _run_scaler
    if args.operation == "clip":
        runner(args.mode, warm, args.max_norm)
        latency = runner(args.mode, timed, args.max_norm)
    else:
        runner(args.mode, warm)
        latency = runner(args.mode, timed)

    row = {
        "operation": args.operation,
        "mode": args.mode,
        "count": args.count,
        "total_elements": args.total_elements,
        "max_norm": args.max_norm,
        "repeats": args.repeats,
        "latency_ms": latency,
    }
    print(json.dumps(row, sort_keys=True), flush=True)
    _output(args.jsonl, row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
