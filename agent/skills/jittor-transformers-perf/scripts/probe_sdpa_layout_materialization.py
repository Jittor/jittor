#!/usr/bin/env python3
"""Probe flash-attn input materialization strategies without changing Jittor."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np


from _paths import REPO_ROOT as ROOT, WORK_ROOT


def _reference(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    qf, kf, vf = q.astype("float32"), k.astype("float32"), v.astype("float32")
    scores = np.einsum("bhld,bhsd->bhls", qf, kf) / math.sqrt(q.shape[-1])
    scores -= scores.max(axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs /= probs.sum(axis=-1, keepdims=True)
    return np.einsum("bhls,bhsd->bhld", probs, vf)


def _to_numpy(value) -> np.ndarray:
    return np.asarray(value.float32().numpy(), dtype="float32")


def _append(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--paired-repeats", type=int, default=0)
    parser.add_argument(
        "--modes", default="current_clone,pre_bshd_direct,settled_no_clone,contiguous,no_clone")
    parser.add_argument(
        "--jsonl",
        default="results/sdpa_layout_gpu3_20260711.jsonl")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "python"))
    import jittor as torch

    sys.modules["torch"] = torch
    torch.flags.use_cuda = 1
    import flash_attn
    from jittor.torch_shim import flashattn_jittor

    shape = (4, 12, 128, 64)  # BHLD, the torch F.sdpa input layout.
    batch, heads, length, dim = shape
    scale = dim ** -0.5
    slots = []
    for index in range(args.repeats):
        rng = np.random.default_rng(20260711 + index)
        host = [rng.standard_normal(shape).astype("float16") for _ in range(3)]
        bhld = [torch.array(value) for value in host]
        bshd = [torch.array(value.transpose(0, 2, 1, 3).copy()) for value in host]
        slots.append({"host": host, "bhld": bhld, "bshd": bshd,
                      "reference": _reference(*host)})
    torch.sync_all(True)

    probe_dense = slots[0]["bhld"][0].permute(0, 2, 1, 3).reshape(
        (batch, length, heads, dim))
    contiguous_is_identity = probe_dense.contiguous() is probe_dense

    def invoke(mode: str, slot: dict):
        if mode == "pre_bshd_direct":
            q, k, v = slot["bshd"]
        else:
            inputs = slot["bhld"]
            if mode.startswith("lazy_packed_"):
                packed = torch.stack(inputs, dim=0)
                inputs = [packed[0], packed[1], packed[2]]
            q, k, v = [
                value.permute(0, 2, 1, 3).reshape((batch, length, heads, dim))
                for value in inputs
            ]
            if mode in ("current_clone", "lazy_packed_clone"):
                q, k, v = q.clone(), k.clone(), v.clone()
            elif mode == "contiguous":
                q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
            elif mode == "settled_no_clone":
                q.sync()
                k.sync()
                v.sync()
            elif mode not in ("no_clone", "lazy_packed_no_clone"):
                raise ValueError(mode)
        out = flash_attn.flash_attn_func(q, k, v, 0.0, scale, False)
        return out.permute(0, 2, 1, 3)

    output = pathlib.Path(args.jsonl)
    if not output.is_absolute():
        output = WORK_ROOT / output
    output = output.resolve()
    output.relative_to(WORK_ROOT)

    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    with torch.no_grad():
        for mode in modes:
            # Compile and settle every shape/layout before timing.
            for index in range(args.warmup):
                invoke(mode, slots[index % len(slots)])
                torch.sync_all(True)

            elapsed = []
            arrays = []
            for slot in slots:
                start = time.perf_counter()
                result = invoke(mode, slot)
                torch.sync_all(True)
                elapsed.append((time.perf_counter() - start) * 1000.0)
                arrays.append(_to_numpy(result))

            max_abs = max(
                float(np.max(np.abs(got - slot["reference"])))
                for got, slot in zip(arrays, slots))
            mean_abs = float(np.mean([
                np.mean(np.abs(got - slot["reference"]))
                for got, slot in zip(arrays, slots)
            ]))
            finite = all(bool(np.isfinite(value).all()) for value in arrays)

            # Queue repeated calls using one input and retain all outputs. This
            # specifically stresses transient Var metadata and external-output
            # lifetime rather than only the synchronized single-call case.
            repeated = [invoke(mode, slots[0]) for _ in range(args.repeats)]
            torch.sync_all(True)
            repeated_np = [_to_numpy(value) for value in repeated]
            repeat_drift = max(
                float(np.max(np.abs(value - repeated_np[0])))
                for value in repeated_np[1:]
            ) if len(repeated_np) > 1 else 0.0
            repeat_ref_max_abs = max(
                float(np.max(np.abs(value - slots[0]["reference"])))
                for value in repeated_np)

            row = {
                "backend": flashattn_jittor.backend_name(),
                "contiguous_is_identity": contiguous_is_identity,
                "dtype": "float16",
                "finite": finite,
                "latency_max_ms": max(elapsed),
                "latency_mean_ms": float(np.mean(elapsed)),
                "latency_median_ms": float(np.median(elapsed)),
                "latency_min_ms": min(elapsed),
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "mode": mode,
                "repeat_drift_max_abs": repeat_drift,
                "repeat_ref_max_abs": repeat_ref_max_abs,
                "repeats": args.repeats,
                "shape_bhld": list(shape),
            }
            print(json.dumps(row, sort_keys=True), flush=True)
            _append(output, row)

        if args.paired_repeats:
            paired_modes = ("current_clone", "no_clone")
            for mode in paired_modes:
                for index in range(4):
                    invoke(mode, slots[index % len(slots)])
                    torch.sync_all(True)
            timings = {mode: [] for mode in paired_modes}
            for index in range(args.paired_repeats):
                # Reverse order every round so allocator/cache state is balanced.
                order = paired_modes if index % 2 else paired_modes[::-1]
                slot = slots[index % len(slots)]
                for mode in order:
                    start = time.perf_counter()
                    value = invoke(mode, slot)
                    torch.sync_all(True)
                    timings[mode].append((time.perf_counter() - start) * 1000.0)
                    del value
            for mode in paired_modes:
                values = timings[mode]
                row = {
                    "benchmark": "paired_interleaved",
                    "contiguous_is_identity": contiguous_is_identity,
                    "latency_max_ms": max(values),
                    "latency_mean_ms": float(np.mean(values)),
                    "latency_median_ms": float(np.median(values)),
                    "latency_min_ms": min(values),
                    "mode": mode,
                    "repeats": args.paired_repeats,
                    "shape_bhld": list(shape),
                }
                print(json.dumps(row, sort_keys=True), flush=True)
                _append(output, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
