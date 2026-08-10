#!/usr/bin/env python3
"""Benchmark torch-compatible gradient management inside real optimizer steps."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

import numpy as np


from _paths import REPO_ROOT as ROOT, WORK_ROOT


def _stats(values):
    return {
        "mean_ms": float(statistics.fmean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def _append(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", default="128,512")
    parser.add_argument("--optimizers", default="sgd,adamw")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--total-elements", type=int, default=262144)
    parser.add_argument(
        "--jsonl",
        default="results/optimizer_step_gpu3_20260711.jsonl")
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT / "python"))
    import jittor as torch

    sys.modules["torch"] = torch
    torch.flags.use_cuda = 1
    from jittor.torch_compat import _GradScaler, _clip_grad_norm_device

    counts = [int(value) for value in args.counts.split(",")]
    optimizer_names = [value.strip() for value in args.optimizers.split(",")]
    output = pathlib.Path(args.jsonl)
    if not output.is_absolute():
        output = WORK_ROOT / output
    output = output.resolve()
    output.relative_to(WORK_ROOT)

    class State:
        def __init__(self, optimizer_name: str, count: int):
            size = max(1, args.total_elements // count)
            self.params = [
                torch.array(np.full(size, 0.1 + index * 1e-7, dtype="float32"))
                for index in range(count)
            ]
            for param in self.params:
                param.start_grad()
            if optimizer_name == "sgd":
                self.optimizer = torch.optim.SGD(
                    self.params, lr=1e-3, momentum=0.9, weight_decay=0.01)
            elif optimizer_name == "adamw":
                self.optimizer = torch.optim.AdamW(
                    self.params, lr=1e-3, weight_decay=0.01)
            else:
                raise ValueError(optimizer_name)
            self.grads = [torch.ones(param.shape, dtype="float32").stop_grad()
                          for param in self.params]
            self.sources = [torch.ones(param.shape, dtype="float32").stop_grad()
                            for param in self.params]
            self.optimizer.param_groups[0]["grads"] = self.grads
            self.optimizer._build_grad_map()
            self.scaler = _GradScaler(init_scale=2.0, growth_interval=1000000)
            torch.sync_all(True)

        def reset_grads(self, scale: float) -> None:
            scalar = np.float32(scale)
            for grad, source in zip(self.grads, self.sources):
                grad.update(source * scalar)
            self.optimizer.param_groups[0]["grads"] = self.grads
            self.optimizer._build_grad_map()
            # post_step()/zero_grad() clears torch-facing attributes, but the
            # optimizer and management helpers consume the pg gradient Vars.
            self.scaler._unscaled = False
            self.scaler._found_inf = False
            torch.sync_all(True)

        def manage(self, mode: str) -> None:
            if mode == "clip":
                _clip_grad_norm_device(self.grads, 1.0, 2.0)
            elif mode == "scaler":
                self.scaler.unscale_(self.optimizer)
            elif mode != "none":
                raise ValueError(mode)

        def step_after_manage(self, mode: str) -> None:
            if mode == "scaler":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()

        def integrated_step(self, mode: str) -> None:
            if mode == "scaler":
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.manage(mode)
                self.optimizer.step()

    def run_integrated(state: State, mode: str, iterations: int):
        values = []
        grad_scale = 2.0 if mode == "scaler" else 1.0
        for _ in range(iterations):
            state.reset_grads(grad_scale)
            start = time.perf_counter()
            state.integrated_step(mode)
            torch.sync_all(True)
            values.append((time.perf_counter() - start) * 1000.0)
        return values

    def run_segmented(state: State, mode: str, iterations: int):
        management, step = [], []
        grad_scale = 2.0 if mode == "scaler" else 1.0
        for _ in range(iterations):
            state.reset_grads(grad_scale)
            start = time.perf_counter()
            state.manage(mode)
            torch.sync_all(True)
            management.append((time.perf_counter() - start) * 1000.0)
            start = time.perf_counter()
            state.step_after_manage(mode)
            torch.sync_all(True)
            step.append((time.perf_counter() - start) * 1000.0)
        return management, step

    for optimizer_name in optimizer_names:
        for count in counts:
            for mode in ("none", "clip", "scaler"):
                integrated_state = State(optimizer_name, count)
                run_integrated(integrated_state, mode, args.warmup)
                integrated = run_integrated(integrated_state, mode, args.repeats)

                if mode == "none":
                    management = [0.0] * args.repeats
                    step = list(integrated)
                else:
                    segmented_state = State(optimizer_name, count)
                    run_segmented(segmented_state, mode, args.warmup)
                    management, step = run_segmented(
                        segmented_state, mode, args.repeats)

                management_mean = statistics.fmean(management)
                step_mean = statistics.fmean(step)
                segmented_total = management_mean + step_mean
                row = {
                    "count": count,
                    "dtype": "float32",
                    "integrated": _stats(integrated),
                    "management": _stats(management),
                    "management_share_segmented": (
                        management_mean / segmented_total if segmented_total else 0.0),
                    "mode": mode,
                    "optimizer": optimizer_name,
                    "repeats": args.repeats,
                    "segmented_total_mean_ms": segmented_total,
                    "step_after_management": _stats(step),
                    "total_elements": args.total_elements,
                    "warmup": args.warmup,
                }
                checksum = float(sum(
                    value.float32().sum().item()
                    for value in integrated_state.params[:min(2, count)]))
                row["finite"] = bool(np.isfinite(checksum))
                row["sample_checksum"] = checksum
                print(json.dumps(row, sort_keys=True), flush=True)
                _append(output, row)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
