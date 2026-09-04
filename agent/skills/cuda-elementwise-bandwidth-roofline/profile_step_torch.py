"""Kernel-level profile of the same step under real PyTorch.

Must run in an interpreter that owns a genuine ``torch`` (the development
environment's ``torch`` is the Jittor shim).  The case definition is imported
from the speed gate so both sides describe the same workload.
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.environ.get("JITTOR_REFACTOR_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tests", "compat", "torch"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.pop("JITTOR_TORCH_SHIM", None)

import torch  # noqa: E402
import _ecosystem_speed  # noqa: E402


def primary(result):
    for attribute in ("logits", "sample", "last_hidden_state"):
        value = getattr(result, attribute, None)
        if value is not None:
            return value
    return result[0] if isinstance(result, (tuple, list)) else result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="large_diffusers_unet2d")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", default="")
    options = parser.parse_args()

    assert hasattr(torch, "_C"), "this interpreter does not own a real torch"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")

    builder, _ = _ecosystem_speed.CASES[options.case]
    model, spec = builder(torch)
    model.eval().cuda()

    generator = np.random.RandomState(11)
    inputs = {}
    for name, (dtype, shape, high) in spec.items():
        if dtype == "int64":
            inputs[name] = torch.from_numpy(
                generator.randint(0, high, size=shape).astype("int64")).cuda()
        else:
            tensor = torch.from_numpy(
                generator.randn(*shape).astype("float32")).cuda()
            tensor.requires_grad_(True)
            inputs[name] = tensor

    with torch.no_grad():
        shape = tuple(primary(model(**inputs)).shape)
    loss_weights = torch.from_numpy(
        np.random.RandomState(12).randn(*shape).astype("float32")).cuda()

    def step():
        output = primary(model(**inputs))
        loss = (output * loss_weights).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        return loss

    for _ in range(3):
        step()
    torch.cuda.synchronize()

    trace = options.trace or os.path.join(
        os.environ.get("TMPDIR", "/tmp"), "torch_step_trace.json")
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(options.steps):
            step()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(trace)

    with open(trace) as handle:
        events = json.load(handle)["traceEvents"]
    kernels = defaultdict(lambda: {"calls": 0, "us": 0.0})
    for event in events:
        if event.get("cat") not in ("kernel", "Kernel"):
            continue
        entry = kernels[event["name"]]
        entry["calls"] += 1
        entry["us"] += float(event.get("dur", 0))
    assert kernels, "no CUDA kernels in the trace"

    records = [{"name": name, "calls": entry["calls"] / options.steps,
                "step_us": entry["us"] / options.steps}
               for name, entry in kernels.items()]
    records.sort(key=lambda r: -r["step_us"])
    with open(options.out, "w") as handle:
        json.dump({"case": options.case, "steps": options.steps,
                   "torch": torch.__version__, "records": records},
                  handle, indent=1)
    print("kernels %d  step_us %.1f"
          % (len(records), sum(r["step_us"] for r in records)))


if __name__ == "__main__":
    main()
