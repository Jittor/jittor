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


def attribute_launchers(events):
    """correlation id -> the innermost ``aten::`` operator that launched it.

    A kernel event carries a ``correlation`` that also appears on its
    ``cudaLaunchKernel`` runtime event; that runtime event sits inside the CPU
    operator stack of the launching thread.  Matching by name alone is not
    enough for caliber work: PyTorch reuses one templated kernel symbol across
    unrelated operators (``reduce_kernel<...sum_functor...>`` serves both
    convolution bias gradients and ordinary ``sum`` calls) and buries whole
    operators inside a fused one, so a kernel-name bucket does not correspond
    to any bucket on the other framework's side.
    """
    spans = defaultdict(list)
    for event in events:
        if event.get("cat") in ("cpu_op", "user_annotation") and "dur" in event:
            spans[event["tid"]].append(
                (event["ts"], event["ts"] + event["dur"], event["name"]))
    for thread in spans.values():
        thread.sort()

    attribution = {}
    for event in events:
        if event.get("cat") != "cuda_runtime":
            continue
        correlation = event.get("args", {}).get("correlation")
        if correlation is None:
            continue
        start = event["ts"]
        enclosing = [span for span in spans.get(event["tid"], ())
                     if span[0] <= start <= span[1] and span[2].startswith("aten::")]
        if enclosing:
            # outermost first: an autograd node such as
            # aten::convolution_backward calls aten::sum for the bias gradient,
            # and only the outer name says which Jittor kernel it corresponds to
            enclosing.sort(key=lambda s: -(s[1] - s[0]))
            names = []
            for span in enclosing:
                if span[2] not in names:
                    names.append(span[2])
            attribution[correlation] = ">".join(names)
    return attribution


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
    parser.add_argument("--attribute", action="store_true",
                        help="also record CPU activity and name, for every kernel, "
                             "the aten operator that launched it. PyTorch splits or "
                             "fuses operators differently from Jittor, so kernel-name "
                             "buckets alone cannot be compared; use this run for "
                             "identity and a plain run for the times")
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
    activities = [torch.profiler.ProfilerActivity.CUDA]
    if options.attribute:
        activities.append(torch.profiler.ProfilerActivity.CPU)
    with torch.profiler.profile(activities=activities) as profiler:
        for _ in range(options.steps):
            step()
        torch.cuda.synchronize()
    profiler.export_chrome_trace(trace)

    with open(trace) as handle:
        events = json.load(handle)["traceEvents"]
    launcher = attribute_launchers(events) if options.attribute else {}
    kernels = defaultdict(lambda: {"calls": 0, "us": 0.0,
                                   "aten": defaultdict(float)})
    for event in events:
        if event.get("cat") not in ("kernel", "Kernel"):
            continue
        entry = kernels[event["name"]]
        entry["calls"] += 1
        entry["us"] += float(event.get("dur", 0))
        if options.attribute:
            correlation = event.get("args", {}).get("correlation")
            entry["aten"][launcher.get(correlation, "?")] += \
                float(event.get("dur", 0))
    assert kernels, "no CUDA kernels in the trace"

    records = [{"name": name, "calls": entry["calls"] / options.steps,
                "step_us": entry["us"] / options.steps,
                "aten": {k: v / options.steps
                         for k, v in sorted(entry["aten"].items(),
                                            key=lambda kv: -kv[1])}}
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
