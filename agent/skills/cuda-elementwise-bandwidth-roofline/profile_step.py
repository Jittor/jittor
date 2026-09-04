"""Profile one training step of a speed-gate case under Jittor.

Two modes, because neither alone is enough:

``--mode profiler`` runs the step inside ``jt.profile_scope`` and dumps the
per-operator report.  That report is the only place the *identity* of a fused
kernel (which meta-operators went into it) and its external byte traffic are
available.  It measures each operator in isolation.

``--mode nsys`` runs steady-state steps between ``cudaProfilerStart/Stop`` and
does no measuring itself; launch it under
``nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop`` to
get what a real, pipelined step costs.  ``classify.py`` joins the two on the
jit-key hash.

The model, input shapes and loss come from ``_ecosystem_speed`` so the numbers
describe the workload the speed gate measures.
"""

import argparse
import ctypes
import json
import os
import sys
import time

import numpy as np

REPO = os.environ.get("JITTOR_REFACTOR_REPO") or os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tests", "compat", "torch"))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["JITTOR_TORCH_SHIM"] = "1"

import jittor as jt  # noqa: E402
import torch  # noqa: E402  (resolves to the Jittor shim)

import _ecosystem_speed  # noqa: E402


def primary(result):
    for attribute in ("logits", "sample", "last_hidden_state"):
        value = getattr(result, attribute, None)
        if value is not None:
            return value
    return result[0] if isinstance(result, (tuple, list)) else result


def build(case, seed):
    builder, _ = _ecosystem_speed.CASES[case]
    model, spec = builder(torch)
    model.eval()
    for parameter in model.parameters():
        parameter.start_grad()
    generator = np.random.RandomState(seed)
    inputs = {}
    for name, (dtype, shape, high) in spec.items():
        if dtype == "int64":
            inputs[name] = torch.from_numpy(
                generator.randint(0, high, size=shape).astype("int64"))
        else:
            tensor = torch.from_numpy(generator.randn(*shape).astype("float32"))
            tensor.requires_grad_(True)
            inputs[name] = tensor
    return model, inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="large_diffusers_unet2d")
    parser.add_argument("--mode", choices=("profiler", "nsys"), default="profiler")
    parser.add_argument("--rerun", type=int, default=31)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=10, help="--mode nsys only")
    parser.add_argument("--out", default="", help="--mode profiler only")
    parser.add_argument("--tag", default="")
    parser.add_argument("--flag", action="append", default=[],
                        help="name=int jt.flags override applied before the step")
    options = parser.parse_args()

    jt.flags.use_cuda = 1
    # The same precision configuration the speed gate applies to both runtimes.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    for entry in options.flag:
        name, _, value = entry.partition("=")
        setattr(jt.flags, name, int(value))

    model, inputs = build(options.case, 11)
    parameters = list(model.parameters())
    weights = np.random.RandomState(12)
    loss_weights = [None]

    def step():
        output = primary(model(**inputs))
        if loss_weights[0] is None:
            loss_weights[0] = torch.from_numpy(
                weights.randn(*tuple(output.shape)).astype("float32"))
        loss = (output * loss_weights[0]).sum()
        # The shim's Tensor.backward() aborts on this graph (see the board
        # entry on backward liveness); jt.grad submits the same backward
        # operators without going through it.
        grads = jt.grad(loss, parameters)
        jt.sync([loss] + list(grads), device_sync=True)
        return float(loss.detach().numpy().reshape(-1)[0]), grads

    reference, grads = step()
    checksum = float(sum(float(g.sum().numpy().reshape(-1)[0]) for g in grads))
    del grads
    for _ in range(3):
        step()
    jt.sync_all(True)

    if options.mode == "nsys":
        runtime = ctypes.CDLL("libcudart.so")
        assert runtime.cudaProfilerStart() == 0
        for _ in range(options.steps):
            step()
        jt.sync_all(True)
        runtime.cudaProfilerStop()
        print("steps %d loss %.6f grad_checksum %.6f"
              % (options.steps, reference, checksum))
        return

    # Wall time of the same step outside the profiler. The profiler serialises
    # every operator, so this is the sanity check that its per-kernel sum is a
    # plausible share of a real step rather than an artefact of serialisation.
    walls = []
    for _ in range(5):
        started = time.perf_counter()
        step()
        walls.append(time.perf_counter() - started)
    jt.sync_all(True)

    with jt.profile_scope(warmup=options.warmup, rerun=options.rerun) as rep:
        step()
        jt.sync_all(True)

    payload = {
        "case": options.case,
        "tag": options.tag,
        "flags": options.flag,
        "loss": reference,
        "grad_checksum": checksum,
        "rerun": options.rerun,
        "wall_s": min(walls),
        "header": rep[0],
        "records": rep[1:],
    }
    if options.out:
        with open(options.out, "w") as handle:
            json.dump(payload, handle)
    print("records %d loss %.6f grad_checksum %.6f wall_ms %.2f"
          % (len(rep) - 1, reference, checksum, min(walls) * 1e3))


if __name__ == "__main__":
    main()
