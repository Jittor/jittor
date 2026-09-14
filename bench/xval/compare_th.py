"""Replay `dump_jt.py`'s recording in torch and report where the two diverge.

The torch models are loaded with jittor's initial weights and fed jittor's
inputs, so any difference that shows up is the frameworks' arithmetic and not
the initialisation or the data.

    python bench/xval/compare_th.py DIR [--hf32] [--steps N]

Read the numbers this way: the step-0 loss and gradients isolate one forward
and one backward, and are the tight check. A long trajectory is not -- see
`control_th.py` before treating drift over many steps as a defect.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401  (registers the npu device)

_BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH)
sys.path.insert(0, os.path.join(_BENCH, "xval"))

import bench_torch as B  # noqa: E402
from _canon import check_complete, torch_parameters  # noqa: E402

DEV = "npu"


def load_inputs(record):
    x = record["x"]
    x = torch.from_numpy(x.astype(np.int64) if x.dtype.kind == "i" else x)
    y = torch.from_numpy(record["y"].astype(np.int64))
    return x.to(DEV), y.to(DEV)


def build_loaded(name, record):
    """A torch model carrying jittor's initial weights."""
    model, _, _, flat = B.build(name)
    params = torch_parameters(name, model)
    check_complete(name, params, model.parameters())
    with torch.no_grad():
        for key, parameter in params.items():
            stored = record["init/" + key]
            if tuple(stored.shape) != tuple(parameter.shape):
                raise SystemExit("%s: %s is %s here and %s there"
                                 % (name, key, tuple(parameter.shape), stored.shape))
            parameter.copy_(torch.from_numpy(stored))
    return model, params, flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--models", default="mlp,cnn,transformer")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hf32", action="store_true")
    args = parser.parse_args()
    torch.npu.matmul.allow_hf32 = bool(args.hf32)
    torch.npu.conv.allow_hf32 = bool(args.hf32)

    print("torch %s / torch_npu %s, hf32=%s, %d steps, weights and inputs from jittor"
          % (torch.__version__, torch_npu.__version__, bool(args.hf32), args.steps))
    for name in args.models.split(","):
        record = np.load(os.path.join(args.dir, name + ".npz"))
        x, y = load_inputs(record)

        # --- step 0: one forward, one backward -------------------------------
        model, params, flat = build_loaded(name, record)
        loss0 = torch.nn.functional.cross_entropy(flat(model(x)), y)
        loss0.backward()
        worst_grad, worst_key = 0.0, None
        for key, parameter in params.items():
            here = parameter.grad.detach().cpu().numpy()
            there = record["grad/" + key]
            scale = max(float(np.abs(there).max()), 1e-30)
            relative = float(np.abs(here - there).max() / scale)
            if relative > worst_grad:
                worst_grad, worst_key = relative, key
        jt_loss0 = float(record["step0_loss"])
        th_loss0 = float(loss0.item())

        # --- the same training run -------------------------------------------
        model, params, flat = build_loaded(name, record)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        losses = []
        for _ in range(args.steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(flat(model(x)), y)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        jt_losses = record["losses"][:args.steps]
        th_losses = np.array(losses)
        relative_loss = np.abs(th_losses - jt_losses) / np.maximum(np.abs(jt_losses), 1e-8)
        absolute = relative = 0.0
        for key, parameter in params.items():
            here = parameter.detach().cpu().numpy()
            there = record["final/" + key]
            difference = float(np.abs(here - there).max())
            scale = float(np.abs(there).max())
            absolute = max(absolute, difference)
            if scale > 0:
                relative = max(relative, difference / scale)

        print("%-12s step-0 loss jt %.7f / th %.7f (rel %.2e), worst gradient rel %.3e at %s"
              % (name, jt_loss0, th_loss0, abs(th_loss0 - jt_loss0) / abs(jt_loss0),
                 worst_grad, worst_key))
        print("%-12s %d-step loss rel max %.3e, final parameters abs %.3e rel %.3e"
              % ("", args.steps, relative_loss.max(), absolute, relative), flush=True)


if __name__ == "__main__":
    main()
