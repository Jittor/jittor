"""How far apart do two *torch* runs drift when nothing differs but rounding?

`compare_th.py` shows the mlp and the cnn moving apart from jittor over a long
training run while the transformer does not. This is the control that says
whether that means anything: it runs torch against torch, with the only
difference a relative 1e-7 perturbation of the initial weights -- the size of a
float32 rounding step. Both networks are pure ReLU, so a difference that small
eventually flips a ReLU sign, which is a discrete change that then amplifies;
the transformer has no such branch and stays put.

    python bench/xval/control_th.py DIR [--hf32] [--steps N]

Compare the numbers here with `compare_th.py`'s. When they match, the drift is
the model's, not the framework's.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch_npu  # noqa: F401

_BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH)
sys.path.insert(0, os.path.join(_BENCH, "xval"))

import bench_torch as B  # noqa: E402
from _canon import torch_parameters  # noqa: E402
from compare_th import load_inputs  # noqa: E402

DEV = "npu"


def run(name, record, epsilon, steps, seed):
    model, _, _, flat = B.build(name)
    params = torch_parameters(name, model)
    generator = np.random.RandomState(seed)
    with torch.no_grad():
        for key, parameter in params.items():
            stored = record["init/" + key].copy()
            if epsilon:
                noise = generator.standard_normal(stored.shape).astype(np.float32)
                stored = stored * (1.0 + epsilon * noise)
            parameter.copy_(torch.from_numpy(stored))
    x, y = load_inputs(record)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(flat(model(x)), y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return (np.array(losses),
            {k: p.detach().cpu().numpy().copy() for k, p in params.items()})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir")
    parser.add_argument("--models", default="mlp,cnn,transformer")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=1e-7)
    parser.add_argument("--hf32", action="store_true")
    args = parser.parse_args()
    torch.npu.matmul.allow_hf32 = bool(args.hf32)
    torch.npu.conv.allow_hf32 = bool(args.hf32)

    print("torch against torch, initial weights perturbed by rel %g, %d steps, hf32=%s"
          % (args.epsilon, args.steps, bool(args.hf32)))
    for name in args.models.split(","):
        record = np.load(os.path.join(args.dir, name + ".npz"))
        base_losses, base = run(name, record, 0.0, args.steps, 1)
        moved_losses, moved = run(name, record, args.epsilon, args.steps, 1)
        relative_loss = (np.abs(moved_losses - base_losses)
                         / np.maximum(np.abs(base_losses), 1e-8))
        absolute = relative = 0.0
        for key in base:
            difference = float(np.abs(moved[key] - base[key]).max())
            scale = float(np.abs(base[key]).max())
            absolute = max(absolute, difference)
            if scale > 0:
                relative = max(relative, difference / scale)
        print("%-12s loss %.6f / %.6f, rel max %.3e | parameters abs %.3e rel %.3e"
              % (name, base_losses[-1], moved_losses[-1], relative_loss.max(),
                 absolute, relative), flush=True)


if __name__ == "__main__":
    main()
