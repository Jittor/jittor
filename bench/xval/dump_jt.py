"""Record what jittor computes for the shared benchmark models.

Writes one ``<model>.npz`` per model holding the initial weights, the inputs,
the step-0 loss and gradients, the loss trajectory of a short training run and
the final parameters. ``compare_th.py`` replays all of it in torch.

    python bench/xval/dump_jt.py OUT_DIR [--hf32] [--steps N]
"""
import argparse
import os
import sys

import numpy as np

_BENCH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BENCH)
sys.path.insert(0, os.path.join(_BENCH, "xval"))

import jittor as jt  # noqa: E402  (bench_jittor asserts on the ACL build first)

import bench_jittor as B  # noqa: E402
from _canon import check_complete, jittor_parameters  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--models", default="mlp,cnn,transformer")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hf32", action="store_true",
                        help="allow HF32 cube math, matching the timing runs")
    args = parser.parse_args()
    jt.acl_allow_hf32 = bool(args.hf32)
    os.makedirs(args.out_dir, exist_ok=True)

    for name in args.models.split(","):
        model, x, y, flat = B.build(name)
        params = jittor_parameters(name, model)
        check_complete(name, params, model.parameters())
        keys = list(params)
        initial = {key: params[key].numpy().copy() for key in keys}

        # Step 0 in isolation: the forward and the backward, before any
        # optimizer state exists to carry a difference forward.
        loss0 = jt.nn.cross_entropy_loss(flat(model(x)), y)
        grads = jt.grad(loss0, [params[key] for key in keys])
        step0 = {key: grad.numpy().copy() for key, grad in zip(keys, grads)}
        step0_loss = float(loss0.numpy())

        optimizer = jt.nn.SGD(model.parameters(), lr=0.01, momentum=0.9)
        losses = []
        for _ in range(args.steps):
            loss = jt.nn.cross_entropy_loss(flat(model(x)), y)
            optimizer.step(loss)
            losses.append(float(loss.numpy()))

        np.savez(os.path.join(args.out_dir, name + ".npz"),
                 x=x.numpy(), y=y.numpy(),
                 step0_loss=np.array(step0_loss),
                 losses=np.array(losses),
                 **{"init/" + k: v for k, v in initial.items()},
                 **{"grad/" + k: v for k, v in step0.items()},
                 **{"final/" + k: params[k].numpy().copy() for k in keys})
        print("%-12s step-0 loss %.7f | %d-step loss %.6f -> %.6f"
              % (name, step0_loss, args.steps, losses[0], losses[-1]), flush=True)


if __name__ == "__main__":
    main()
