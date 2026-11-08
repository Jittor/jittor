#!/usr/bin/env python3
"""Generate an independent binary Torch reference for native FSDP fixtures.

This process never imports Jittor or its compatibility frontend. The shared
fixture contains only NumPy setup and real framework calls; no sharding math.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--kind", choices=("sgd", "adamw"), required=True)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context"), (
        "binary Torch is required"
    )
    root = args.source_root.resolve()
    sys.path.insert(0, str(root / "tests"))
    from _helpers.native_fsdp import model, optimizer, samples

    assert "jittor" not in sys.modules and not any(
        n.startswith("jittor.compat") for n in sys.modules
    )
    if args.device == "cuda":
        assert torch.cuda.is_available(), "real CUDA is required"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    module = model(torch, native=False).to(args.device)
    opt = optimizer(torch, module, args.kind)
    arrays = {
        f"initial/parameter/{name}": value.detach().cpu().numpy().copy()
        for name, value in sorted(module.named_parameters())
    }
    for step in range(5):
        inputs = []
        losses = []
        predictions = []
        for rank in range(2):
            x, y = samples(step, rank)
            x = torch.from_numpy(x).to(args.device).requires_grad_(True)
            y = torch.from_numpy(y).to(args.device)
            with torch.autocast(
                device_type=args.device, dtype=torch.bfloat16, enabled=args.dtype == "bfloat16"
            ):
                prediction = module(x)
            loss = (prediction - y).square().mean()
            inputs.append(x)
            predictions.append(prediction)
            losses.append(loss)
        for rank in range(2):
            prefix = f"step/{step}/rank/{rank}/"
            arrays[prefix + "input_gradient"] = (
                torch.autograd.grad(losses[rank], inputs[rank], retain_graph=True)[0]
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
            arrays[prefix + "prediction"] = predictions[rank].detach().cpu().numpy().copy()
            arrays[prefix + "loss"] = losses[rank].detach().cpu().numpy().copy()
        (sum(losses) / 2).backward()
        for name, value in sorted(module.named_parameters()):
            arrays[f"step/{step}/gradient/{name}"] = value.grad.detach().cpu().numpy().copy()
        opt.step()
        for name, value in sorted(module.named_parameters()):
            arrays[f"step/{step}/parameter/{name}"] = value.detach().cpu().numpy().copy()
            for field, state in opt.state[value].items():
                arrays[f"step/{step}/state/{name}/{field}"] = (
                    state.detach().cpu().numpy().copy()
                    if isinstance(state, torch.Tensor)
                    else np.asarray(state)
                )
        opt.zero_grad()
    assert "jittor" not in sys.modules
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"oracle-{args.kind}-{args.dtype}-{args.device}.npz"
    np.savez(output, **arrays)
    metadata = {
        "runtime": "torch",
        "binary": torch._C.__file__,
        "version": torch.__version__,
        "device": args.device,
        "kind": args.kind,
        "dtype": args.dtype,
        "steps": 5,
        "world_reference": 2,
        "dropout": 0,
        "tf32": False,
        "matmul_precision": "highest",
        "arrays": len(arrays),
        "npz_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "fixture_sha256": hashlib.sha256(
            (root / "tests/_helpers/native_fsdp.py").read_bytes()
        ).hexdigest(),
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
