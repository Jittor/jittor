#!/usr/bin/env python3
"""Calibrate native FSDP fixtures with independent binary Torch FSDP2 on two GPUs.

Run with the independent Torch interpreter's torch.distributed.run, passing
--source-root and --output-dir. This uses the same five-step fixed fixture and
optimizers as native acceptance; it introduces no alternative sharding math or
numerical budget. Per-rank outputs are full tensors collected by real DTensor.
It complements, and does not replace, the dense independent reference.
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    from pathlib import Path
    import hashlib
    import json
    import os
    import sys
    import subprocess
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
    from torch.distributed.tensor import DTensor
    from torch.distributed.device_mesh import init_device_mesh

    assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
    assert "jittor" not in sys.modules
    root = Path(args.source_root).resolve()
    out = Path(args.output_dir).resolve()
    if out == root or root in out.parents:
        raise ValueError("Raw calibration artifacts must be outside the source checkout")
    source_commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    sys.path.insert(0, str(root / "tests"))
    from _helpers.native_fsdp import model, optimizer, samples

    rank = int(os.environ["RANK"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    mesh = init_device_mesh("cuda", (2,))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(2)
    out.mkdir(parents=True, exist_ok=True)

    def array(value):
        if isinstance(value, DTensor):
            value = value.full_tensor()
        return value.detach().float().cpu().numpy().copy()

    for kind in ("sgd", "adamw"):
        module = model(torch, native=False).cuda()
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, output_dtype=torch.float32
            ),
        )
        opt = optimizer(torch, module, kind)
        named = sorted(module.named_parameters())
        arrays = {f"initial/parameter/{name}": array(value) for name, value in named}
        for step in range(5):
            xn, yn = samples(step, rank)
            x = torch.from_numpy(xn).cuda().requires_grad_(True)
            y = torch.from_numpy(yn).cuda()
            prediction = module(x)
            loss = (prediction - y).square().mean()
            loss.backward()
            prefix = f"step/{step}/"
            arrays[prefix + "prediction"] = array(prediction)
            arrays[prefix + "loss"] = array(loss)
            arrays[prefix + "input_gradient"] = array(x.grad)
            for name, value in named:
                assert value.dtype == value.grad.dtype == torch.float32
                arrays[prefix + "gradient/" + name] = array(value.grad)
            opt.step()
            for name, value in named:
                arrays[prefix + "parameter/" + name] = array(value)
                for field, item in opt.state[value].items():
                    arrays[prefix + "state/" + name + "/" + field] = (
                        array(item) if isinstance(item, torch.Tensor) else np.asarray(item)
                    )
            opt.zero_grad()
        torch.cuda.synchronize()
        path = out / f"{kind}-rank{rank}.npz"
        np.savez(path, **arrays)
        metadata = {
            "runtime": "binary Torch FSDP2",
            "source_commit": source_commit,
            "version": torch.__version__,
            "binary": torch._C.__file__,
            "rank": rank,
            "world": 2,
            "local_device": torch.cuda.current_device(),
            "visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "param_dtype": "bfloat16",
            "reduce_dtype": "float32",
            "output_dtype": "float32",
            "tf32": False,
            "matmul_precision": "highest",
            "kind": kind,
            "steps": 5,
            "arrays": len(arrays),
            "calibration_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "fixture_sha256": hashlib.sha256(
                (root / "tests/_helpers/native_fsdp.py").read_bytes()
            ).hexdigest(),
            "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        print("TORCH_FSDP_CALIBRATION " + json.dumps(metadata, sort_keys=True), flush=True)
        del opt, module
    assert "jittor" not in sys.modules
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
