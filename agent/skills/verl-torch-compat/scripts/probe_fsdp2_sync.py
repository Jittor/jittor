"""Real two-rank FSDP2 shared-weight, gradient clipping and SGD probe.

Uses public Torch/FSDP2 and the upstream verl clipping helper, with independent
NumPy derivatives. No private implementation mutation or fabricated gradients.
This small linear fixture is not a multi-rank LLM or complete RL workflow.
Launch two processes with RANK/LOCAL_RANK/WORLD_SIZE and a fresh rendezvous.
Only scalar error summaries are emitted; tensors stay in the running process.
"""

import argparse
import contextlib
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("jittor", "torch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-cpu-state", action="store_true")
    args = parser.parse_args()
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    assert world == 2 and rank in (0, 1)
    if args.framework == "jittor":
        import jittor as jt
    else:
        jt = None
    import torch
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    guard = contextlib.nullcontext()
    if args.framework == "torch":
        import torch_npu  # noqa: F401
        assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
    else:
        from jittor._runtime.fallback import forbid_backend_fallbacks
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner
        assert namespace_owner(torch) is jt and activation_status(jt).active
        # Public frontend policy, not assignment of a fabricated library version.
        # Direct verl utility imports gate FSDP2 on the Torch API level; the
        # normal TrainingWorker import activates this policy transitively.
        torch.compat_report_torch_api_version(True)
        guard = forbid_backend_fallbacks()
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    initial = np.arange(8, dtype=np.float32).reshape(4, 2) / 16
    cotangent = np.arange(1, 5, dtype=np.float32)
    report = {"schema": 1, "scope": "two-rank analytical FSDP2 fixture, not LLM/RL workflow",
              "framework": args.framework, "rank": rank, "world_size": world,
              "torch_reported_version": torch.__version__,
              "native_jittor_version": jt.__version__ if jt is not None else None,
              "version_policy": "public_frontend_api_level" if jt is not None else "native_torch",
              "load_cpu_state": args.load_cpu_state, "status": "running", "cases": []}
    output = Path(str(args.output) + ".rank%d.json" % rank)
    output.parent.mkdir(parents=True, exist_ok=True)

    def flush():
        output.write_text(json.dumps(report, indent=2) + "\n")

    @contextlib.contextmanager
    def outcome():
        try:
            yield
        except Exception as error:
            report["status"] = "failed"
            report["error_type"] = type(error).__name__
            report["error"] = str(error)
            flush()
            raise

    def sync():
        jt.sync_all(True) if jt is not None else torch.npu.synchronize()

    def resident(value):
        local = value.to_local() if hasattr(value, "to_local") else value
        sync()
        if jt is not None:
            native = getattr(local, "_var", local)
            assert native.location() == "device", native.location()
            assert native.device_id == local_rank, native.device_id
        else:
            assert local.device.type == "npu" and local.device.index == local_rank, str(local.device)
        return local

    def snapshot(value):
        resident(value)
        with torch.no_grad():
            full = value.full_tensor() if hasattr(value, "full_tensor") else value
            resident(full)
            result = np.array(full.detach().cpu().numpy(), copy=True)
        assert np.isfinite(result).all()
        return result

    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
        return float(np.max(np.abs(actual - expected)))

    class SharedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.left = torch.nn.Linear(2, 4, bias=False)
            self.right = torch.nn.Linear(2, 4, bias=False)
            self.right.weight = self.left.weight

        def forward(self, x):
            return self.left(x) + self.right(x)

    flush()
    with outcome(), guard:
        before = jt.core.backend_fallback_count() if jt is not None else None
        report["stage"] = "init_process_group"
        flush()
        dist.init_process_group("hccl", rank=rank, world_size=world, timeout=timedelta(seconds=120))
        assert dist.get_world_size() == world and dist.get_rank() == rank
        assert str(dist.get_backend()) == "hccl"
        mesh = init_device_mesh("npu", (world,), mesh_dim_names=("fsdp",))
        # Resolve native frameworks first; append only shared downstream Python
        # dependencies, never prepend a different Torch installation.
        from transformers import Qwen3ForTokenClassification  # noqa: F401
        extra = os.environ.get("VERL_EXTRA_SITE_PACKAGES")
        if extra and extra not in sys.path:
            sys.path.append(extra)
        from verl.utils.fsdp_utils import fsdp2_clip_grad_norm_

        for shift in (0.0, 2.0):
            report["stage"] = "construct shift=%s" % shift
            flush()
            with torch.device("cpu"):
                model = SharedLinear()
                with torch.no_grad():
                    model.left.weight.copy_(torch.tensor(
                        initial + (10.0 + rank if args.load_cpu_state else 0.0)))
            assert model.left.weight is model.right.weight
            if not args.load_cpu_state:
                model = model.to(device)
            fully_shard(model, mesh=mesh, reshard_after_forward=True)
            assert model.left.weight is model.right.weight, "shared identity broken by fully_shard"
            if args.load_cpu_state:
                from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
                full_state = {"left.weight": torch.tensor(initial, device="cpu"),
                              "right.weight": torch.tensor(initial, device="cpu")} if rank == 0 else {}
                model.to(device)
                report["stage"] = "rank0 CPU full state load"
                flush()
                set_model_state_dict(model, full_state, options=StateDictOptions(
                    full_state_dict=True, broadcast_from_rank0=True))
            assert model.left.weight is model.right.weight
            assert len(list(model.parameters())) == 1, "shared storage must be optimized once"
            assert int(resident(model.left.weight).numel()) == 4, "must own half of the 8-element weight"
            if jt is not None:
                # Read-only diagnostics: check backing storage, not only published labels.
                st = model._fsdp_state
                assert st.true_fsdp_initialized and st.true_fsdp_world_size == 2
                assert len(st.true_fsdp_params) == 1
                resident(st.true_fsdp_flat_shard if st.true_fsdp_flat else st.true_fsdp_params[0].shard)
            init_error = close(snapshot(model.left.weight), initial)
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            x_np = np.array([[1.0, 2.0] if rank == 0 else [3.0, 4.0]], dtype=np.float32)
            if rank == 0:
                x_np += shift
            x = torch.tensor(x_np, device=device)
            c = torch.tensor(cotangent, device=device)
            expected_grad = 2 * np.outer(cotangent, np.array([2., 3.], dtype=np.float32) + shift / 2)
            expected_norm = float(np.linalg.norm(expected_grad))
            clipped = expected_grad * min(1.0, 1.0 / (expected_norm + 1e-6))
            expected_weight = initial.copy()
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                report["stage"] = "backward shift=%s step=%s" % (shift, step)
                flush()
                loss = (model(x) * c).sum()
                resident(loss)
                loss.backward()
                assert model.left.weight is model.right.weight
                gradient = model.left.weight.grad
                assert gradient is not None
                gradient_error = close(snapshot(gradient), expected_grad)
                report["stage"] = "clip shift=%s step=%s" % (shift, step)
                from verl.utils.device import get_device_id, get_device_name
                report["clip_placement"] = {
                    "verl_device": get_device_name(), "verl_device_id": get_device_id(),
                    "npu_current_device": torch.npu.current_device(),
                    "gradient_device": str(gradient.device),
                }
                flush()
                norm = fsdp2_clip_grad_norm_(model.parameters(), max_norm=1.0)
                norm_error = close(snapshot(norm), np.array(expected_norm))
                clip_error = close(snapshot(model.left.weight.grad), clipped)
                optimizer.step()
                assert model.left.weight is model.right.weight, "shared identity broken by optimizer"
                expected_weight -= 0.01 * clipped
                update_error = close(snapshot(model.left.weight), expected_weight)
                report["cases"].append({"rank0_input_shift": shift, "step": step,
                    "initial_max_abs": init_error, "gradient_max_abs": gradient_error,
                    "norm": float(snapshot(norm).reshape(())), "norm_max_abs": norm_error,
                    "clipped_gradient_max_abs": clip_error, "update_max_abs": update_error,
                    "shared_identity_preserved": True})
                flush()
        assert report["cases"][2]["norm"] > report["cases"][0]["norm"] + 1
        sync()
        dist.barrier()
        report["fallback_delta"] = jt.core.backend_fallback_count() - before if jt is not None else None
        if jt is not None:
            assert report["fallback_delta"] == 0
        report["status"] = "pass"
        report["stage"] = "complete"
        flush()
        print("FSDP2_SYNC_PASS " + json.dumps(report), flush=True)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
