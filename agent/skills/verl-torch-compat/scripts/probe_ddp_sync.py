"""Two-rank CUDA/NCCL or NPU/HCCL DDP with an independent linear-model oracle.

Use the same isolated interpreters and rank environment as run_npu_benchmark.sh.
This is a correctness probe, not a performance measurement. Only diagnostics
and NumPy reference arithmetic run on the host; every tested tensor operation,
backward, optimizer update, and collective must execute on the accelerator.
"""

import argparse
import contextlib
import json
import os
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jittor"), required=True)
    parser.add_argument("--device", choices=("npu", "cuda"), default="npu")
    parser.add_argument("--output", required=True, help="JSON prefix; .rankN.json is appended")
    return parser.parse_args()


def rank_environment(device):
    prefix = "JT_NCCL_" if device == "cuda" else "JT_HCCL_"
    rank = int(os.environ.get("RANK", os.environ.get(prefix + "RANK", "0")))
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get(prefix + "LOCAL_RANK", rank)))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get(prefix + "WORLD_SIZE", "1")))
    return rank, local_rank, world_size


def distributed_backend(device):
    return "nccl" if device == "cuda" else "hccl"


def initialize_runtime(args, guard, rank, local_rank, world_size):
    jt = None
    if args.framework == "torch":
        assert hasattr(torch, "_C") and not hasattr(torch, "_torch_compat_install_context")
        if args.device == "npu":
            import torch_npu  # noqa: F401
            torch.npu.set_device(local_rank)
        else:
            assert torch.cuda.is_available(), "official PyTorch CUDA unavailable"
            torch.cuda.set_device(local_rank)
            assert torch.cuda.current_device() == local_rank, "CUDA device selection failed"
        device = torch.device(args.device, local_rank)

        def synchronize():
            getattr(torch, args.device).synchronize()
    else:
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        from jittor.compat.shim.runtime import activation_status
        from jittor.compat.torch.publication import namespace_owner

        assert namespace_owner(torch) is jt and activation_status(jt).active
        if args.device == "npu":
            assert getattr(jt.compiler, "has_acl", 0), "Jittor ACL unavailable"
            guard.enter_context(jt.runtime.scope(use_acl=1, use_cuda=1))
            torch.npu.set_device(local_rank)
        else:
            assert getattr(jt.compiler, "has_cuda", False), "Jittor CUDA unavailable"
            assert not any(getattr(jt.compiler, name, False) for name in
                           ("has_acl", "has_rocm", "has_corex")), "build is not native CUDA"
            # Other backend flags alias use_cuda; do not clear them here.
            guard.enter_context(jt.runtime.scope(use_cuda=1))
            torch.cuda.set_device(local_rank)
            assert torch.cuda.current_device() == local_rank, "CUDA device selection failed"
        assert jt.world_size == world_size and jt.rank == rank
        guard.enter_context(jt.runtime.scope(backend_fallback="error"))
        guard.enter_context(forbid_backend_fallbacks())
        device = torch.device(args.device, local_rank)

        def synchronize():
            jt.sync_all(True)

    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("highest")
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.get_float32_matmul_precision() == "highest"
    return jt, device, synchronize


def check_location(value, jt, device, local_rank):
    """Check actual placement before any native Var host snapshot can migrate it."""
    if jt is None:
        assert value.device.type == device, str(value.device)
        assert value.device.index == local_rank, "tensor on wrong accelerator device"
        return str(value.device)
    assert bool(jt.flags.use_cuda), "accelerator dispatch changed"
    native = getattr(value, "_var", value)
    result = native.location()
    assert result == "device", "tensor left requested backend: " + result
    assert native.device_id == local_rank, "tensor on wrong accelerator device"
    if device == "cuda":
        assert torch.cuda.current_device() == local_rank, "CUDA device selection changed"
    return result


def main():
    args = parse_args()
    rank, local_rank, world_size = rank_environment(args.device)
    assert world_size == 2 and rank in (0, 1), "this independent analytic fixture requires two ranks"
    guard = contextlib.ExitStack()
    jt, device, synchronize = initialize_runtime(args, guard, rank, local_rank, world_size)
    fallback_before = jt.core.backend_fallback_count() if jt is not None else None
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    def location(value):
        return check_location(value, jt, args.device, local_rank)

    def snapshot(value):
        synchronize()
        location(value)
        return np.array(value.detach().cpu().numpy(), copy=True)

    def tensor(value):
        result = torch.tensor(value, dtype=torch.float32)
        return result.to(device)

    initial = [np.array([[0.25, -0.5]], dtype=np.float32), np.array([0.125], dtype=np.float32)]
    fixtures = [
        (np.array([[1., 2.], [3., -1.]], dtype=np.float32), np.array([[0.5], [-0.25]], dtype=np.float32)),
        (np.array([[-2., 1.], [0.5, 4.]], dtype=np.float32), np.array([[-1.], [1.5]], dtype=np.float32)),
    ]
    local_references = []
    for x, target in fixtures:
        residual = x @ initial[0].T + initial[1] - target
        local_references.append([2.0 * residual.T @ x / len(x), 2.0 * residual.mean(axis=0)])
    mean_reference = [sum(values) / world_size for values in zip(*local_references)]
    learning_rate = 0.1
    report = {"framework": args.framework, "rank": rank, "world_size": world_size,
              "device": args.device, "local_rank": local_rank,
              "backend": distributed_backend(args.device),
              "scope": "two-rank analytical DDP fixture; not full PPO or performance acceptance"}

    def fresh_model():
        module = torch.nn.Linear(2, 1, bias=True)
        module = module.to(device)
        with torch.no_grad():
            for parameter, value in zip(module.parameters(), initial):
                parameter.copy_(tensor(value + rank))
        before = [snapshot(p) for p in module.parameters()]
        for actual, expected in zip(before, initial):
            np.testing.assert_allclose(actual, expected + rank, rtol=0, atol=0)
        ddp = DistributedDataParallel(module, device_ids=[local_rank], broadcast_buffers=False)
        for parameter, expected in zip(module.parameters(), initial):
            np.testing.assert_allclose(snapshot(parameter), expected, rtol=0, atol=0)
        optimizer = torch.optim.SGD(module.parameters(), lr=learning_rate)
        return module, ddp, optimizer

    def spread(value):
        mean = value.detach().clone()
        dist.all_reduce(mean, op=dist.ReduceOp.SUM)
        mean = mean / world_size
        return float(np.max(np.abs(snapshot(value) - snapshot(mean))))

    with guard:
        dist.init_process_group(backend=distributed_backend(args.device), rank=rank, world_size=world_size)
        assert str(dist.get_backend()) == distributed_backend(args.device)
        assert dist.get_rank() == rank and dist.get_world_size() == world_size
        module, ddp, optimizer = fresh_model()
        report["initial_broadcast_exact"] = True
        report["marker_diagnostics"] = [
            {"type": type(p).__name__, "has_marker": getattr(p, "_jittor_ddp_state", None) is not None,
             "order": getattr(p, "_jittor_ddp_order", None)}
            for p in module.parameters()
        ]
        x, target = (tensor(value) for value in fixtures[rank])
        optimizer.zero_grad(set_to_none=True)
        loss = ((ddp(x) - target) ** 2).mean()
        loss.backward()
        grads = [p.grad for p in module.parameters()]
        assert len(grads) == 2 and all(g is not None for g in grads)
        gradient_errors = []
        for gradient, expected in zip(grads, mean_reference):
            actual = snapshot(gradient)
            assert np.isfinite(actual).all() and np.any(actual != 0)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
            gradient_errors.append(float(np.max(np.abs(actual - expected))))
        report["gradient_max_abs_error"] = max(gradient_errors)
        report["gradient_rank_spread_before_step"] = max(spread(g) for g in grads)
        assert report["gradient_rank_spread_before_step"] <= 1e-6
        optimizer.step()
        changed = []
        for parameter, before, gradient in zip(module.parameters(), initial, mean_reference):
            actual = snapshot(parameter)
            np.testing.assert_allclose(actual, before - learning_rate * gradient, rtol=1e-5, atol=1e-6)
            changed.append(float(np.max(np.abs(actual - before))))
        report["parameter_max_abs_change"] = max(changed)
        assert report["parameter_max_abs_change"] > 1e-4
        report["parameter_rank_spread_after_step"] = max(spread(p) for p in module.parameters())
        assert report["parameter_rank_spread_after_step"] <= 1e-6

        # Negative control: cover both forward and backward, as PyTorch requires.
        negative_module, negative_ddp, negative_optimizer = fresh_model()
        negative_optimizer.zero_grad(set_to_none=True)
        with negative_ddp.no_sync():
            negative_loss = ((negative_ddp(x) - target) ** 2).mean()
            negative_loss.backward()
        negative_grads = [p.grad for p in negative_module.parameters()]
        assert all(g is not None for g in negative_grads)
        for gradient, expected in zip(negative_grads, local_references[rank]):
            np.testing.assert_allclose(snapshot(gradient), expected, rtol=1e-5, atol=1e-6)
        report["no_sync_gradient_rank_spread"] = max(spread(g) for g in negative_grads)
        assert report["no_sync_gradient_rank_spread"] > 1e-4
        negative_optimizer.step()
        for parameter, before, gradient in zip(negative_module.parameters(), initial, local_references[rank]):
            np.testing.assert_allclose(snapshot(parameter), before - learning_rate * gradient, rtol=1e-5, atol=1e-6)
        report["no_sync_parameter_rank_spread"] = max(spread(p) for p in negative_module.parameters())
        assert report["no_sync_parameter_rank_spread"] > 1e-4
        report["parameter_location"] = location(next(iter(module.parameters())))
        report["loss_location"] = location(loss)
        synchronize()
        dist.barrier()

    if jt is not None:
        report["fallback_delta"] = jt.core.backend_fallback_count() - fallback_before
        assert report["fallback_delta"] == 0
    report["status"] = "pass"
    output = Path("{}.rank{}.json".format(args.output, rank))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("DDP_SYNC_PASS", json.dumps(report, ensure_ascii=False), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
