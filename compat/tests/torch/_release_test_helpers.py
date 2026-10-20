"""Shared public runtime and scheduler helpers for release contract tests."""
from contextlib import contextmanager
import os
import warnings

import torch


@contextmanager
def public_device_scope(device=None, set_default=False):
    device = device or os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
    assert device in ("cpu", "cuda")
    if device == "cuda":
        assert torch.cuda.is_available(), "requires a real CUDA device"
    is_shim = hasattr(torch, "_torch_compat_install_context")
    previous = torch.get_default_device() if set_default else None
    if is_shim:
        import jittor as jt
        from jittor._runtime.fallback import forbid_backend_fallbacks
        scope = jt.runtime.scope(use_cuda=int(device == "cuda"), backend_fallback="error")
        scope.__enter__()
        guard = forbid_backend_fallbacks()
        guard.__enter__()
        before = jt.core.backend_fallback_count()
    try:
        if set_default:
            torch.set_default_device(device)
        yield device
        if device == "cuda":
            torch.cuda.synchronize()
        if is_shim:
            assert jt.core.backend_fallback_count() == before
    finally:
        if set_default:
            torch.set_default_device(previous)
        if is_shim:
            guard.__exit__(None, None, None)
            scope.__exit__(None, None, None)


def values(tensor):
    return tensor.detach().cpu().numpy().copy()


def on_device(tensor, device):
    assert tensor.device.type == device
    if hasattr(torch, "_torch_compat_install_context"):
        import jittor as jt
        assert jt.core.dispatch_context([tensor])[0] == device


def linear_schedule(device, *, start_factor=0.5, end_factor=1.0, total_iters=4):
    parameters = [torch.nn.Parameter(torch.tensor([1.0], device=device)),
                  torch.nn.Parameter(torch.tensor([2.0], device=device))]
    assert all(parameter.device.type == device for parameter in parameters)
    optimizer = torch.optim.SGD([{"params": [parameters[0]], "lr": 0.2},
                                 {"params": [parameters[1]], "lr": 0.4}])
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=end_factor,
        total_iters=total_iters)
    return optimizer, scheduler


def step_scheduler(optimizer, scheduler, epoch=None):
    optimizer.step()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scheduler.step(epoch)


def optimizer_lrs(optimizer):
    return [group["lr"] for group in optimizer.param_groups]
