"""Torch and native DDP must have distinct reduction ownership."""
from unittest.mock import patch

import numpy as np
import pytest
import jittor as jt
import torch
from jittor import distributed
from jittor.nn.parallel import DistributedDataParallel as NativeDDP
from jittor.compat import collectives


class _TwoRanks:
    def size(self):
        return 2


@pytest.mark.parametrize("optimizer_first", [True, False])
def test_torch_owner_is_not_registered_with_native_optimizer(optimizer_first):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1) if optimizer_first else None
    wrapped = torch.nn.parallel.DistributedDataParallel(model)
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    parameter, = list(model.parameters())
    assert parameter._torch_ddp_state is wrapped._jittor_ddp_state
    assert getattr(parameter, "_jittor_ddp_state", None) is None
    assert optimizer._jittor_ddp_states == {}
    optimizer.zero_grad()
    # The native pre_step check must not treat the Torch owner as its reducer.
    optimizer.step()


@pytest.mark.parametrize("optimizer_first", [True, False])
def test_torch_no_sync_accumulates_and_only_final_backward_reduces(optimizer_first):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1) if optimizer_first else None
    wrapped = torch.nn.parallel.DistributedDataParallel(model)
    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    first = torch.tensor([[1.0]], device="cpu")
    second = torch.tensor([[2.0]], device="cpu")
    # One-process deterministic collective probe: the function doubles the
    # accumulated gradient so an omitted or repeated call changes the answer.
    with patch.object(collectives, "_world_size", return_value=2), \
            patch.object(collectives, "_all_reduce_mean", side_effect=lambda x: x * 2) as reduce:
        with wrapped.no_sync():
            assert not wrapped.require_backward_grad_sync
            with wrapped.no_sync():
                assert not wrapped.require_backward_grad_sync
            assert not wrapped.require_backward_grad_sync
            wrapped(first).sum().backward()
            assert reduce.call_count == 0
        assert wrapped.require_backward_grad_sync
        wrapped(second).sum().backward()
        assert reduce.call_count == 1
        parameter, = list(model.parameters())
        np.testing.assert_allclose(parameter.grad.detach().cpu().numpy(), [[6.0]], rtol=0, atol=1e-6)
        optimizer.step()
        assert reduce.call_count == 1
        np.testing.assert_allclose(parameter.detach().cpu().numpy(), [[0.4]], rtol=0, atol=1e-6)


def test_native_wrapper_rejects_torch_owned_parameters_before_communication():
    module = torch.nn.Linear(1, 1)
    wrapped = torch.nn.parallel.DistributedDataParallel(module)
    with patch.object(distributed, "get_default_group", return_value=_TwoRanks()), \
            patch.object(distributed, "is_initialized", return_value=True), \
            patch.object(distributed, "all_reduce") as reduce, \
            patch.object(distributed, "broadcast") as broadcast:
        with pytest.raises(ValueError, match="both Torch and native DDP"):
            NativeDDP(module)
        reduce.assert_not_called()
        broadcast.assert_not_called()
    assert all(p._torch_ddp_state is wrapped._jittor_ddp_state for p in module.parameters())
    assert all(getattr(p, "_jittor_ddp_state", None) is None for p in module.parameters())


def test_torch_wrapper_rejects_native_owned_parameters_before_broadcast():
    module = torch.nn.Linear(1, 1)
    with patch.object(distributed, "get_default_group", return_value=None), \
            patch.object(distributed, "get_world_size", return_value=1):
        wrapped = NativeDDP(module)
    with patch.object(collectives, "_world_size", return_value=2), \
            patch.object(collectives, "_broadcast_from_rank0") as broadcast:
        with pytest.raises(ValueError, match="both Torch and native DDP"):
            torch.nn.parallel.DistributedDataParallel(module)
        broadcast.assert_not_called()
    assert all(p._jittor_ddp_state is wrapped._jittor_ddp_state for p in module.parameters())
    assert all(getattr(p, "_torch_ddp_state", None) is None for p in module.parameters())


@pytest.mark.parametrize("optimizer_type", [jt.optim.SGD, torch.optim.SGD, torch.optim.AdamW])
def test_native_loss_backward_or_step_rejects_torch_ddp_ownership(optimizer_type):
    model = torch.nn.Linear(1, 1, bias=False)
    wrapped = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = optimizer_type(model.parameters(), lr=0.1)
    loss = wrapped(torch.tensor([[1.0]], device="cpu")).sum()
    before_values = [p.detach().cpu().numpy().copy() for p in model.parameters()]
    before_n_step = int(optimizer.n_step)
    before_steps = [list(group.get("_torch_steps", [])) for group in optimizer.param_groups]
    with pytest.raises(RuntimeError, match=r"native optimizer\.backward\(loss\).*Torch DDP"):
        optimizer.backward(loss)
    with pytest.raises(RuntimeError, match=r"native optimizer\.backward\(loss\).*Torch DDP"):
        optimizer.step(loss)
    assert int(optimizer.n_step) == before_n_step
    assert [list(group.get("_torch_steps", [])) for group in optimizer.param_groups] == before_steps
    for parameter, original in zip(model.parameters(), before_values):
        np.testing.assert_array_equal(parameter.detach().cpu().numpy(), original)


@pytest.mark.parametrize("optimizer_type", [torch.optim.SGD, torch.optim.AdamW])
def test_torch_ddp_optimizer_closure_uses_public_backward(optimizer_type):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    wrapped = torch.nn.parallel.DistributedDataParallel(model)
    optimizer = optimizer_type(model.parameters(), lr=0.1)
    inputs = torch.tensor([[1.0]], device="cpu")
    calls = []

    def closure():
        calls.append(True)
        optimizer.zero_grad()
        loss = wrapped(inputs).sum()
        loss.backward()
        return loss

    before = model.weight.detach().cpu().numpy().copy()
    loss = optimizer.step(closure)
    assert calls == [True]
    assert loss is not None and np.isfinite(float(loss.item()))
    after = model.weight.detach().cpu().numpy()
    assert np.isfinite(after).all()
    assert not np.array_equal(before, after)


@pytest.mark.parametrize("optimizer_type", [jt.optim.SGD, torch.optim.SGD, torch.optim.AdamW])
def test_torch_ddp_input_parameter_cannot_enter_native_backward(optimizer_type):
    model = torch.nn.Linear(1, 1, bias=False)
    wrapped = torch.nn.parallel.DistributedDataParallel(model)
    independent = torch.nn.Parameter(torch.ones((1,), device="cpu"))
    optimizer = optimizer_type([independent], lr=0.1)
    optimizer.set_input_into_param_group([model.weight])
    loss = wrapped(torch.tensor([[1.0]], device="cpu")).sum() + independent.sum()
    before = independent.detach().cpu().numpy().copy()
    before_n_step = int(optimizer.n_step)
    before_steps = [list(group.get("_torch_steps", [])) for group in optimizer.param_groups]
    with pytest.raises(RuntimeError, match=r"native optimizer\.backward\(loss\).*Torch DDP"):
        optimizer.backward(loss)
    with pytest.raises(RuntimeError, match=r"native optimizer\.backward\(loss\).*Torch DDP"):
        optimizer.step(loss)
    assert int(optimizer.n_step) == before_n_step
    assert [list(group.get("_torch_steps", [])) for group in optimizer.param_groups] == before_steps
    np.testing.assert_array_equal(independent.detach().cpu().numpy(), before)



def test_native_reducer_bucket_sizes_accept_torch_dtype_properties():
    from importlib import import_module

    ddp = import_module("jittor.nn.parallel.distributed_data_parallel")
    specs = [(torch.float16, 2), (torch.float16, 2), (torch.float16, 2),
             (torch.bfloat16, 2), (torch.bfloat16, 2), (torch.float32, 2),
             (torch.float32, 3), (torch.float32, 1), (torch.float64, 1)]
    parameters = [(str(i), torch.nn.Parameter(torch.zeros((count,), dtype=dtype)))
                  for i, (dtype, count) in enumerate(specs)]
    assert str(parameters[3][1].dtype) == "torch.bfloat16"
    with patch.object(ddp, "_BUCKET_BYTES", 8):
        assert ddp._gradient_buckets(parameters) == (
            (0, 1), (2,), (3, 4), (5,), (6,), (7,), (8,))

    # Preserve the existing public-dtype grouping rule for mixed frontends,
    # even when the native storage width happens to match.
    native = jt.zeros((1,), dtype="float32").start_grad()
    frontend = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))
    assert str(native.dtype) != str(frontend.dtype)
    with patch.object(ddp, "_BUCKET_BYTES", 8):
        assert ddp._gradient_buckets((("native", native), ("torch", frontend))) == ((0,), (1,))
