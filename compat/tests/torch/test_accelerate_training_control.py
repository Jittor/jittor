"""Exact single-process Accelerate accumulation and scheduler trajectories."""
import os
from contextlib import nullcontext

import numpy as np
import pytest
import torch
try:
    from accelerate import Accelerator
    from accelerate.state import AcceleratorState, GradientState
except ModuleNotFoundError as exc:
    if exc.name != "accelerate" or os.environ.get("JITTOR_REQUIRE_ACCELERATE") == "1":
        raise
    Accelerator = AcceleratorState = GradientState = None

pytestmark = pytest.mark.skipif(Accelerator is None,
                                reason="optional dependency accelerate is not installed")


_DEVICE = os.environ.get("JITTOR_TEST_DEVICES", "cpu").split(",")[0]
_SHIM = hasattr(torch, "_torch_compat_install_context")
if _SHIM:
    import jittor as jt


def _values(value):
    return value.detach().clone().cpu().numpy().copy()


@pytest.fixture(autouse=True)
def _isolated_state():
    if _DEVICE == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._shared_state.clear()
    scope = jt.flag_scope(use_cuda=int(_DEVICE == "cuda")) if _SHIM else nullcontext()
    with scope:
        yield
    AcceleratorState._reset_state(reset_partial_state=True)
    GradientState._shared_state.clear()


@pytest.mark.parametrize("samples, expected_weight, expected_lr, sync_pattern", (
    (2, .5, .05, [False, True]),
    (3, .275, .025, [False, True, True]),
))
def test_accumulation_updates_and_scheduler_follow_exact_oracle(
        samples, expected_weight, expected_lr, sync_pattern):
    accelerator = Accelerator(cpu=_DEVICE == "cpu", gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.5)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor([[float(i)] for i in range(1, samples + 1)]),
        torch.zeros((samples, 1))), batch_size=1)
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    observed = []
    updates = 0
    for x, y in loader:
        with accelerator.accumulate(model):
            assert x.device.type == _DEVICE
            accelerator.backward((model(x) - y).square().mean())
            before = _values(model.weight)
            observed.append(accelerator.sync_gradients)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if not np.array_equal(_values(model.weight), before):
                updates += 1
            if not accelerator.sync_gradients:
                np.testing.assert_array_equal(_values(model.weight), before)
                assert scheduler.get_last_lr() == [.1]
    assert observed == sync_pattern
    assert updates == sum(sync_pattern)
    np.testing.assert_allclose(_values(model.weight), [[expected_weight]], rtol=0, atol=2e-7)
    assert scheduler.get_last_lr() == [expected_lr]
    assert scheduler.scheduler.last_epoch == updates
    assert scheduler.scheduler._step_count == updates + 1
    assert model.weight.grad is None


def test_two_optimizers_clip_and_schedulers_have_independent_exact_updates():
    accelerator = Accelerator(cpu=_DEVICE == "cpu", gradient_accumulation_steps=2)
    left, right = torch.nn.Linear(1, 1, bias=False), torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        left.weight.fill_(1.)
        right.weight.fill_(2.)
    oa, ob = torch.optim.SGD(left.parameters(), lr=.1), torch.optim.SGD(right.parameters(), lr=.2)
    sa = torch.optim.lr_scheduler.StepLR(oa, 1, gamma=.5)
    sb = torch.optim.lr_scheduler.StepLR(ob, 1, gamma=.25)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.ones((2, 1))), batch_size=1)
    left, right, oa, ob, sa, sb, loader = accelerator.prepare(left, right, oa, ob, sa, sb, loader)
    for (x,) in loader:
        with accelerator.accumulate(left, right):
            accelerator.backward(left(x).sum() + 2 * right(x).sum())
            if accelerator.sync_gradients:
                norm = accelerator.clip_grad_norm_([left.weight, right.weight], 1.)
                np.testing.assert_allclose(_values(norm), np.sqrt(5), rtol=2e-6)
            oa.step()
            ob.step()
            sa.step()
            sb.step()
            oa.zero_grad()
            ob.zero_grad()
    coefficient = 1 / (np.sqrt(5) + 1e-6)
    np.testing.assert_allclose(_values(left.weight), [[1 - .1 * coefficient]], rtol=0, atol=2e-7)
    np.testing.assert_allclose(_values(right.weight), [[2 - .4 * coefficient]], rtol=0, atol=2e-7)
    assert sa.get_last_lr() == [.05]
    assert sb.get_last_lr() == [.05]
    assert sa.scheduler.last_epoch == sb.scheduler.last_epoch == 1


@pytest.mark.skipif(_DEVICE != "cuda", reason="Accelerate native fp16 scaling requires CUDA")
def test_fp16_overflow_accumulation_skips_scheduler_then_finite_step_succeeds():
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=2)
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(.0625)
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=.5)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor([[.125], [.25], [.125], [.25]]),
        torch.tensor([[float("inf")], [0.], [0.], [0.]])), batch_size=1)
    model, optimizer, scheduler, loader = accelerator.prepare(model, optimizer, scheduler, loader)
    for index, (x, y) in enumerate(loader):
        with accelerator.accumulate(model):
            with accelerator.autocast():
                loss = (model(x) - y).square().mean()
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if index == 1:
                np.testing.assert_array_equal(_values(model.weight), [[.0625]])
                assert optimizer.step_was_skipped
                assert scheduler.get_last_lr() == [.1]
                assert scheduler.scheduler.last_epoch == 0
                assert accelerator.scaler.get_scale() == 32768.
    assert not optimizer.step_was_skipped
    np.testing.assert_allclose(_values(model.weight), [[.06201171875]], rtol=0, atol=2e-7)
    assert scheduler.get_last_lr() == [.05]
    assert scheduler.scheduler.last_epoch == 1
    assert accelerator.scaler.get_scale() == 32768.
