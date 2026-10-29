"""LambdaLR initial-rate and optimizer-only reconstruction regressions."""

import os

import jittor as jt
import pytest

from jittor.compat.torch.tensor_state import compatibility_owner


def _factor(step):
    return max(0.0, float(4 - step) / 4.0)


def _optimizer(torch, rates=(0.003, 0.004)):
    parameters = [
        torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device="cpu")),
        torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float32, device="cpu")),
    ]
    return torch.optim.AdamW(
        [{"params": [parameter], "lr": rate}
         for parameter, rate in zip(parameters, rates)],
        lr=0.003,
    )


def _number(value):
    return float(value.item()) if hasattr(value, "item") else float(value)


def _rates(values):
    return [_number(value) for value in values]


def _after_two_steps(torch):
    optimizer = _optimizer(torch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _factor)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    return optimizer, scheduler


def test_fresh_lambda_records_distinct_initial_lr_and_preserves_existing_values():
    torch = compatibility_owner(jt)
    optimizer = _optimizer(torch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _factor)
    assert _rates(group["initial_lr"] for group in optimizer.param_groups) == pytest.approx([0.003, 0.004])
    assert _rates(scheduler.base_lrs) == pytest.approx([0.003, 0.004])
    optimizer.step()
    scheduler.step()
    assert _rates(scheduler.get_last_lr()) == pytest.approx([0.00225, 0.003])

    custom = _optimizer(torch)
    custom.param_groups[0]["initial_lr"] = 0.012
    custom.param_groups[1]["initial_lr"] = 0.008
    custom_scheduler = torch.optim.lr_scheduler.LambdaLR(custom, _factor)
    assert _rates(custom_scheduler.base_lrs) == pytest.approx([0.012, 0.008])
    assert _rates(custom_scheduler.get_last_lr()) == pytest.approx([0.012, 0.008])


def test_optimizer_only_restore_rebuilds_lambda_from_original_and_custom_rates():
    torch = compatibility_owner(jt)
    optimizer, _ = _after_two_steps(torch)
    saved = optimizer.state_dict()
    restored = _optimizer(torch)
    restored.load_state_dict(saved)
    scheduler = torch.optim.lr_scheduler.LambdaLR(restored, _factor, last_epoch=2)
    assert scheduler.last_epoch == 3
    assert _rates(scheduler.base_lrs) == pytest.approx([0.003, 0.004])
    assert _rates(scheduler.get_last_lr()) == pytest.approx([0.00075, 0.001])

    custom = _optimizer(torch)
    custom.load_state_dict(saved)
    custom.param_groups[0]["initial_lr"] = 0.012
    custom.param_groups[1]["initial_lr"] = 0.008
    custom_scheduler = torch.optim.lr_scheduler.LambdaLR(custom, _factor, last_epoch=2)
    assert _rates(custom_scheduler.base_lrs) == pytest.approx([0.012, 0.008])
    assert _rates(custom_scheduler.get_last_lr()) == pytest.approx([0.003, 0.002])


@pytest.mark.parametrize("missing_group", (0, 1))
def test_reconstructed_lambda_requires_each_initial_lr(missing_group):
    torch = compatibility_owner(jt)
    optimizer, _ = _after_two_steps(torch)
    restored = _optimizer(torch)
    restored.load_state_dict(optimizer.state_dict())
    restored.param_groups[missing_group].pop("initial_lr")
    before = _rates(group["lr"] for group in restored.param_groups)
    with pytest.raises(KeyError, match=rf"param_groups\[{missing_group}\]"):
        torch.optim.lr_scheduler.LambdaLR(restored, _factor, last_epoch=2)
    assert _rates(group["lr"] for group in restored.param_groups) == before


def test_tensor_learning_rate_is_cloned_for_initial_lr():
    torch = compatibility_owner(jt)
    rate = torch.tensor(0.003, dtype=torch.float32, device="cpu")
    optimizer = _optimizer(torch, (rate, 0.004))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _factor)
    initial = optimizer.param_groups[0]["initial_lr"]
    assert initial is not rate
    rate.fill_(0.007)
    assert _number(initial) == pytest.approx(0.003, rel=1e-6, abs=1e-9)
    assert _number(scheduler.base_lrs[0]) == pytest.approx(0.003, rel=1e-6, abs=1e-9)


def test_optimizer_and_scheduler_checkpoint_restore_preserves_curve(tmp_path):
    torch = compatibility_owner(jt)
    optimizer, scheduler = _after_two_steps(torch)
    checkpoint = tmp_path / "lambda-state.pt"
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()},
               os.fspath(checkpoint))
    saved = torch.load(os.fspath(checkpoint), map_location="cpu", weights_only=True)
    restored = _optimizer(torch)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored, _factor)
    restored.load_state_dict(saved["optimizer"])
    restored_scheduler.load_state_dict(saved["scheduler"])
    assert restored_scheduler.last_epoch == 2
    assert _rates(restored_scheduler.base_lrs) == pytest.approx([0.003, 0.004])
    restored.step()
    restored_scheduler.step()
    assert _rates(restored_scheduler.get_last_lr()) == pytest.approx([0.00075, 0.001])
