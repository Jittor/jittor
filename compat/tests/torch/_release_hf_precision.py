"""Pure validation of observed public GradScaler optimizer attempts."""

import math


def validate_optimizer_attempt(precision, scaled_nonfinite, unscaled_health, skipped,
                               scaler_before, scaler_after, skipped_state_unchanged=None):
    """An overflow consumes data/a Trainer step, never an optimizer update.

    This function observes the default GradScaler contract. It never changes
    the scale, gradients, optimizer, sampler or Trainer's update scheduling.
    """
    unscaled_nonfinite = int(unscaled_health["nonfinite_count"]) > 0
    overflow = bool(scaled_nonfinite or unscaled_nonfinite)
    if precision != "fp16":
        if overflow or skipped:
            raise AssertionError("FP32/BF16 finite workflow cannot overflow or skip an optimizer step")
        return {"overflow": False, "successful_update": True}
    if scaler_before is None or scaler_after is None:
        raise AssertionError("FP16 attempt lacks enabled GradScaler state")
    before, after = float(scaler_before["scale"]), float(scaler_after["scale"])
    if not all(math.isfinite(value) and value > 0 for value in (before, after)):
        raise AssertionError("FP16 GradScaler scale is not finite and positive")
    if overflow:
        if not skipped:
            raise AssertionError("nonfinite scaled/unscaled gradients did not skip the optimizer step")
        if skipped_state_unchanged is not True:
            raise AssertionError("skipped optimizer attempt changed model/optimizer/scheduler state")
        if after != before * 0.5 or int(scaler_after["_growth_tracker"]) != 0:
            raise AssertionError("overflow did not halve the default scale and reset growth tracker")
    else:
        if skipped:
            raise AssertionError("finite unscaled gradients unexpectedly skipped an optimizer step")
        tracker = int(scaler_before["_growth_tracker"]) + 1
        grows = tracker >= int(scaler_before["growth_interval"])
        expected_scale = before * float(scaler_before["growth_factor"]) if grows else before
        if after != expected_scale or int(scaler_after["_growth_tracker"]) != (0 if grows else tracker):
            raise AssertionError("finite optimizer update violated GradScaler growth policy")
    return {"overflow": overflow, "successful_update": not skipped,
            "scale_before": before, "scale_after": after,
            "skipped_state_unchanged": skipped_state_unchanged if skipped else None}
