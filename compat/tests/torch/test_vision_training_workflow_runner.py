"""Framework-free contracts for the real-data vision acceptance protocol."""

import argparse
import copy

import numpy as np
import pytest


import _vision_training_workflow_runner as runner


@pytest.mark.parametrize("samples,updates", [(257, 5), (3137, 50)])
def test_real_data_protocol_has_single_example_batch_and_partial_accumulation(samples, updates):
    windows = runner._windows(samples, 32, 2)
    assert len(windows) == updates
    assert len(windows[-1]) == 1
    assert len(windows[-1][0]) == 1
    flattened = [index for window in windows for batch in window for index in batch]
    assert flattened == list(range(samples))


def test_tail_window_normalization_uses_actual_samples():
    windows = runner._windows(257, 32, 2)
    assert [sum(map(len, window)) for window in windows] == [64, 64, 64, 64, 1]
    full = runner._cursor_after(0, 9, 9)
    assert full == {"epoch": 1, "batch": 0}
    assert runner._cursor_after(1, 4, 9) == {"epoch": 1, "batch": 4}


def test_augmentation_replays_independent_of_access_order_and_global_rng():
    image = np.arange(3 * 32 * 32, dtype=np.int64).astype(np.uint8).reshape(3, 32, 32)
    expected = runner._augmentation(image, 17, 1, 9, True)
    np.random.seed(183)
    runner._augmentation(image, 91, 2, 5, True)
    np.testing.assert_array_equal(runner._augmentation(image, 17, 1, 9, True), expected)
    assert expected.dtype == np.float32
    assert expected.shape == (3, 32, 32)
    assert expected.flags.c_contiguous
    assert not np.array_equal(expected, runner._augmentation(image, 17, 2, 9, True))


def _checkpoint():
    configuration = {"epochs": 2, "data": {"subset_sha256": "fixture"}}
    return {"configuration": configuration, "cursor": {"epoch": 1, "batch": 4},
            "completed_updates": 7}, configuration


def test_checkpoint_validates_fixture_cursor_and_completed_updates():
    checkpoint, configuration = _checkpoint()
    runner._validate_checkpoint(checkpoint, configuration, 9, 2)
    wrong = copy.deepcopy(configuration)
    wrong["data"]["subset_sha256"] = "different"
    with pytest.raises(ValueError, match="fixture differs"):
        runner._validate_checkpoint(checkpoint, wrong, 9, 2)
    checkpoint["cursor"]["batch"] = 3
    with pytest.raises(ValueError, match="accumulation window"):
        runner._validate_checkpoint(checkpoint, configuration, 9, 2)
    checkpoint["cursor"]["batch"] = 4
    checkpoint["completed_updates"] = 8
    with pytest.raises(ValueError, match="update count"):
        runner._validate_checkpoint(checkpoint, configuration, 9, 2)


def _artifact(path, steps, final_value):
    arrays = {"completed_updates": np.asarray(max(steps)),
              "configuration_json": np.asarray("fixture"),
              "final/param/weight": np.asarray([final_value], dtype=np.float32)}
    for step in steps:
        arrays["history/step_%05d/loss" % step] = np.asarray(float(step))
    np.savez(path, **arrays)


def test_split_merge_orders_updates_and_takes_only_latest_complete_state(tmp_path):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _artifact(first, [1, 2], 0.2)
    _artifact(second, [3, 4], 0.4)
    joined = runner._merge([second, first])
    assert joined["completed_updates"] == 4
    np.testing.assert_array_equal(joined["final/param/weight"], np.asarray([0.4], dtype=np.float32))
    assert len([key for key in joined if key.startswith("history/")]) == 4
    _artifact(second, [4], 0.4)
    with pytest.raises(AssertionError, match="without gaps"):
        runner._merge([first, second])


def test_split_merge_rejects_conflicting_duplicate_final_states(tmp_path):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    _artifact(first, [1, 2], 0.2)
    _artifact(second, [1, 2], 0.4)
    with pytest.raises(AssertionError, match="conflicting final states"):
        runner._merge([first, second])


def _tolerances():
    return argparse.Namespace(parameter_atol=2e-5, parameter_rtol=0.003,
                              buffer_atol=5e-5, buffer_rtol=0.004,
                              loss_atol=2e-5, loss_rtol=0.004, accuracy_atol=0.0)


def test_comparison_uses_frozen_scale_and_near_zero_absolute_floor():
    args = _tolerances()
    assert runner._difference("final/param/weight", np.asarray([1e-6]), np.asarray([0.0]), args) is None
    mismatch = runner._difference("final/param/weight", np.asarray([1.01]), np.asarray([1.0]), args)
    assert mismatch["kind"] == "numeric"
    assert mismatch["threshold"] == pytest.approx(0.003)
    assert runner._difference("final/buffer/counter", np.asarray([2]), np.asarray([1]), args)["kind"] == "exact"
    assert runner._difference("history/step_00001/loss", np.asarray(np.nan), np.asarray(np.nan), args)["kind"] == "nonfinite"
    assert runner._difference("final/param/weight", np.asarray([1.0], dtype=np.float32),
                              np.asarray([1.0], dtype=np.float64), args)["kind"] == "shape_or_dtype"


def test_scheduler_and_scaler_are_exact_state_contracts():
    args = _tolerances()
    assert runner._difference("final/scheduler/_last_lr/0", np.asarray(0.001001),
                              np.asarray(0.001), args)["kind"] == "numeric"
    assert runner._difference("final/scaler/scale", np.asarray(128.0),
                              np.asarray(64.0), args)["kind"] == "numeric"
