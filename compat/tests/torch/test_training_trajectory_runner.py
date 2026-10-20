"""Contract tests for trajectory artifact validation and merging."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _training_trajectory_runner as runner


@pytest.mark.parametrize(
    "left,right,kind",
    [
        (np.asarray([1.0, 2.0]), np.asarray([1.0]), "shape"),
        (np.asarray([1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float64), "dtype"),
        (np.asarray([np.inf]), np.asarray([np.inf]), "non-finite"),
        (np.asarray([np.nan]), np.asarray([np.nan]), "non-finite"),
    ],
)
def test_compare_rejects_shape_dtype_and_nonfinite_values(left, right, kind):
    result = runner._compare_array("state/step_01/param/weight", left, right, 1e-5, 1e-4)
    assert result["kind"] == kind


def test_compare_reports_numeric_trajectory_difference():
    result = runner._compare_array(
        "state/step_02/param/weight",
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1.1], dtype=np.float64),
        atol=1e-5,
        rtol=1e-4,
    )
    assert result["kind"] == "numeric"
    assert result["max_abs"] == pytest.approx(0.1)


def test_compare_accepts_exact_non_numeric_values():
    result = runner._compare_array(
        "state/step_01/trainable_names",
        np.asarray(["adapter.lora_A.weight"], dtype="U"),
        np.asarray(["adapter.lora_A.weight"], dtype="U"),
        atol=1e-5,
        rtol=1e-4,
    )
    assert result == {"key": "state/step_01/trainable_names", "exact": True}


def _write_segment(path, steps):
    values = [float(step) for step in steps]
    artifact = {"losses": np.asarray(values, dtype=np.float64)}
    for step, loss in zip(steps, values):
        artifact["data/step_%02d/loss" % step] = np.asarray(loss, dtype=np.float64)
        artifact["state/step_%02d/param/weight" % step] = np.asarray(
            [loss], dtype=np.float32
        )
    np.savez(path, **artifact)


def test_merge_orders_segmented_losses_by_update_step(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_segment(first, [1, 2])
    _write_segment(second, [3, 4])

    merged = runner._merge([str(second), str(first)])

    np.testing.assert_array_equal(merged["losses"], [1.0, 2.0, 3.0, 4.0])
    assert "state/step_01/param/weight" in merged
    assert "state/step_04/param/weight" in merged


def test_merge_rejects_gaps_and_conflicting_duplicate_step_losses(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_segment(first, [1])
    _write_segment(second, [3])
    with pytest.raises(AssertionError, match="contain a gap"):
        runner._merge([str(first), str(second)])

    _write_segment(second, [1])
    with np.load(second) as source:
        duplicate = {name: source[name].copy() for name in source.files}
    duplicate["data/step_01/loss"] = np.asarray(9.0, dtype=np.float64)
    duplicate["losses"] = np.asarray([9.0], dtype=np.float64)
    np.savez(second, **duplicate)
    with pytest.raises(AssertionError, match="duplicate step loss differs"):
        runner._merge([str(first), str(second)])


def test_expected_update_count_must_match_and_have_no_partial_window():
    runner._assert_complete_updates(0, 4, 4, 0)
    with pytest.raises(AssertionError, match="expected 4 complete updates"):
        runner._assert_complete_updates(0, 4, 3, 0)
    with pytest.raises(AssertionError, match="pending batches"):
        runner._assert_complete_updates(0, 4, 4, 1)
