"""Shared Engine protocol validation and unchanged ecosystem numeric thresholds."""
import json
import os

import numpy as np

from _ecosystem_harness import (
    _comparison_floor, _divergence, _versions, EcosystemComparison,
)


def _compare_engine_outputs(actual, reference, *, device=None, world_size=1, rank=0,
                            dtype="float32", optimizer=None, zero_stage=None, perturbation=None):
    assert set(actual.files) == set(reference.files), "DeepSpeed observation schema differs"
    ours = json.loads(str(actual["metadata"]))
    theirs = json.loads(str(reference["metadata"]))
    assert ours["runtime"] == "jittor" and theirs["runtime"] == "torch"
    assert theirs["torch_binary"].endswith((".so", ".pyd"))
    assert _versions(ours) == _versions(theirs)
    for key in ("device", "world_size", "rank", "dtype", "optimizer", "zero_stage", "perturbation",
                "engine_class", "parameter_gradients_per_microstep", "input_gradients_per_microstep",
                "microsteps", "global_steps", "runtime_conditions", "optimizer_groups",
                "optimizer_partition_sizes", "model_parameter_elements", "trainable_parameters",
                "frozen_parameters", "strict_resume_fields", "gradient_dtypes", "jittor_commit",
                "seed", "hf_hub_offline", "protocol",
                "measurement_source_sha256"):
        assert ours[key] == theirs[key], "DeepSpeed protocol differs: " + key
    if ours["zero_stage"] == 0:
        # Stage zero retains each runtime's actual public optimizer class.
        # Compat publishes AdamW at torch.optim; binary Torch owns it in the
        # adamw module. Preserve both observed identities in the raw metadata.
        assert ours["optimizer_class"] == "torch.optim." + ours["optimizer"]
        assert theirs["optimizer_class"] == (
            "torch.optim." + theirs["optimizer"].lower() + "." + theirs["optimizer"])
    else:
        assert ours["optimizer_class"] == theirs["optimizer_class"]
    device = os.environ.get("JITTOR_TEST_DEVICES", "cpu") if device is None else device
    for key, expected in (("device", device), ("world_size", world_size), ("rank", rank),
                          ("dtype", dtype), ("optimizer", optimizer), ("zero_stage", zero_stage),
                          ("perturbation", perturbation)):
        if expected is not None:
            assert ours[key] == expected, "DeepSpeed requested protocol differs: " + key
    assert ours["strict_resume_fields"] > 0
    assert ours["seed"] == 0 and ours["hf_hub_offline"] is True
    if dtype == "bfloat16":
        assert ours["initial_weights_sha256"] == theirs["initial_weights_sha256"]
    expected_backend = "nccl" if device == "cuda" else "gloo"
    assert theirs["process_group_backend"] == expected_backend
    if world_size > 1:
        # WORLD's legacy singleton backend label does not identify an active
        # transport. Only real multi-rank observations establish a provider.
        assert ours["process_group_backend"] == ("mpi" if device == "cpu" else expected_backend)
    assert ours["updated_master_elements"] > 0 and theirs["updated_master_elements"] > 0
    assert ours["fallback_policy"] == "error" and ours["fallback_count"] == 0
    assert bool(ours["backend"]["use_cuda"]) == (device == "cuda")
    keys = [key for key in actual.files if key != "metadata"]
    for key in keys:
        if dtype == "bfloat16" and key.startswith("initial/parameter/"):
            np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)
    gradient_keys = [key for key in keys if "_grad/" in key]
    assert set(ours["gradient_dtypes"]) == set(gradient_keys), "DeepSpeed gradient dtype observations incomplete"
    forward_keys = [key for key in keys if key not in gradient_keys]
    errors = {}
    for category, group, tolerance in (
            ("forward_and_state", forward_keys, EcosystemComparison.forward_tolerance),
            ("gradients", gradient_keys, EcosystemComparison.backward_tolerance)):
        assert group, "DeepSpeed runner omitted " + category
        floor = _comparison_floor(reference, group)
        for key in group:
            assert actual[key].shape == reference[key].shape, key
            assert np.isfinite(actual[key]).all() and np.isfinite(reference[key]).all(), key
            difference = _divergence(actual[key], reference[key], floor)
            assert difference <= tolerance, (key, difference, tolerance)
            errors[category] = max(errors.get(category, 0), difference)
    return errors
