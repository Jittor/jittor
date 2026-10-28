"""Two real native NCCL ranks: full oracle and fresh-launch continuation."""

import json
import os
from pathlib import Path
import sys

import jittor as jt
from jittor import distributed as dist
import numpy as np
import pytest
from _helpers.native_fsdp import assert_oracle_tensor, assert_oracle_update, run_cuda_trajectory

_READY = bool(jt.compiler.has_cuda and jt.flags.use_cuda and dist.get_world_size() == 2)
if os.environ.get("JITTOR_REQUIRE_NATIVE_FSDP") == "1" and not _READY:
    raise RuntimeError("required native FSDP acceptance needs two real CUDA NCCL ranks")
pytestmark = pytest.mark.skipif(not _READY, reason="requires two actual CUDA NCCL ranks")


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_oracle_and_fresh_launcher_resume(kind, dtype):
    assert "torch" not in sys.modules and not any(
        n.startswith("jittor.compat") for n in sys.modules
    )
    root = Path(os.environ["JITTOR_NATIVE_FSDP_TEST_DIR"])
    mode = os.environ["JITTOR_NATIVE_FSDP_PHASE"]
    assert mode in ("parity", "baseline", "resume")
    output = root / f"{mode}-{kind}-{dtype}"
    arrays, metadata = run_cuda_trajectory(
        mode=mode,
        kind=kind,
        dtype=dtype,
        output=output,
        checkpoint_path=root / f"checkpoint-{kind}-{dtype}",
    )
    assert metadata["fallback_delta"] == 0
    rank = dist.get_rank()
    if mode == "parity":
        oracle = (
            Path(os.environ["JITTOR_NATIVE_FSDP_ORACLE_DIR"]) / f"oracle-{kind}-{dtype}-cuda.npz"
        )
        diagnostics = {}
        with np.load(oracle) as expected:
            for key, actual in arrays.items():
                parts = key.split("/")
                if parts[2] in ("prediction", "loss", "input_gradient"):
                    reference = expected[f"step/{parts[1]}/rank/{rank}/{parts[2]}"]
                else:
                    if parts[2] == "state":
                        parts[4] = {
                            "m": "exp_avg",
                            "values": "momentum_buffer" if kind == "sgd" else "exp_avg_sq",
                        }[parts[4]]
                    global_value = expected["/".join(parts)].reshape(-1)
                    size = (global_value.size + 1) // 2
                    padded = np.pad(global_value, (0, 2 * size - global_value.size))
                    reference = padded[rank * size : (rank + 1) * size]
                assert_oracle_tensor(actual, reference, key, dtype)
                if parts[2] == "gradient":
                    # Inject a zero gradient for every actual parameter/step.
                    with pytest.raises(AssertionError):
                        assert_oracle_tensor(np.zeros_like(actual), reference, key, dtype)
                    diagnostics[key] = {
                        "oracle_norm": float(np.linalg.norm(reference)),
                        "error_norm": float(np.linalg.norm(actual - reference)),
                        "zero_gradient_rejected": True,
                    }
                if parts[2] == "parameter":
                    step = int(parts[1])
                    name = parts[3]
                    previous_key = (
                        f"step/{step - 1}/parameter/{name}" if step else f"initial/parameter/{name}"
                    )
                    previous_global = expected[previous_key].reshape(-1)
                    previous_reference = np.pad(
                        previous_global, (0, 2 * size - previous_global.size)
                    )[rank * size : (rank + 1) * size]
                    previous_actual = arrays[previous_key] if step else previous_reference
                    actual_delta = actual - previous_actual
                    reference_delta = reference - previous_reference
                    diagnostics[key] = assert_oracle_update(
                        actual_delta, reference_delta, key, dtype
                    )
                    with pytest.raises(AssertionError):
                        assert_oracle_update(
                            np.zeros_like(actual_delta), reference_delta, key, dtype
                        )
                    diagnostics[key]["zero_update_rejected"] = True
        (output / f"rank{rank}-oracle-comparison.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
        )
    if mode == "resume":
        with np.load(root / f"baseline-{kind}-{dtype}" / f"rank{rank}.npz") as expected:
            keys = {key for key in expected.files if key.startswith(("step/3/", "step/4/"))}
            assert set(arrays) == keys
            for key, actual in arrays.items():
                assert np.isfinite(actual).all()
                np.testing.assert_array_equal(actual, expected[key], err_msg=key)


def test_initial_parameter_drift_is_rejected_before_collectives():
    from _helpers.native_fsdp import model
    from jittor.distributed.fsdp import FullyShardedDataParallel

    module = model(jt, native=True)
    if dist.get_rank():
        module.fc1.weight.update(module.fc1.weight + 0.125)
    with pytest.raises(ValueError, match="rank metadata or initial values differ"):
        FullyShardedDataParallel(module)


def test_optimizer_control_drift_is_rejected_before_reduce_scatter():
    from _helpers.native_fsdp import model, optimizer
    from jittor.distributed.fsdp import FullyShardedDataParallel

    module = FullyShardedDataParallel(model(jt, native=True))
    opt = optimizer(jt, module, "sgd")
    opt.param_groups[0]["lr"] += dist.get_rank() * 0.1
    loss = module(jt.ones((3, 5))).sum()
    with pytest.raises(ValueError, match="rank metadata or initial values differ"):
        opt.backward(loss)


def test_replicated_buffer_cannot_be_hidden_in_rank_local_checkpoint():
    from _helpers.native_fsdp import model, optimizer
    from jittor.distributed import checkpoint
    from jittor.distributed.fsdp import FullyShardedDataParallel

    module = FullyShardedDataParallel(model(jt, native=True))
    opt = optimizer(jt, module, "sgd")
    opt.step((module(jt.ones((3, 5))) ** 2).mean())
    if dist.get_rank():
        module.module.gain.update(module.module.gain + 0.125)
    directory = Path(os.environ["JITTOR_NATIVE_FSDP_TEST_DIR"]) / "bad-replicated-buffer"
    with pytest.raises(checkpoint.CheckpointError, match="replicas differ at /model/module.gain"):
        checkpoint.save_training_state(module, opt, directory)
