"""Native FSDP arithmetic, lifetime and checkpoint contracts."""

import sys

import jittor as jt
import numpy as np
import pytest
from jittor.distributed import checkpoint
from jittor.distributed.fsdp import FullyShardedDataParallel, MixedPrecisionPolicy, fully_shard
from _helpers.native_fsdp import model, optimizer, run_step, equal


@pytest.mark.parametrize("kind", ["sgd", "adamw"])
@pytest.mark.parametrize("accumulate", [1, 2])
def test_complete_gradient_update_accumulation_and_resume(kind, accumulate, tmp_path):
    assert "torch" not in sys.modules and not any(
        n.startswith("jittor.compat") for n in sys.modules
    )
    module = FullyShardedDataParallel(model(jt, native=True))
    opt = optimizer(jt, module, kind)
    baseline = model(jt, native=True)
    reference = optimizer(jt, baseline, kind)
    for step in range(3):
        equal(
            run_step(module, opt, step, accumulate), run_step(baseline, reference, step, accumulate)
        )
        assert all(
            e.full_param is None and getattr(e.owner, e.attr) is e.shard
            for e in module.module._fsdp_state.true_fsdp_params
        )
    directory = tmp_path / "checkpoint"
    checkpoint.save_training_state(module, opt, directory, extra={"step": 3})
    expected = [run_step(module, opt, step, accumulate) for step in range(3, 5)]
    fresh = FullyShardedDataParallel(model(jt, native=True))
    resumed = optimizer(jt, fresh, kind)
    assert checkpoint.load_training_state(fresh, resumed, directory) == {"step": 3}
    for index, want in enumerate(expected, 3):
        equal(run_step(fresh, resumed, index, accumulate), want)


def test_partial_and_pre_sharding_optimizer_are_refused():
    dense = model(jt, native=True)
    stale = optimizer(jt, dense, "sgd")
    module = FullyShardedDataParallel(dense)
    loss = module(jt.ones((3, 5))).sum()
    with pytest.raises(ValueError, match="built after sharding"):
        stale.backward(loss)
    partial = jt.optim.SGD([next(iter(module.parameters()))], lr=0.1)
    with pytest.raises(NotImplementedError, match="every trainable"):
        partial.backward(loss)


def test_pending_forward_backward_and_step_contract():
    module = FullyShardedDataParallel(model(jt, native=True))
    opt = optimizer(jt, module, "sgd")
    with pytest.raises(RuntimeError, match="fresh forward"):
        opt.backward(jt.ones(1).sum())
    loss = module(jt.ones((3, 5))).sum()
    with pytest.raises(RuntimeError, match="before the next forward"):
        module(jt.ones((3, 5)))
    with pytest.raises(RuntimeError, match="backward"):
        opt.step()
    with pytest.raises(NotImplementedError, match="one backward"):
        opt.backward(loss, retain_graph=True)
    opt.backward(loss)
    with pytest.raises(RuntimeError, match="fresh forward"):
        opt.backward(loss)
    opt.step()
    assert all(e.full_param is None for e in module.module._fsdp_state.true_fsdp_params)


def test_failed_forward_restores_bindings_and_releases_full_parameters():
    class Bad(jt.Module):
        def __init__(self):
            self.weight = jt.ones(3).start_grad()

        def execute(self, x):
            raise ArithmeticError("fixture failed")

    module = fully_shard(Bad())
    with pytest.raises(ArithmeticError, match="fixture failed"):
        module(jt.ones(3))
    entry = module._fsdp_state.true_fsdp_params[0]
    assert entry.full_param is None and module.weight is entry.shard
    assert module._fsdp_state.true_fsdp_execute_depth == 0


def test_alias_and_nested_parameters_are_refused():
    class Alias(jt.Module):
        def __init__(self):
            self.first = jt.ones(2).start_grad()
            self.second = self.first

        def execute(self, x):
            return x * self.first

    with pytest.raises(ValueError, match="aliased"):
        fully_shard(Alias())
    inner = fully_shard(model(jt, native=True))
    with pytest.raises(ValueError, match="already"):
        fully_shard(inner)
    outer = jt.nn.Sequential(inner)
    with pytest.raises(ValueError, match="nested"):
        fully_shard(outer)


@pytest.mark.parametrize("field", ["shape", "dtype", "world_size", "shard_numel", "reduce_dtype"])
def test_checkpoint_layout_mismatch_refused_before_model_mutation(field, tmp_path):
    module = FullyShardedDataParallel(model(jt, native=True))
    opt = optimizer(jt, module, "adamw")
    run_step(module, opt, 0)
    directory = tmp_path / "good"
    checkpoint.save_training_state(module, opt, directory)
    state = checkpoint.load(directory)
    spec = next(iter(state["fsdp_layout"].values()))
    spec[field] = {
        "shape": [100],
        "dtype": "float64",
        "world_size": 9,
        "shard_numel": 99,
        "reduce_dtype": "bfloat16",
    }[field]
    from jittor.serialization.sharded import RankLocal

    state["rng"] = RankLocal(state["rng"])
    broken = tmp_path / "broken"
    checkpoint.save(state, broken)
    fresh = FullyShardedDataParallel(model(jt, native=True))
    resumed = optimizer(jt, fresh, "adamw")
    before = {name: value.numpy().copy() for name, value in fresh.state_dict().items()}
    with pytest.raises(ValueError, match="FSDP ownership/layout"):
        checkpoint.load_training_state(fresh, resumed, broken)
    for name, value in fresh.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[name])


def test_cpu_bfloat16_policy_refuses_unvalidated_device():
    with pytest.raises(NotImplementedError, match="requires CUDA"):
        FullyShardedDataParallel(
            model(jt, native=True), mixed_precision=MixedPrecisionPolicy(param_dtype="bfloat16")
        )


def test_public_export_is_lazy_and_native():
    from _helpers.child_process import run_python_child

    result = run_python_child(
        [
            "-c",
            """
import sys
import jittor
assert 'jittor.distributed.fsdp' not in sys.modules
from jittor.nn.parallel import FullyShardedDataParallel
from jittor.distributed.fsdp import FullyShardedDataParallel as canonical
assert FullyShardedDataParallel is canonical
assert canonical.__module__ == 'jittor.distributed.fsdp.api'
assert 'torch' not in sys.modules
assert not any(name.startswith('jittor.compat') for name in sys.modules)
""",
        ]
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_required_cuda_gate_cannot_skip_on_cpu():
    from pathlib import Path
    from _helpers.child_process import run_python_child

    target = Path(__file__).resolve().parents[1] / "backends/comm/nccl/test_native_fsdp_nccl.py"
    result = run_python_child(
        ["-c", "import runpy; runpy.run_path(" + repr(str(target)) + ")"],
        env={"JITTOR_REQUIRE_NATIVE_FSDP": "1", "JT_USE_CUDA": "0", "CUDA_VISIBLE_DEVICES": ""},
    )
    assert result.returncode != 0
    assert "required native FSDP acceptance needs two real CUDA NCCL ranks" in result.stderr


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_each_parameter_zero_update_or_gradient_is_rejected(dtype):
    from _helpers.native_fsdp import assert_oracle_tensor, assert_oracle_update

    gradients = {"weight": np.full((3,), 0.001), "bias": np.full((1,), 0.002)}
    updates = {"weight": np.full((3,), 0.00014), "bias": np.full((1,), 0.0002)}
    for missing in gradients:
        for name, expected in gradients.items():
            actual = np.zeros_like(expected) if name == missing else expected.copy()
            if name == missing:
                with pytest.raises(AssertionError):
                    assert_oracle_tensor(actual, expected, name, dtype)
            else:
                assert_oracle_tensor(actual, expected, name, dtype)
        for name, expected in updates.items():
            actual = np.zeros_like(expected) if name == missing else expected.copy()
            if name == missing:
                with pytest.raises(AssertionError):
                    assert_oracle_update(actual, expected, name, dtype)
            else:
                assert_oracle_update(actual, expected, name, dtype)
    with pytest.raises(AssertionError, match="cannot be distinguished"):
        assert_oracle_update(np.zeros(1), np.full(1, 1e-8), "tiny", dtype)
