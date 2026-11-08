"""Actual two-rank CUDA ZeRO1 owned allocations and exact checkpoint resume."""

import ctypes
import datetime
import json
import os
from pathlib import Path
import sys

import jittor as jt
from jittor import distributed as dist
from jittor.distributed import ZeroRedundancyOptimizer, checkpoint
from jittor.nn.parallel import DistributedDataParallel
from jittor._runtime.fallback import forbid_backend_fallbacks
import numpy as np
import pytest

from _helpers.native_zero import (model_optimizer, step, numpy_trajectory,
                                 assert_reference, assert_exact, check_bfloat16_resume)


pytestmark = pytest.mark.skipif(
    not jt.compiler.has_cuda or dist.get_world_size() != 2,
    reason="requires two actual CUDA NCCL ranks through the native launcher",
)


def _observe(value):
    value.sync()
    assert jt.core.dispatch_context([value]) == ("cuda", 0)
    assert value.location() == "device"
    _storage(value)


def _storage(value):
    # Inactive frozen moments are lazy zero views, and NumPy checkpoint/trace
    # I/O can leave their storage on host. Materialize their declared placement;
    # the strict observer above separately proves active arithmetic ran on CUDA.
    assert jt.core.dispatch_context([value]) == ("cuda", 0)
    ordinal = ctypes.c_int(-1)
    assert ctypes.CDLL("libcuda.so.1").cuPointerGetAttribute(
        ctypes.byref(ordinal), 9, ctypes.c_void_p(value.device_raw_ptr)) == 0
    assert ordinal.value == 0
    assert value.location() == "device"


def _context():
    assert jt.flags.use_cuda and "torch" not in sys.modules
    if dist.get_default_store() is None:
        dist.init_process_group("nccl", init_method="env://", timeout=datetime.timedelta(seconds=60))
    assert dist.get_backend() == "nccl"
    return dist.get_rank(), Path(os.environ["JITTOR_ZERO_TEST_DIR"])


@pytest.mark.parametrize("algorithm", ["Adam", "AdamW", "SGD"])
@pytest.mark.parametrize("ddp,frozen,single", [(False, False, False), (True, False, False),
                                              (True, True, False), (False, False, True),
                                              (False, True, False)])
def test_owned_allocation_math_and_fresh_resume(monkeypatch, algorithm, ddp, frozen, single):
    rank, root = _context()
    before = jt.core.backend_fallback_count()
    allocations = []
    errors = {"parameters": 0.0, "gradients": 0.0, "owned_moments": 0.0, "loss": 0.0}

    def record_errors(trace, reference):
        for name, expected in reference["model"].items():
            got = trace["model"][("module." if ddp else "") + name]
            errors["parameters"] = max(errors["parameters"], float(np.max(np.abs(got - expected))))
        for got, expected in zip(trace["gradients"], reference["gradients"]):
            errors["gradients"] = max(errors["gradients"], float(np.max(np.abs(got - expected))))
        for gi, group in enumerate(trace["local"]):
            for name, values in group.items():
                source = "v" if name == "values" and "m" in group else "m"
                for got in values:
                    errors["owned_moments"] = max(errors["owned_moments"],
                        float(np.max(np.abs(got - reference[source][gi]))))
        errors["loss"] = max(errors["loss"], abs(trace["loss"] - reference["losses"][rank]))
    module = jt.optim.algorithms.sgd if algorithm == "SGD" else jt.optim.algorithms.adam
    original_buffer = module._state_buffer

    def buffer(parameter):
        allocations.append((id(parameter), int(parameter.numel())))
        return original_buffer(parameter)

    with forbid_backend_fallbacks():
        with monkeypatch.context() as probe:
            probe.setattr(module, "_state_buffer", buffer)
            model, optimizer = model_optimizer(algorithm, frozen=frozen, single=single)
        partition = optimizer.state_partition()
        owned = [p for group in optimizer.local_optimizer.param_groups for p in group["params"]]
        assert {identity for identity, _ in allocations} == {id(p) for p in owned}
        multiplier = 1 if algorithm == "SGD" else 2
        assert len(allocations) == len(owned) * multiplier
        assert len(owned) == (0 if single and rank else 1)
        assert all("m" not in group and "values" not in group for group in optimizer.param_groups)
        local_numel = sum(size for _, size in allocations)
        full_numel = sum(int(p.numel()) for p in model.parameters()) * multiplier
        assert local_numel == (6 if rank == 0 else 0 if single else 2) * multiplier
        if ddp:
            model = DistributedDataParallel(model)
        references = numpy_trajectory(algorithm, world=2, frozen=frozen, single=single)
        reduce_calls = []
        original_reduce = dist.all_reduce

        def reduce(value, op="mean", **kwargs):
            assert op == "mean"
            if tuple(value.shape) in ((2, 3), (2,)):
                reduce_calls.append(tuple(value.shape))
            return original_reduce(value, op=op, **kwargs)

        with monkeypatch.context() as counter:
            counter.setattr(dist, "all_reduce", reduce)
            for index in range(3):
                actual = step(model, optimizer, rank, accumulate=2, observer=_observe,
                              storage_observer=_storage, ddp_accumulate=ddp)
                assert_reference(actual, references[index], optimizer, rank, "module." if ddp else "")
                record_errors(actual, references[index])
        trainable = 1 if single or frozen else 2
        assert len(reduce_calls) == 3 * trainable * (1 if ddp else 2)
        directory = root / "{}-ddp{}-frozen{}-single{}".format(algorithm, ddp, frozen, single)
        checkpoint.save_training_state(model, optimizer, directory, extra={"step": 3})
        manifest = checkpoint.read_manifest(directory)
        # Every rank writes its owned state directly; there is no full-state
        # creation/gather followed by an I/O slice masquerading as training sharding.
        for record in manifest["records"]:
            owned_elements = sum(spec["count"] for path, spec in record["tensors"].items()
                                 if "/zero_state/" in path and ("/m/" in path or "/values/" in path))
            expected_numel = (6 if record["rank"] == 0 else 0 if single else 2) * multiplier
            assert owned_elements == expected_numel
        expected = []
        for index in range(3, 5):
            actual = step(model, optimizer, rank, accumulate=2, observer=_observe,
                          storage_observer=_storage, ddp_accumulate=ddp)
            assert_reference(actual, references[index], optimizer, rank, "module." if ddp else "")
            record_errors(actual, references[index])
            expected.append(actual)
        # Existing native DDP fixes trainability at construction. Standalone
        # loading also exercises restore into initially opposite flags.
        fresh, resumed = model_optimizer(algorithm, frozen=frozen if ddp else not frozen, single=single)
        if ddp:
            fresh = DistributedDataParallel(fresh)
        assert checkpoint.load_training_state(fresh, resumed, directory) == {"step": 3}
        for want in expected:
            assert_exact(step(fresh, resumed, rank, accumulate=2, observer=_observe,
                              storage_observer=_storage, ddp_accumulate=ddp), want)
        checkpoint.load_training_state(model, optimizer, directory)
        for want in expected:
            assert_exact(step(model, optimizer, rank, accumulate=2, observer=_observe,
                              storage_observer=_storage, ddp_accumulate=ddp), want)
    assert jt.core.backend_fallback_count() == before == 0
    assert "torch" not in sys.modules
    print("NATIVE_ZERO_RESULT " + json.dumps({"rank": rank, "algorithm": algorithm,
          "ddp": ddp, "frozen": frozen, "empty_rank": single and rank == 1,
          "owned_state_numel": local_numel, "replicated_state_numel": full_numel,
          "reduction_calls_three_accumulated_steps": len(reduce_calls),
          "partition": partition, "max_absolute_reference_error": errors,
          "exact_resume": True, "fallback_count": 0}), flush=True)


def test_rank_configuration_mismatch_fails_before_collectives():
    rank, _ = _context()
    parameter = jt.ones((2, 3)).start_grad()
    with pytest.raises(ValueError, match="differs across ranks"):
        ZeroRedundancyOptimizer([parameter], jt.optim.Adam, lr=0.01 + rank * 0.01)


@pytest.mark.parametrize("algorithm", ["Adam", "AdamW", "SGD"])
def test_standalone_late_unfreeze(algorithm):
    rank, _ = _context()
    with forbid_backend_fallbacks():
        model, optimizer = model_optimizer(algorithm, frozen=True)
        references = numpy_trajectory(algorithm, world=2, frozen_until=2)
        for index in range(5):
            if index == 2:
                model.bias.start_grad()
            actual = step(model, optimizer, rank, accumulate=2, observer=_observe, storage_observer=_storage)
            assert_reference(actual, references[index], optimizer, rank)
    assert jt.core.backend_fallback_count() == 0


def test_bfloat16_owned_state_checkpoint_resume():
    rank, root = _context()
    with forbid_backend_fallbacks():
        result = check_bfloat16_resume(root / "bfloat16-zero", observer=_observe)
    assert result["owned_bfloat16_state_numel"] == (2 if rank == 0 else 0)
    assert jt.core.backend_fallback_count() == 0
    print("NATIVE_ZERO_BFLOAT16_RESULT " + json.dumps(result), flush=True)


@pytest.mark.parametrize("algorithm", ["Adam", "AdamW", "SGD"])
def test_group_override_removal_and_changed_default_resume(algorithm):
    rank, root = _context()
    directory = root / ("group-default-" + algorithm)
    references = numpy_trajectory(algorithm, world=2,
                                 group_lrs=[[0.01, 0.02]] + [[0.005, 0.02]] * 4)
    with forbid_backend_fallbacks():
        model, optimizer = model_optimizer(algorithm)
        expected = []
        for index in range(5):
            if index == 1:
                del optimizer.param_groups[0]["lr"]
                optimizer.lr = 0.005
            trace = step(model, optimizer, rank, observer=_observe)
            assert_reference(trace, references[index], optimizer, rank)
            if index == 2:
                checkpoint.save_training_state(model, optimizer, directory)
            if index > 2:
                expected.append(trace)
        fresh, resumed = model_optimizer(algorithm)
        checkpoint.load_training_state(fresh, resumed, directory)
        assert "lr" not in resumed.param_groups[0]
        assert "lr" not in resumed.local_optimizer.param_groups[0]
        for want in expected:
            assert_exact(step(fresh, resumed, rank, observer=_observe), want)
    assert jt.core.backend_fallback_count() == 0


def test_tied_parameters_and_no_sync_step_guard():
    rank, _ = _context()

    class Tied(jt.Module):
        def __init__(self):
            self.weight = jt.array([0.25], dtype="float32").start_grad()
            self.alias = self.weight

        def execute(self, value):
            return (self.weight + self.alias).sum() * value

    model = Tied()
    optimizer = ZeroRedundancyOptimizer(model.parameters(), jt.optim.SGD, lr=0.125, fused=False)
    assert len(optimizer.local_optimizer.param_groups[0]["values"]) == (1 if rank == 0 else 0)
    wrapped = DistributedDataParallel(model)
    with forbid_backend_fallbacks():
        with wrapped.no_sync():
            optimizer.backward(wrapped(float(rank + 1)))
        with pytest.raises(RuntimeError, match="still local"):
            optimizer.step()
        optimizer.zero_grad()
        optimizer.step(wrapped(float(rank + 1)))
        _observe(model.weight)
        np.testing.assert_array_equal(model.weight.numpy(), [-0.125])
        assert model.weight is model.alias
    assert jt.core.backend_fallback_count() == 0
