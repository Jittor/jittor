"""Run under jtrun: actual native CUDA/DDP checkpoint continuation."""

import datetime
import ctypes
import json
import os
from pathlib import Path
import random
import sys

import jittor as jt
from jittor import distributed as dist
from jittor.distributed import checkpoint
from jittor.nn.parallel import DistributedDataParallel
from jittor._runtime.fallback import forbid_backend_fallbacks
import numpy as np
import pytest

from _helpers import native_checkpoint as case


pytestmark = pytest.mark.skipif(
    not jt.compiler.has_cuda or dist.get_world_size() != 2,
    reason="requires two real CUDA NCCL ranks via the native launcher",
)


def _pointer_device(value):
    ordinal = ctypes.c_int(-1)
    driver = ctypes.CDLL("libcuda.so.1")
    assert driver.cuPointerGetAttribute(ctypes.byref(ordinal), 9,
        ctypes.c_void_p(value.device_raw_ptr)) == 0
    assert ordinal.value == 0


def _observe_cuda(value):
    # Observe computed buffers before numpy() migrates them to host storage.
    value.sync()
    assert jt.core.dispatch_context([value]) == ("cuda", 0)
    assert value.location() == "device"
    _pointer_device(value)


def _device_state(model, optimizer):
    tensors = list(model.parameters()) + [
        value for group in optimizer.param_groups for field in ("m", "values", "grads")
        for value in group.get(field, ())]
    for value in tensors:
        value.sync()
        assert jt.core.dispatch_context([value]) == ("cuda", 0)
        # Checkpoint/trace NumPy snapshots are explicit host I/O. Materialize
        # the tensor's current native CUDA placement, then ask the driver.
        _pointer_device(value)
        assert value.location() == "device"


def _random_values():
    python = [random.random() for _ in range(4)]
    numpy = np.random.random(11)
    with jt.flag_scope(use_cuda=0):
        cpu = jt.rand(13)
        assert jt.core.dispatch_context([cpu])[0] == "cpu"
        cpu = cpu.numpy().copy()
    cuda = jt.rand(15)
    assert jt.core.dispatch_context([cuda]) == ("cuda", 0)
    return python, numpy, cpu, cuda.numpy().copy()


def _same_random(actual, expected):
    assert actual[0] == expected[0]
    for got, want in zip(actual[1:], expected[1:]):
        np.testing.assert_array_equal(got, want)


def test_native_ddp_sharded_model_optimizer_rng_resume():
    assert "torch" not in sys.modules
    assert jt.flags.use_cuda
    if dist.get_default_store() is None:
        dist.init_process_group("nccl", init_method="env://",
                                timeout=datetime.timedelta(seconds=60))
    assert dist.get_backend() == "nccl"
    rank = dist.get_rank()
    root = Path(os.environ["JITTOR_CHECKPOINT_TEST_DIR"])
    before_fallback = jt.core.backend_fallback_count()
    reference_errors = {"parameters": 0.0, "moments": 0.0, "gradients": 0.0, "loss": 0.0}

    def record_reference_errors(trace, expected):
        for name, value in expected["model"].items():
            reference_errors["parameters"] = max(reference_errors["parameters"],
                float(np.max(np.abs(trace["model"]["module." + name] - value))))
        for field, destination in (("m", "moments"), ("v", "moments"), ("gradients", "gradients")):
            for actual, want in zip(trace[field], expected[field]):
                reference_errors[destination] = max(reference_errors[destination],
                    float(np.max(np.abs(actual - want))))
        reference_errors["loss"] = max(reference_errors["loss"],
            abs(trace["loss"] - expected["losses"][rank]))

    with forbid_backend_fallbacks():
        model, optimizer = case.model_optimizer()
        if rank:
            model.weight.update(model.weight + 0.17)
        model = DistributedDataParallel(model)
        reference = case.numpy_trajectory(5, world_size=2)
        for index in range(3):
            trace = case.training_step(model, optimizer, rank, observer=_observe_cuda)
            case.assert_trace(trace, reference[index], rank=rank, model_prefix="module.")
            record_reference_errors(trace, reference[index])
            _device_state(model, optimizer)
        random.seed(43 + rank)
        np.random.seed(47 + rank)
        jt.set_seed(53 + rank)
        with jt.flag_scope(use_cuda=0):
            jt.rand(7).sync()
        jt.rand(9).sync()
        checkpoint.save_training_state(model, optimizer, root / "training", extra={"epoch": 3})
        manifest = checkpoint.read_manifest(root / "training")
        assert manifest["world_size"] == 2
        assert manifest["ranks"] == [0, 1]
        for record in manifest["records"]:
            for spec in record["tensors"].values():
                if spec["role"] == "replicated":
                    assert spec["count"] < int(np.prod(spec["shape"]))
        expected_random = _random_values()
        expected = [case.training_step(model, optimizer, rank, observer=_observe_cuda) for _ in range(2)]
        for existing in (True, False):
            if existing:
                fresh, resumed = model, optimizer
            else:
                fresh, resumed = case.model_optimizer()
                fresh = DistributedDataParallel(fresh)
            assert checkpoint.load_training_state(fresh, resumed, root / "training") == {"epoch": 3}
            _same_random(_random_values(), expected_random)
            for index in range(2):
                actual = case.training_step(fresh, resumed, rank, observer=_observe_cuda)
                case.assert_trace(actual, expected[index], exact=True)
                case.assert_trace(actual, reference[index + 3], rank=rank, model_prefix="module.")
                record_reference_errors(actual, reference[index + 3])
                _device_state(fresh, resumed)

        # Actual local inputs have different values and only rank-sized storage.
        local = jt.array(np.arange(rank * 7, (rank + 1) * 7, dtype=np.float32), dtype="float32")
        _pointer_device(local)
        checkpoint.save({"weight": checkpoint.TensorShard(local, (2, 7), rank * 7)}, root / "local")
        full = checkpoint.load(root / "local")["weight"]
        _pointer_device(full)
        np.testing.assert_array_equal(full.numpy(), np.arange(14).reshape(2, 7))
        target = checkpoint.TensorShard(jt.empty((5,), dtype="float32"), (2, 7), rank * 5 + 2)
        restored = checkpoint.load(root / "local", tensor_shards={"/weight": target})["weight"]
        _pointer_device(restored.tensor)
        np.testing.assert_array_equal(restored.tensor.numpy(), np.arange(rank * 5 + 2, rank * 5 + 7))

        # Metadata mismatch and a one-rank writer error must fail on both peers.
        with pytest.raises(checkpoint.CheckpointError, match="differs across ranks"):
            checkpoint.save({"epoch": rank, "weight": local}, root / "mismatch")
        assert not (root / "mismatch/manifest.json").exists()
        with pytest.raises(checkpoint.CheckpointError, match="rank writes failed"):
            checkpoint.save({"weight": local, "value": object() if rank else 3}, root / "one-rank-error")
        assert not (root / "one-rank-error/manifest.json").exists()

        bits = np.array([0, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1, 1], dtype=np.uint16)
        bf16 = jt.reinterpret_view(jt.array(bits, dtype="uint16"), (8,), "bfloat16")
        _pointer_device(bf16)
        checkpoint.save({"bf16": bf16}, root / "bf16")
        loaded = checkpoint.load(root / "bf16")["bf16"]
        assert str(loaded.dtype) == "bfloat16"
        assert jt.core.dispatch_context([loaded]) == ("cuda", 0)
        _pointer_device(loaded)
        np.testing.assert_array_equal(jt.reinterpret_view(loaded, loaded.shape, "uint16").numpy(), bits)
        bf16_model, bf16_optimizer = case.check_bfloat16_training_resume(root / "bf16-training")
        _device_state(bf16_model, bf16_optimizer)
    assert jt.core.backend_fallback_count() == before_fallback == 0
    assert "torch" not in sys.modules
    version = ctypes.c_int()
    assert ctypes.CDLL("libnccl.so.2").ncclGetVersion(ctypes.byref(version)) == 0
    print("NATIVE_CHECKPOINT_RESULT", json.dumps({"rank": rank, "world_size": 2,
          "backend": "nccl", "device": "cuda:0", "numpy_reference_steps": 5,
          "exact_resume": True, "four_rng": True, "bf16_raw_bits": True,
          "bf16_parameter_training_resume": True,
          "nccl_version": version.value, "numpy_max_abs_error": reference_errors,
          "fallbacks": 0, "torch_imported": False}), flush=True)
