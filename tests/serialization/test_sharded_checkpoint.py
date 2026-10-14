"""Native checkpoint storage and deterministic numerical continuation."""

import copy
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import random
import socket
import sys

import jittor as jt
from jittor.distributed import checkpoint
from jittor.serialization.sharded import (
    FORMAT, VERSION, json_bytes, read_manifest, validate_records, write_rank_state,
)
import numpy as np
import pytest

from _helpers.native_checkpoint import (
    assert_trace, check_bfloat16_training_resume, model_optimizer,
    numpy_trajectory, training_step,
)


@pytest.mark.parametrize("restore_existing", [False, True])
def test_native_training_resume_and_rng(tmp_path, restore_existing):
    assert "torch" not in sys.modules
    model, optimizer = model_optimizer()
    reference = numpy_trajectory(5)
    for step in range(3):
        assert_trace(training_step(model, optimizer), reference[step])
    random.seed(19)
    np.random.seed(23)
    jt.set_seed(29)
    jt.rand(7).sync()
    flags = [parameter.is_stop_grad() for parameter in model.parameters()]
    checkpoint.save_training_state(model, optimizer, tmp_path, extra={"epoch": 3})
    assert [parameter.is_stop_grad() for parameter in model.parameters()] == flags
    next_python = [random.random() for _ in range(4)]
    next_numpy = np.random.random(11)
    next_native = jt.rand(13).numpy().copy()
    expected = [training_step(model, optimizer) for _ in range(2)]
    fresh, resumed = (model, optimizer) if restore_existing else model_optimizer()
    fresh.weight.update(jt.full((2, 3), -9.0))
    resumed.lr = 0.7
    assert checkpoint.load_training_state(fresh, resumed, tmp_path) == {"epoch": 3}
    assert all(not parameter.is_stop_grad() for parameter in fresh.parameters())
    assert resumed.lr == 0.01
    assert [random.random() for _ in range(4)] == next_python
    np.testing.assert_array_equal(np.random.random(11), next_numpy)
    np.testing.assert_array_equal(jt.rand(13).numpy(), next_native)
    for index in range(2):
        got = training_step(fresh, resumed)
        assert_trace(got, expected[index], exact=True)
        assert_trace(got, reference[index + 3])
    assert "torch" not in sys.modules


@pytest.mark.parametrize("restore_existing", [False, True])
def test_training_checkpoint_preserves_frozen_parameter_flags(tmp_path, restore_existing):
    model, optimizer = model_optimizer()
    model.bias.stop_grad()
    reference = numpy_trajectory(5, frozen_bias=True)
    for index in range(3):
        assert_trace(training_step(model, optimizer), reference[index])
    before_flags = [value.is_stop_grad() for value in model.parameters()]
    checkpoint.save_training_state(model, optimizer, tmp_path)
    assert [value.is_stop_grad() for value in model.parameters()] == before_flags
    expected = [training_step(model, optimizer) for _ in range(2)]
    fresh, resumed = (model, optimizer) if restore_existing else model_optimizer()
    checkpoint.load_training_state(fresh, resumed, tmp_path)
    assert not fresh.weight.is_stop_grad()
    assert fresh.bias.is_stop_grad()
    for index in range(2):
        trace = training_step(fresh, resumed)
        assert_trace(trace, expected[index], exact=True)
        assert_trace(trace, reference[index + 3])


def test_pending_accumulation_refuses_before_writing_checkpoint(tmp_path):
    model, optimizer = model_optimizer()
    optimizer.backward(model(jt.ones((4, 3))).sum())
    with pytest.raises(NotImplementedError, match="completed optimizer step"):
        checkpoint.save_training_state(model, optimizer, tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_bfloat16_storage_preserves_original_dtype_and_every_bit(tmp_path):
    bits = np.array([0, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80,
                     0x7fc1, 1, 0x7f7f, 0x8080], dtype=np.uint16)
    source = jt.reinterpret_view(jt.array(bits, dtype="uint16"), (10,), "bfloat16")
    checkpoint.save({"tensor": source}, tmp_path)
    restored = checkpoint.load(tmp_path)["tensor"]
    assert str(restored.dtype) == "bfloat16"
    np.testing.assert_array_equal(jt.reinterpret_view(restored, restored.shape, "uint16").numpy(), bits)


def test_bfloat16_parameter_and_momentum_training_resume(tmp_path):
    check_bfloat16_training_resume(tmp_path)


def test_corrupt_payload_refuses_before_model_or_rng_mutation(tmp_path):
    model, optimizer = model_optimizer()
    training_step(model, optimizer)
    checkpoint.save_training_state(model, optimizer, tmp_path)
    fresh, resumed = model_optimizer()
    before = fresh.weight.numpy().copy()
    before_rng = jt.core.rng_state("cpu")
    payload = next(tmp_path.glob("*/rank-00000/tensor-*.npy"))
    data = bytearray(payload.read_bytes())
    data[-1] ^= 1
    payload.write_bytes(data)
    with pytest.raises(checkpoint.CheckpointError, match="checksum"):
        checkpoint.load_training_state(fresh, resumed, tmp_path)
    np.testing.assert_array_equal(fresh.weight.numpy(), before)
    assert jt.core.rng_state("cpu") == before_rng


def test_optimizer_fqn_order_is_checked_before_model_mutation(tmp_path):
    model, optimizer = model_optimizer()
    training_step(model, optimizer)
    checkpoint.save_training_state(model, optimizer, tmp_path)
    fresh, _ = model_optimizer()
    wrong = jt.optim.Adam(list(reversed(fresh.parameters())), lr=0.01)
    before = fresh.weight.numpy().copy()
    with pytest.raises(ValueError, match="FQN/group"):
        checkpoint.load_training_state(fresh, wrong, tmp_path)
    np.testing.assert_array_equal(fresh.weight.numpy(), before)


@pytest.mark.parametrize("invalid", ["fqn", "flags", "shape", "dtype", "moment", "rng", "rng_backend"])
def test_training_preflight_refuses_invalid_state_without_mutating_target(tmp_path, invalid):
    model, optimizer = model_optimizer()
    training_step(model, optimizer)
    checkpoint.save_training_state(model, optimizer, tmp_path / "source")
    state = checkpoint.load(tmp_path / "source")
    if invalid == "fqn":
        del state["model"]["bias"]
    elif invalid == "flags":
        state["model_stop_grad"]["weight"] = 1
    elif invalid == "shape":
        state["model"]["weight"] = jt.zeros((1, 3))
    elif invalid == "dtype":
        state["model"]["weight"] = state["model"]["weight"].float64()
    elif invalid == "moment":
        state["optimizer"]["state"]["defaults"]["param_groups"][0]["m"][0] = jt.zeros((1, 3))
    elif invalid == "rng":
        state["rng"]["numpy"] = ("invalid generator",) + state["rng"]["numpy"][1:]
    else:
        state["rng"]["cuda"] = [(1, 0)]
    checkpoint.save(state, tmp_path / "invalid")
    fresh, resumed = model_optimizer()
    fresh.bias.stop_grad()
    resumed.lr = 0.7
    before = {name: value.numpy().copy() for name, value in fresh.state_dict().items()}
    before_flags = [value.is_stop_grad() for value in fresh.parameters()]
    before_rng = jt.core.rng_state("cpu")
    before_python, before_numpy = random.getstate(), np.random.get_state()
    with pytest.raises(ValueError):
        checkpoint.load_training_state(fresh, resumed, tmp_path / "invalid")
    for name, value in fresh.state_dict().items():
        np.testing.assert_array_equal(value.numpy(), before[name])
    assert [value.is_stop_grad() for value in fresh.parameters()] == before_flags
    assert resumed.lr == 0.7
    assert jt.core.rng_state("cpu") == before_rng
    assert random.getstate() == before_python
    after_numpy = np.random.get_state()
    assert after_numpy[0] == before_numpy[0] and after_numpy[2:] == before_numpy[2:]
    np.testing.assert_array_equal(after_numpy[1], before_numpy[1])


def test_requested_local_interval_rejects_wrong_dtype_or_tensor_owner(tmp_path):
    checkpoint.save({"value": jt.arange(12).float32()}, tmp_path)
    for tensor in (jt.empty((3,), dtype="float64"), np.empty(3, dtype=np.float32)):
        target = checkpoint.TensorShard(tensor, (12,), 2)
        with pytest.raises(checkpoint.CheckpointError, match="dtype mismatch|type mismatch"):
            checkpoint.load(tmp_path, tensor_shards={"/value": target})


def test_numpy_local_interval_preserves_non_native_byte_order(tmp_path):
    source = np.arange(6, dtype=">f4")
    checkpoint.save({"value": source}, tmp_path)
    target = checkpoint.TensorShard(np.empty(2, dtype=">f4"), (6,), 2)
    restored = checkpoint.load(tmp_path, tensor_shards={"/value": target})["value"].tensor
    assert restored.dtype.str == source.dtype.str
    assert restored.tobytes() == source[2:4].tobytes()


def test_failed_write_has_no_complete_marker_and_failed_overwrite_preserves_old(tmp_path):
    incomplete = tmp_path / "incomplete"
    with pytest.raises(checkpoint.CheckpointError, match="unsupported checkpoint value"):
        checkpoint.save({"tensor": jt.arange(9), "bad": object()}, incomplete)
    assert not (incomplete / "manifest.json").exists()
    with pytest.raises(checkpoint.CheckpointError, match="no complete"):
        checkpoint.load(incomplete)
    complete = tmp_path / "complete"
    checkpoint.save({"tensor": jt.arange(9)}, complete)
    before = (complete / "manifest.json").read_bytes()
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.save({"bad": object()}, complete, overwrite=True)
    assert (complete / "manifest.json").read_bytes() == before
    np.testing.assert_array_equal(checkpoint.load(complete)["tensor"].numpy(), np.arange(9))


def test_real_tcp_store_status_exchange_uses_portable_timeout_contract():
    from jittor.distributed import TCPStore

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    master = TCPStore("127.0.0.1", port, 2, True, timeout=5, wait_for_workers=False)
    peer = TCPStore("127.0.0.1", master.port, 2, False, timeout=5, wait_for_workers=False)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(checkpoint._exchange, store, "checkpoint-test", rank, 2,
                                   {"rank": rank}, 5)
                       for rank, store in enumerate((master, peer))]
            assert [future.result(timeout=10) for future in futures] == [
                [{"rank": 0}, {"rank": 1}], [{"rank": 0}, {"rank": 1}]]
        master.set("generation", checkpoint.json_bytes({"generation": "test"}))
        assert checkpoint._get_json(peer, "generation", 5) == {"generation": "test"}
    finally:
        peer.close()
        master.close()


def test_real_local_shard_files_reassemble_and_reshard_with_overlap_gap_rejection(tmp_path):
    generation = "a" * 32
    root = tmp_path / generation
    root.mkdir()
    records = [write_rank_state(
        {"weight": checkpoint.TensorShard(np.arange(rank * 6, (rank + 1) * 6,
                                                    dtype=np.float64), (3, 4), rank * 6)},
        root / "rank-{:05d}".format(rank), rank, 2, rank) for rank in range(2)]
    validate_records(records, [0, 1])
    manifest = {"format": FORMAT, "version": VERSION, "generation": generation,
                "world_size": 2, "ranks": [0, 1], "records": records}
    envelope = {"manifest": manifest, "sha256": hashlib.sha256(json_bytes(manifest)).hexdigest()}
    (tmp_path / "manifest.json").write_bytes(json_bytes(envelope))
    full = checkpoint.load(tmp_path)["weight"]
    np.testing.assert_array_equal(full, np.arange(12).reshape(3, 4))
    target = checkpoint.TensorShard(np.empty(5), (3, 4), 4)
    shard = checkpoint.load(tmp_path, tensor_shards={"/weight": target})["weight"]
    np.testing.assert_array_equal(shard.tensor, np.arange(4, 9))
    for offset, count in ((5, 6), (7, 5)):
        invalid = copy.deepcopy(records)
        invalid[1]["tensors"]["/weight"].update(offset=offset, count=count)
        with pytest.raises(checkpoint.CheckpointError, match="overlap or gap"):
            validate_records(invalid, [0, 1])
    assert read_manifest(tmp_path)["world_size"] == 2
