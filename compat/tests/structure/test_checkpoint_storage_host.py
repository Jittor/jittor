"""Exercise canonical DCP/native I/O with NumPy and isolated runtime doubles.

These tests do not import a Jittor runtime, JIT, Torch, or an accelerator.
Real CPU/HCCL tensor placement and BF16 device execution are separate gates.
"""

import ast
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_source(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class MemoryStore:
    def __init__(self):
        self.values = {}
        self.condition = threading.Condition()

    def set(self, key, value):
        with self.condition:
            self.values[key] = value
            self.condition.notify_all()

    def get(self, key):
        self.wait([key])
        return self.values[key]

    def wait(self, keys, timeout=5):
        deadline = time.monotonic() + min(float(timeout), 5)
        with self.condition:
            while any(key not in self.values for key in keys):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("test Store missing keys: " + repr(keys))
                self.condition.wait(remaining)


@pytest.fixture
def runtime(monkeypatch):
    jt = types.ModuleType("jittor")
    jt.__path__ = []
    jt.Var = type("NoRuntimeVar", (), {})
    jt.sync_all = lambda *args: None
    dist = types.ModuleType("jittor.distributed")
    current = threading.local()
    dist.get_rank = lambda group=None: getattr(current, "rank", 0)
    dist.get_world_size = lambda group=None: getattr(current, "world", 1)
    dist.get_global_rank = lambda group, rank: rank
    store = MemoryStore()
    dist.get_default_store = lambda: store
    jt.distributed = dist
    monkeypatch.setitem(sys.modules, "jittor", jt)
    monkeypatch.setitem(sys.modules, "jittor.distributed", dist)
    serialization = types.ModuleType("jittor.serialization")
    serialization.__path__ = []
    monkeypatch.setitem(sys.modules, "jittor.serialization", serialization)
    serializer = load_source("jittor.serialization.sharded", ROOT / "python/jittor/serialization/sharded.py", monkeypatch)

    # Use the real checkpoint-only DTensor implementation, without importing
    # its unrelated compiler/device modules.
    parsed = ast.parse((ROOT / "compat/fsdp2/dtensor.py").read_text())
    body = [node for node in parsed.body if isinstance(node, ast.ClassDef)
            and node.name in ("_DTensorMeta", "DTensor")]
    dt = types.ModuleType("host_checkpoint_dtensor")
    dt.np = np
    dt.types = types
    dt.DeviceMesh = lambda *args: object()
    dt.Replicate = lambda: object()
    monkeypatch.setitem(sys.modules, dt.__name__, dt)
    exec(compile(ast.Module(body=body, type_ignores=[]), "dtensor.py", "exec"), vars(dt))

    def workers(world):
        result = []
        for rank in range(world):
            native = load_source("host_native_checkpoint_%d" % rank,
                                 ROOT / "python/jittor/distributed/checkpoint.py", monkeypatch)
            bridge = load_source("host_dcp_bridge_%d" % rank,
                                 ROOT / "compat/torch/checkpoint_storage.py", monkeypatch)
            bridge._native = lambda native=native: native
            result.append((native, bridge))
        def run(function):
            def one(rank):
                current.rank, current.world = rank, world
                return function(rank, result[rank][1])
            with ThreadPoolExecutor(max_workers=world) as executor:
                return list(executor.map(one, range(world)))
        return result, run
    return types.SimpleNamespace(jt=jt, dist=dist, serializer=serializer, dtensor=dt.DTensor,
                                 workers=workers, current=current)


def make_sharded_states(runtime, fill=False):
    states = []
    for rank in range(3):
        model, moments = {}, {}
        for name, start, count in (("a", 0, 3), ("b", 3, 5), ("c", 8, 2)):
            offset = min(max(rank * 4 - start, 0), count)
            size = max(0, min((rank + 1) * 4, start + count) - max(rank * 4, start))
            values = (np.arange(offset, offset + size, dtype=np.float32) + start + 1
                      if fill else np.zeros(size, dtype=np.float32))
            def wrapped(value):
                value = runtime.dtensor(value, object(), (object(),),
                                        checkpoint_shape=(count,), checkpoint_offset=offset)
                value.full_tensor = lambda: pytest.fail("DCP must not gather local shards")
                return value
            model[name] = wrapped(values)
            moments[name] = {"exp_avg": wrapped(values.copy() * .25),
                             "exp_avg_sq": wrapped(values.copy() * .125),
                             "step": np.asarray(2 if fill else 0, dtype=np.float32)}
        states.append({"model": model, "optimizer": {"state": moments,
                       "param_groups": [{"params": ["a", "b", "c"], "lr": .01 if fill else .9}]}})
    return states


def test_three_rank_flat_cross_parameter_and_empty_intervals_round_trip(runtime, tmp_path):
    workers, run = runtime.workers(3)
    saved = make_sharded_states(runtime, fill=True)
    run(lambda rank, bridge: bridge.save(saved[rank], checkpoint_id=tmp_path,
                                          planner=bridge.DefaultSavePlanner()))
    manifest = runtime.serializer.read_manifest(tmp_path)
    assert manifest["world_size"] == 3
    # Group padding is absent; rank 2 owns only the final two real elements.
    assert [manifest["records"][rank]["tensors"]["/model/c"]["count"] for rank in range(3)] == [0, 0, 2]
    targets = make_sharded_states(runtime)
    identities = [[id(value.to_local()) for value in state["model"].values()] for state in targets]
    run(lambda rank, bridge: bridge.load(targets[rank], checkpoint_id=tmp_path,
                                          planner=bridge.DefaultLoadPlanner()))
    for rank, state in enumerate(targets):
        assert identities[rank] == [id(value.to_local()) for value in state["model"].values()]
        for name in ("a", "b", "c"):
            np.testing.assert_array_equal(state["model"][name].to_local(), saved[rank]["model"][name].to_local())
            assert state["model"][name].checkpoint_metadata == saved[rank]["model"][name].checkpoint_metadata
            for field in ("exp_avg", "exp_avg_sq"):
                np.testing.assert_array_equal(state["optimizer"]["state"][name][field].to_local(),
                                              saved[rank]["optimizer"]["state"][name][field].to_local())
            assert float(state["optimizer"]["state"][name]["step"]) == 2
        assert state["optimizer"]["param_groups"][0]["lr"] == .01


def test_corruption_rejects_before_any_target_changes(runtime, tmp_path):
    workers, run = runtime.workers(3)
    saved, targets = make_sharded_states(runtime, True), make_sharded_states(runtime)
    run(lambda rank, bridge: bridge.save(saved[rank], checkpoint_id=tmp_path))
    manifest = runtime.serializer.read_manifest(tmp_path)
    path = tmp_path / manifest["generation"] / "rank-00001" / "tensor-000000.npy"
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(runtime.serializer.CheckpointError, match="checksum mismatch"):
        run(lambda rank, bridge: bridge.load(targets[rank], checkpoint_id=tmp_path))
    for state in targets:
        for value in state["model"].values():
            np.testing.assert_array_equal(value.to_local(), np.zeros_like(value.to_local()))


def test_one_rank_dtype_mismatch_prevents_every_rank_from_mutating(runtime, tmp_path):
    workers, run = runtime.workers(3)
    saved, targets = make_sharded_states(runtime, True), make_sharded_states(runtime)
    run(lambda rank, bridge: bridge.save(saved[rank], checkpoint_id=tmp_path))
    target = targets[1]["model"]["b"]
    targets[1]["model"]["b"] = runtime.dtensor(target.to_local().astype(np.float64), object(), (object(),),
                                               checkpoint_shape=(5,), checkpoint_offset=1)
    with pytest.raises(runtime.serializer.CheckpointError, match="dtype mismatch"):
        run(lambda rank, bridge: bridge.load(targets[rank], checkpoint_id=tmp_path))
    for state in targets:
        for value in state["model"].values():
            assert not value.to_local().any()
        assert state["optimizer"]["param_groups"][0]["lr"] == .9


def test_dense_noncontiguous_tensor_and_scalar_round_trip(runtime, tmp_path):
    workers, run = runtime.workers(1)
    bridge = workers[0][1]
    value = np.arange(12, dtype=np.float32).reshape(3, 4).T
    bridge.save({"value": value, "scalar": 7}, checkpoint_id=tmp_path)
    target = np.zeros((3, 4), dtype=np.float32).T
    state = {"value": target, "scalar": 0}
    bridge.load(state, checkpoint_id=tmp_path)
    assert state["value"] is target and state["scalar"] == 7
    np.testing.assert_array_equal(target, value)


def test_custom_and_pytorch_formats_remain_explicitly_unsupported(runtime, tmp_path):
    workers, run = runtime.workers(1)
    bridge = workers[0][1]
    with pytest.raises(NotImplementedError, match="custom DCP save planners"):
        bridge.save({}, checkpoint_id=tmp_path, planner=object())
    with pytest.raises(NotImplementedError, match="custom DCP storage"):
        bridge.load({}, storage_reader=object())
    with pytest.raises(NotImplementedError, match="per-leaf files"):
        bridge.FileSystemWriter(tmp_path, single_file_per_rank=True)
    (tmp_path / ".metadata").write_bytes(b"not native")
    with pytest.raises(runtime.serializer.CheckpointError, match="PyTorch .metadata/.distcp"):
        bridge.load({}, checkpoint_id=tmp_path)


def test_missing_target_keys_and_shapes_fail_before_copy(runtime, tmp_path):
    workers, run = runtime.workers(1)
    bridge = workers[0][1]
    bridge.save({"first": np.ones(3, dtype=np.float32), "last": np.ones(2, dtype=np.float32)}, checkpoint_id=tmp_path)
    target = np.zeros(3, dtype=np.float32)
    with pytest.raises(runtime.serializer.CheckpointError, match="partial/empty DCP templates"):
        bridge.load({"first": target}, checkpoint_id=tmp_path)
    assert not target.any()
    with pytest.raises(runtime.serializer.CheckpointError, match="global shape mismatch"):
        bridge.load({"first": target, "last": np.zeros(3, dtype=np.float32)}, checkpoint_id=tmp_path)
    assert not target.any()


def test_provider_flat_intervals_use_parameter_relative_offsets(runtime):
    storage_nodes = ast.parse((ROOT / "python/jittor/distributed/fsdp/storage.py").read_text()).body
    overlap = next(node for node in storage_nodes if isinstance(node, ast.FunctionDef) and node.name == "_flat_local_overlap")
    owner = {}
    exec(compile(ast.Module(body=[overlap], type_ignores=[]), "storage.py", "exec"), owner)
    state_nodes = ast.parse((ROOT / "compat/fsdp2/_state_dict.py").read_text()).body
    interval = next(node for node in state_nodes if isinstance(node, ast.FunctionDef) and node.name == "_checkpoint_interval")
    namespace = {"shard": types.SimpleNamespace(_flat_local_overlap=owner["_flat_local_overlap"])}
    exec(compile(ast.Module(body=[interval], type_ignores=[]), "_state_dict.py", "exec"), namespace)
    expected = {"a": [(0, 3), (3, 0), (3, 0)],
                "b": [(0, 1), (1, 4), (5, 0)],
                "c": [(0, 0), (0, 0), (0, 2)]}
    for name, offset, count in (("a", 0, 3), ("b", 3, 5), ("c", 8, 2)):
        entry = types.SimpleNamespace(flat_offset=offset, numel=count)
        for rank in range(3):
            state = types.SimpleNamespace(true_fsdp_flat=True, true_fsdp_rank=rank, true_fsdp_flat_shard_numel=4)
            assert namespace["_checkpoint_interval"](state, entry) == expected[name][rank]


def test_checkpoint_dtensor_metadata_survives_clone_detach_cpu_and_to(runtime):
    class Local:
        shape = (2,)
        def detach(self): return Local()
        def clone(self): return Local()
        def cpu(self): return Local()
        def to(self, *args, **kwargs): return Local()
    original = runtime.dtensor(Local(), object(), (object(),), checkpoint_shape=(3, 4), checkpoint_offset=5)
    for result in (original.detach(), original.clone(), original.cpu(), original.to("cpu")):
        assert result.checkpoint_metadata == ((3, 4), 5)
        assert result.shape == (3, 4)
        assert result.to_local().shape == (2,)
        assert result.to_local() is not original.to_local()


def test_acl_bfloat16_writer_uses_raw_bits_without_float_numpy_export(runtime, tmp_path, monkeypatch):
    raw = np.asarray([0x0000, 0x8000, 0x3F80, 0x7FC1], dtype=np.uint16)
    class BFloat(runtime.jt.Var):
        shape = (4,)
        dtype = "bfloat16"
        def contiguous(self): return self
        def numpy(self): pytest.fail("BF16 numeric numpy export would lose raw payloads")
    class Storage:
        def numpy(self): return raw.copy()
    runtime.jt.core = types.SimpleNamespace(dispatch_context=lambda inputs: ("acl", 5))
    runtime.jt.reinterpret_view = lambda value, shape, dtype: Storage()
    dtypes = types.ModuleType("jittor._core.dtypes")
    dtypes.dtype_name = str
    monkeypatch.setitem(sys.modules, dtypes.__name__, dtypes)
    record = runtime.serializer.write_rank_state({"weight": BFloat()}, tmp_path / "rank-00000", 0, 1, 0)
    runtime.serializer.validate_records([record], [0])
    runtime.serializer.validate_payloads(tmp_path, [record])
    spec = record["tensors"]["/weight"]
    assert spec["backend"] == "acl" and spec["source_device"] == 5
    assert spec["dtype"] == "bfloat16" and spec["storage_dtype"] == np.dtype(np.uint16).str
    np.testing.assert_array_equal(np.load(tmp_path / "rank-00000" / spec["file"], allow_pickle=False), raw)


def test_default_store_bridge_reuses_rendezvous_without_a_second_group(runtime, monkeypatch):
    workers, run = runtime.workers(1)
    native, bridge = workers[0]
    runtime.current.rank, runtime.current.world = 0, 3
    runtime.dist.get_default_store = lambda: None
    state = {"store": None}
    context = types.ModuleType("host_dcp_bridge_0.context")
    context.get_install_context = lambda jt: types.SimpleNamespace(state={"distributed_api": {"state": state}})
    # The source module is loaded under a test-only top-level name. Supply
    # its relative context import without invoking a framework package.
    bridge.__package__ = "host_dcp_bridge_0"
    monkeypatch.setitem(sys.modules, context.__name__, context)
    store = MemoryStore()
    calls = []
    stores = types.ModuleType("jittor.distributed.store")
    def rendezvous(url, rank, world_size):
        calls.append((url, rank, world_size))
        yield store, rank, world_size
    stores.rendezvous = rendezvous
    monkeypatch.setitem(sys.modules, stores.__name__, stores)
    assert bridge._io_context(native, None)[3] is store
    assert bridge._io_context(native, None)[3] is store
    assert state["store"] is store and calls == [("env://", 0, 3)]


def test_actual_provider_model_shards_round_trip_without_gather(runtime):
    class Local(np.ndarray):
        def detach(self): return self
        def clone(self): return self.copy().view(Local)
        def update(self, value): np.copyto(self, value)
        def assign(self, value): np.copyto(self, value)
    def local(value): return np.asarray(value, dtype=np.float32).view(Local)
    storage_nodes = ast.parse((ROOT / "python/jittor/distributed/fsdp/storage.py").read_text()).body
    overlap = next(node for node in storage_nodes if isinstance(node, ast.FunctionDef) and node.name == "_flat_local_overlap")
    namespace = {}
    exec(compile(ast.Module(body=[overlap], type_ignores=[]), "storage.py", "exec"), namespace)
    common = types.SimpleNamespace(
        _flatten_var=lambda value: value.reshape(-1),
        _slice_flat=lambda value, offset, count: value[offset:offset + count],
        _pad_flat=lambda value, count: local(np.pad(value, (0, count - value.size))))
    entries = []
    def refresh(state):
        for entry in state.true_fsdp_params:
            start, count = namespace["_flat_local_overlap"](state, entry)
            entry.shard = state.true_fsdp_flat_shard[start:start + count]
    shard = types.SimpleNamespace(_flat_local_overlap=namespace["_flat_local_overlap"],
                                  _refresh_flat_entry_shards=refresh,
                                  _fsdp_param_entry=lambda value: (None, None))
    provider = {"np": np, "jt": types.SimpleNamespace(concat=lambda values, dim: local(np.concatenate(values))),
                "common": common, "shard": shard, "dtensor": types.SimpleNamespace(DTensor=runtime.dtensor, Shard=lambda dim: object()),
                "_checkpoint_entries": lambda module: entries,
                "_checkpoint_tensor": lambda value, reference, preserve_dtype=False: local(value.copy())}
    names = {"_checkpoint_interval", "_checkpoint_sharded_tensor", "_get_sharded_state_dict",
             "_checkpoint_local_value", "_load_sharded_state_dict"}
    nodes = [node for node in ast.parse((ROOT / "compat/fsdp2/_state_dict.py").read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "_state_dict.py", "exec"), provider)
    for rank in range(3):
        owner = types.SimpleNamespace()
        state = types.SimpleNamespace(true_fsdp_flat=True, true_fsdp_rank=rank, true_fsdp_flat_shard_numel=4,
                                      true_fsdp_flat_shard=local(np.arange(rank * 4, rank * 4 + 4) + 1),
                                      true_fsdp_params=[], true_fsdp_unsharded=False, mesh=object())
        entries.clear()
        for name, offset, count in (("a", 0, 3), ("b", 3, 5), ("c", 8, 2)):
            entry = types.SimpleNamespace(flat_offset=offset, numel=count, shape=(count,), owner=owner, attr=name)
            state.true_fsdp_params.append(entry)
            entries.append((name, state, entry))
        refresh(state)
        for entry in state.true_fsdp_params:
            setattr(owner, entry.attr, entry.shard)
        model = types.SimpleNamespace(state_dict=lambda: {entry.attr: entry.shard for entry in state.true_fsdp_params})
        saved = provider["_get_sharded_state_dict"](model)
        expected = state.true_fsdp_flat_shard.copy()
        if rank == 2:
            expected[2:] = 0  # Collective padding is not persisted.
        state.true_fsdp_flat_shard[:] = 0
        provider["_load_sharded_state_dict"](model, saved)
        np.testing.assert_array_equal(state.true_fsdp_flat_shard, expected)
        assert not state.true_fsdp_unsharded


@pytest.mark.parametrize("failure", ["missing_rank", "missing_payload"])
def test_incomplete_checkpoint_rejects_before_any_target_changes(runtime, tmp_path, failure):
    workers, run = runtime.workers(3)
    saved, targets = make_sharded_states(runtime, True), make_sharded_states(runtime)
    run(lambda rank, bridge: bridge.save(saved[rank], checkpoint_id=tmp_path))
    envelope = json.loads((tmp_path / "manifest.json").read_text())
    manifest = envelope["manifest"]
    if failure == "missing_rank":
        manifest["records"].pop()
        envelope["sha256"] = hashlib.sha256(runtime.serializer.json_bytes(manifest)).hexdigest()
        (tmp_path / "manifest.json").write_bytes(runtime.serializer.json_bytes(envelope))
    else:
        path = tmp_path / manifest["generation"] / "rank-00002" / "tensor-000000.npy"
        path.unlink()
    with pytest.raises(runtime.serializer.CheckpointError, match="incomplete|missing checkpoint tensor payload"):
        run(lambda rank, bridge: bridge.load(targets[rank], checkpoint_id=tmp_path))
    for state in targets:
        for value in state["model"].values():
            assert not value.to_local().any()
