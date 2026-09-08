"""Serialization owners preserve values and enforce restricted loading."""
import ast
import importlib
import inspect
import io
import pickle
import pickletools
import textwrap
import types
import zipfile
from types import MappingProxyType

import numpy as np
import pytest
import jittor as jt
from jittor.compat.torch.context import get_install_context, ModuleRegistry
from jittor.compat.torch.tensor_state import compatibility_owner
from jittor.compat.torch import serialization
from jittor.compat.torch.serialization import portable, safetensors as safe_owner, torch_archive


def test_cuda_checkpoint_mapped_to_cpu_stays_on_cpu_for_subsequent_ops():
    import jittor as jt
    import torch
    if not jt.has_cuda or not jt.flags.use_cuda:
        pytest.skip("requires active CUDA runtime")
    previous = torch.get_default_device()
    try:
        torch.set_default_device("cuda")
        source = torch.tensor([2**40 + 1], dtype=torch.int64, device="cuda")
        stream = io.BytesIO()
        torch.save({"value": source}, stream)
        stream.seek(0)
        restored = torch.load(stream, weights_only=True, map_location="cpu")["value"]
        assert type(restored) is torch.Tensor
        assert restored.device.type == "cpu" and restored.placement_backend == 0
        restored.sync()
        assert restored.location() == "cpu" and restored.device_id == -1
        result = restored + 1
        result.sync()
        assert result.location() == "cpu"
        assert restored.location() == "cpu"
        np.testing.assert_array_equal(result.numpy(), [2**40 + 2])
        assert source.location() == "device"
    finally:
        torch.set_default_device(previous)

_EXECUTED = []


def _payload():
    _EXECUTED.append(True)
    return 42


class _UnsafeObject:
    def __reduce__(self):
        return _payload, ()


def _fragment(value):
    return pickletools.optimize(pickle.dumps(value, protocol=2))[2:-1]


def _torch_archive(stride=(1, 3), offset=0):
    # Build the documented persistent-storage pickle without importing a
    # second tensor framework or invoking any reconstruction while writing.
    persistent = (b"(" + _fragment("storage") + b"ctorch\nDoubleStorage\n" +
                  _fragment("0") + _fragment("cpu") + _fragment(6) + b"tQ")
    data = (b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(" + persistent +
            _fragment(offset) + _fragment((3, 2)) + _fragment(stride) + b"\x89NtR.")
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("fixture/data.pkl", data)
        archive.writestr("fixture/data/0", np.arange(6, dtype=np.float64).tobytes())
    return stream.getvalue()


def test_serialization_objects_have_real_module_owners():
    torch = compatibility_owner(jt)
    assert torch.load is portable.load and torch.save is portable.save
    for value in (torch.load, torch.save, safe_owner._PySafeOpen,
                  safe_owner._PySafeSlice, torch_archive._ArchiveUnpickler):
        assert "<locals>" not in value.__qualname__
        assert getattr(importlib.import_module(value.__module__), value.__name__) is value
        assert pickle.loads(pickle.dumps(value)) is value
    for function in (serialization.install, safe_owner._install_safetensors_shim):
        node = ast.parse(textwrap.dedent(inspect.getsource(function))).body[0]
        assert not [child for child in ast.walk(node) if child is not node and
                    isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Lambda))]


@pytest.mark.parametrize("destination", ("file", "bytes"))
def test_portable_roundtrip_dtype_parameter_view_and_map_location(tmp_path, destination):
    torch = compatibility_owner(jt)
    base = torch.tensor(np.arange(6, dtype=np.float64).reshape(2, 3))
    view = base.transpose(0, 1)
    before = (view.data_ptr(), tuple(view.stride()))
    parameter = torch.nn.Parameter(view)
    values = {"view": view, "parameter": parameter,
              "wide": torch.tensor([2**45, 2**45 + 1], dtype=torch.int64),
              "bf16": torch.tensor(3.25, dtype=torch.bfloat16), "dtype": torch.float64}
    output = tmp_path / "weights.pkl" if destination == "file" else io.BytesIO()
    torch.save(values, output)
    assert (view.data_ptr(), tuple(view.stride())) == before
    if destination == "bytes":
        output.seek(0)
    seen = []
    def location(value, source):
        seen.append(source)
        return None
    restored = torch.load(output, map_location=location)
    assert restored["dtype"] is torch.float64
    assert isinstance(restored["parameter"], torch.nn.Parameter)
    assert restored["parameter"].requires_grad
    for name in ("view", "parameter", "wide", "bf16"):
        assert restored[name].dtype is values[name].dtype
        np.testing.assert_array_equal(restored[name].numpy(), values[name].numpy())
    assert tuple(restored["bf16"].shape) == ()
    assert seen == ["cpu"] * 4
    parameter.requires_grad_(False)
    restored["parameter"].requires_grad_(False)
    from jittor.compat.torch.nested import _torch_prune_leaf_registry
    _torch_prune_leaf_registry()


@pytest.mark.parametrize("extension", (".pt", ".pkl"))
def test_weights_only_format_probe_never_executes_payload(tmp_path, extension):
    torch = compatibility_owner(jt)
    _EXECUTED.clear()
    path = tmp_path / ("unsafe" + extension)
    path.write_bytes(pickle.dumps(_UnsafeObject()))
    with pytest.raises(pickle.UnpicklingError, match="not an allowed global"):
        torch.load(path)
    assert _EXECUTED == []
    assert torch.load(path, weights_only=False) == 42
    assert _EXECUTED == [True]
    _EXECUTED.clear()


def test_native_only_paths_require_explicit_unsafe_opt_in(tmp_path, monkeypatch):
    torch = compatibility_owner(jt)
    context = get_install_context(jt)
    calls = []
    def native(path):
        calls.append(path)
        return {"native": True}
    captured = dict(context.state["core_native_api"], load=native)
    monkeypatch.setitem(context.state, "core_native_api", MappingProxyType(captured))
    with pytest.raises(pickle.UnpicklingError, match="native URL loader"):
        torch.load("jittorhub://test-weights")
    assert calls == []
    assert torch.load("jittorhub://test-weights", weights_only=False) == {"native": True}
    legacy = tmp_path / "legacy.pt"
    legacy.write_bytes(pickle.dumps(0x1950a86a20f9469cfc6c))
    module = importlib.import_module("jittor.serialization.load_pytorch")
    monkeypatch.setattr(module, "load_pytorch", native)
    with pytest.raises(pickle.UnpicklingError, match="legacy Torch format"):
        torch.load(legacy)
    assert len(calls) == 1
    assert torch.load(legacy, weights_only=False) == {"native": True}
    assert calls[-1] == str(legacy)


def test_torch_zip_storage_strides_and_invalid_bounds():
    torch = compatibility_owner(jt)
    restored = torch.load(io.BytesIO(_torch_archive()), map_location="cpu")
    assert restored.dtype is torch.float64
    np.testing.assert_array_equal(restored.numpy(), np.arange(6).reshape(2, 3).T)
    for data, message in ((_torch_archive(stride=(-1, 3)), "negative stride"),
                          (_torch_archive(offset=5), "reaches")):
        with pytest.raises(pickle.UnpicklingError, match=message):
            torch.load(io.BytesIO(data))


def test_safetensors_dtype_bytes_reader_and_numpy_delegate(tmp_path):
    st = pytest.importorskip("safetensors.torch")
    numpy_io = importlib.import_module("safetensors.numpy")
    torch = compatibility_owner(jt)
    values = {"wide": torch.tensor([2**45, 2**45 + 1], dtype=torch.int64),
              "bf16": torch.tensor(1.25, dtype=torch.bfloat16)}
    assert st.save is safe_owner._save_dict and st.load is safe_owner._load_bytes
    loaded = st.load(st.save(values))
    assert loaded["bf16"].dtype is torch.bfloat16
    np.testing.assert_array_equal(loaded["wide"].numpy(), values["wide"].numpy())
    path = tmp_path / "wide.safetensors"
    st.save_file({"wide": values["wide"]}, path)
    native = numpy_io.load_file(path)["wide"]
    assert isinstance(native, np.ndarray) and native.dtype == np.int64
    with st.safe_open(path, framework="pt", device="cpu") as reader:
        assert type(reader) is safe_owner._PySafeOpen
        assert reader.get_slice("wide")[1:].item() == 2**45 + 1
    before = path.read_bytes()
    with pytest.raises(NotImplementedError, match="dtype"):
        st.save_file({"bad": np.array([1j], dtype=np.complex64)}, path)
    assert path.read_bytes() == before
    with pytest.raises(NotImplementedError, match="F8"):
        safe_owner._bytes_to_np(b"\x00", "F8_E4M3", [1])


def test_optional_patch_failure_restores_all_owners(monkeypatch):
    context = get_install_context(jt)
    original = object()
    root = types.ModuleType("safetensors")
    root.__path__ = []
    root.safe_open = original
    leaf = types.ModuleType("safetensors.torch")
    leaf.safe_open = leaf.load = leaf.load_file = leaf.save = leaf.save_file = original
    root.torch = leaf
    registry = ModuleRegistry(context.target_namespace, {"safetensors": root},
                              native_backend=context.native_backend)
    import sys
    monkeypatch.setitem(sys.modules, "safetensors", root)
    monkeypatch.setitem(sys.modules, "safetensors.torch", leaf)
    sentinel = object()
    before = context.state.get("safetensors_native_api", sentinel)
    setter = safe_owner.set_attr
    def fail(target, name, value, **kwargs):
        if target is leaf and name == "load_file":
            raise RuntimeError("injected safetensors binding failure")
        return setter(target, name, value, **kwargs)
    monkeypatch.setattr(safe_owner, "set_attr", fail)
    with pytest.raises(RuntimeError, match="injected safetensors"):
        safe_owner._install_safetensors_shim(registry)
    assert root.safe_open is original
    assert not hasattr(root, "_jittor_torch_compat")
    assert leaf.safe_open is leaf.load is leaf.load_file is original
    assert context.state.get("safetensors_native_api", sentinel) is before
