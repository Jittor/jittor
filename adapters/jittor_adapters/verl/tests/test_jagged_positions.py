"""Pinned verl public jagged-position policy: host fakes, no framework/JIT.

The exact upstream excerpts are fixtures, not a replacement implementation.
Whole-module fixture hashes are substituted only inside host setup; production
module/function pins are independently asserted and tested fail-closed.
"""
import ast
import builtins
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


ADAPTERS = Path(__file__).resolve().parents[3]
SOURCE = ADAPTERS / "jittor_adapters" / "verl" / "jagged_positions.py"
FIXTURE = Path(__file__).with_name("_verl_v091_jagged_fixture.py")
BOOTSTRAP = ADAPTERS.parent / "agent/skills/verl-torch-compat/runtime/oracle/sitecustomize.py"


class SymInt(str):
    """Public symbolic-size fake, not a real framework import."""


class Tensor:
    is_nested = False

    def __init__(self, value, dtype=None):
        self.array = np.asarray(value, dtype=dtype)
        self.device = "cpu"
        self.dtype = self.array.dtype
        self.requires_grad = False
        self.contiguous = True

    @property
    def shape(self):
        return self.array.shape

    def dim(self):
        return self.array.ndim

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.array.__array_interface__["data"][0]

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.array.tolist()

    def bool(self):
        return Tensor(self.array.astype(bool))

    def unsqueeze(self, axis):
        return Tensor(np.expand_dims(self.array, axis))

    def squeeze(self, axis):
        return Tensor(np.squeeze(self.array, axis))

    def __getitem__(self, key):
        if isinstance(key, Tensor):
            key = key.array
        if isinstance(key, tuple):
            key = tuple(item.array if isinstance(item, Tensor) else item for item in key)
        return Tensor(self.array[key])


class Nested:
    is_nested = True

    def __init__(self, values, offsets, jagged_dim=1, lengths=None):
        self.payload, self.metadata, self.holes = values, offsets, lengths
        self.private_writes = 0
        self._ragged_idx = jagged_dim
        shape = [len(offsets.tolist()) - 1] + list(values.shape)
        shape[jagged_dim] = SymInt("j")
        self.shape = tuple(shape)

    def __setattr__(self, name, value):
        if name == "_ragged_idx" and hasattr(self, "_ragged_idx"):
            self.private_writes += 1
        object.__setattr__(self, name, value)

    def dim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return self.payload.dtype

    @property
    def device(self):
        return self.payload.device

    @property
    def requires_grad(self):
        return self.payload.requires_grad

    def values(self):
        return self.payload

    def offsets(self):
        return self.metadata

    def lengths(self):
        return self.holes

    def unbind(self):
        offsets = self.metadata.tolist()
        axis = self._ragged_idx - 1
        if offsets[-1] != self.payload.shape[axis]:
            raise RuntimeError("split sizes do not match the chosen dimension")
        return tuple(Tensor(row) for row in np.split(self.payload.array, offsets[1:-1], axis=axis))


class Torch:
    Tensor = Tensor
    long = np.int64
    uint8, int8, int16, int32, int64 = np.uint8, np.int8, np.int16, np.int32, np.int64
    SymInt = SymInt
    jagged = "jagged"

    def __init__(self):
        self.calls = []
        self.nested = types.SimpleNamespace(
            as_nested_tensor=self.as_nested_tensor,
            nested_tensor_from_jagged=self.from_jagged)

    def cat(self, tensors, dim):
        return Tensor(np.concatenate([tensor.array for tensor in tensors], axis=dim))

    def tensor(self, value, dtype=None, device=None):
        return Tensor(value, dtype)

    def zeros(self, size, dtype=None, device=None):
        return Tensor(np.zeros(size, dtype=dtype))

    def cumsum(self, tensor, dim, out):
        out.array[...] = np.cumsum(tensor.array, axis=dim)
        return out

    def from_jagged(self, values, offsets, jagged_dim=None, lengths=None):
        self.calls.append((values.shape, offsets.tolist(), jagged_dim))
        return Nested(values, offsets, 1 if jagged_dim is None else jagged_dim, lengths)

    def as_nested_tensor(self, rows, layout):
        sizes = [row.shape[0] for row in rows]
        return self.from_jagged(self.cat(rows, 0), Tensor([0] + np.cumsum(sizes).tolist()))


@pytest.fixture(autouse=True)
def no_framework_imports(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in ("jittor", "torch", "torch_npu", "torchvision"):
            raise ImportError("host policy forbids real framework imports: " + name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ADAPTERS))
    importlib.import_module("jittor_adapters.verl")
    name = "jittor_adapters.verl.jagged_positions"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, adapter)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(sys.modules["jittor_adapters.verl"], "jagged_positions", adapter, raising=False)
    root = types.ModuleType("verl")
    root.__version__ = "0.9.1"
    monkeypatch.setitem(sys.modules, "verl", root)
    torch = Torch()
    fixture_source = FIXTURE.read_text()
    fixture_tree = ast.parse(fixture_source)
    modules = {}
    for module_name, functions in (
        (adapter.TU_MODULE, ("nested_tensor_from_tensor_list", "maybe_fix_3d_position_ids")),
        (adapter.PADDING_MODULE, ("left_right_2_no_padding",)),
    ):
        filename = tmp_path / (module_name.rsplit(".", 1)[-1] + ".py")
        filename.write_text(fixture_source)
        module = types.ModuleType(module_name)
        module.__file__ = str(filename)
        module.torch = torch
        module.TensorDict = dict
        selected = ast.Module(body=[node for node in fixture_tree.body
                                    if isinstance(node, ast.FunctionDef) and node.name in functions], type_ignores=[])
        exec(compile(selected, str(filename), "exec"), vars(module))
        monkeypatch.setitem(sys.modules, module_name, module)
        # Fixture excerpts are deliberately not falsely presented as complete
        # upstream modules. The exact production pins have a separate assertion.
        monkeypatch.setitem(adapter.MODULE_SHA256, module_name,
                            hashlib.sha256(fixture_source.encode()).hexdigest())
        modules[module_name] = module
    tu, padding = modules[adapter.TU_MODULE], modules[adapter.PADDING_MODULE]
    padding.tu = tu
    tu.assign_non_tensor_data = lambda data, key, value: data.__setitem__(key, value)

    def unpad(input_ids, mask):
        rows = [input_ids.array[index][mask.array[index].astype(bool)] for index in range(mask.shape[0])]
        sizes = [row.shape[0] for row in rows]
        return (Tensor(np.concatenate(rows)), Tensor(np.flatnonzero(mask.array)),
                Tensor([0] + np.cumsum(sizes).tolist()), None)

    padding.unpad_input = unpad
    monkeypatch.setenv(adapter.FLAG, "1")
    return adapter, root, tu, padding, torch


def apply_prepared(adapter, module):
    plans = adapter.prepare(module)
    for name, original, replacement in plans:
        assert vars(module)[name] is original
        setattr(module, name, replacement)
    return plans


def rows(lengths=(3, 5), axes=4):
    return [Tensor(np.arange(axes * width).reshape(axes, width) + index * 1000)
            for index, width in enumerate(lengths)]


def test_original_constructor_and_blind_fix_are_negative_controls(setup):
    adapter, _, tu, _, torch = setup
    original = tu.nested_tensor_from_tensor_list(rows(), ragged_idx=2)
    assert original.shape[1] == "j" and original.private_writes == 1
    bad = torch.nested.as_nested_tensor(rows((3, 3)), layout=torch.jagged)
    tu.maybe_fix_3d_position_ids({"position_ids": bad})
    with pytest.raises(RuntimeError, match="split sizes"):
        bad.unbind()
    assert bad.private_writes == 1


@pytest.mark.parametrize("lengths,axes", (((3, 3), 4), ((3, 5), 4), ((1, 2), 3)))
def test_public_axis_two_exact_rows_shape_and_read_only_validation(setup, lengths, axes):
    adapter, _, tu, _, torch = setup
    plans = apply_prepared(adapter, tu)
    inputs = rows(lengths, axes)
    nested = tu.nested_tensor_from_tensor_list(inputs, ragged_idx=2)
    assert nested.shape == (2, axes, "j")
    assert torch.calls[-1] == ((axes, sum(lengths)), [0, lengths[0], sum(lengths)], 2)
    before = (nested.values().array.copy(), nested.offsets().array.copy(), nested.shape)
    tu.maybe_fix_3d_position_ids({"position_ids": nested})
    assert nested.private_writes == 0
    for observed, expected in zip(nested.unbind(), inputs, strict=True):
        np.testing.assert_array_equal(observed.array, expected.array)
    np.testing.assert_array_equal(nested.values().array, before[0])
    np.testing.assert_array_equal(nested.offsets().array, before[1])
    assert nested.shape == before[2]
    for _, original, replacement in plans:
        assert replacement.__globals__ is vars(tu)
        assert replacement.__code__.co_filename == original.__code__.co_filename
        assert replacement.__defaults__ is original.__defaults__
    assert adapter.prepare(tu) == []


def test_rank_one_rows_keep_original_public_as_nested_path(setup):
    adapter, _, tu, _, torch = setup
    apply_prepared(adapter, tu)
    nested = tu.nested_tensor_from_tensor_list([Tensor([1, 2]), Tensor([3])])
    assert nested.shape == (2, "j")
    assert torch.calls[-1][2] is None
    tu.maybe_fix_3d_position_ids({"position_ids": nested})
    tu.maybe_fix_3d_position_ids({})
    tu.maybe_fix_3d_position_ids({"position_ids": Tensor([[1, 2]])})


def test_padding_uses_corrected_helper_without_dropping_axes_or_tokens(setup):
    adapter, _, tu, padding, _ = setup
    apply_prepared(adapter, tu)
    apply_prepared(adapter, padding)
    ids = Tensor(np.arange(12).reshape(2, 6))
    positions = Tensor(np.arange(48).reshape(2, 4, 6))
    masks = Tensor([[0, 1, 1, 1, 0, 0], [0, 1, 1, 1, 1, 1]])
    responses = Tensor(np.ones((2, 2), dtype=np.int64))
    data = {"input_ids": ids, "attention_mask": masks,
            "position_ids": positions, "response_mask": responses}
    assert padding.left_right_2_no_padding(data) is data
    assert data["position_ids"].shape == (2, 4, "j")
    assert data["position_ids"].offsets().tolist() == [0, 3, 8]
    assert data["input_ids"].offsets().tolist() == [0, 3, 8]
    tu.maybe_fix_3d_position_ids(data)
    assert data["loss_mask"] is responses
    for index, observed in enumerate(data["position_ids"].unbind()):
        np.testing.assert_array_equal(observed.array, positions.array[index][:, masks.array[index].astype(bool)])
    assert adapter.prepare(padding) == []


@pytest.mark.parametrize("damage", ("old_axis", "batch", "fixed_axis", "offset_end",
                                    "negative", "float_offsets", "offset_device", "offset_rank",
                                    "holes", "token_offsets"))
def test_metadata_damage_is_rejected_without_mutation_or_fallback(setup, damage):
    adapter, _, tu, _, torch = setup
    apply_prepared(adapter, tu)
    nested = tu.nested_tensor_from_tensor_list(rows(), ragged_idx=2)
    data = {"position_ids": nested}
    if damage == "old_axis":
        nested = torch.nested.as_nested_tensor(rows((3, 3)), layout=torch.jagged)
        data["position_ids"] = nested
    elif damage == "batch":
        nested.shape = (3, 4, "j")
    elif damage == "fixed_axis":
        nested.shape = (2, 5, "j")
    elif damage == "offset_end":
        nested.metadata = Tensor([0, 3, 7])
    elif damage == "negative":
        nested.metadata = Tensor([0, -1, 8])
    elif damage == "float_offsets":
        nested.metadata = Tensor([0.0, 3.0, 8.0])
    elif damage == "offset_device":
        nested.metadata.device = "cuda:0"
    elif damage == "offset_rank":
        nested.metadata = Tensor([[0, 3, 8]])
    elif damage == "holes":
        nested.holes = Tensor([2, 4])
    else:
        data["input_ids"] = Nested(Tensor(np.arange(8)), Tensor([0, 4, 8]), 1)
    before = (nested.shape, nested.metadata.array.copy(), nested.private_writes)
    with pytest.raises(ValueError, match="jagged|multi-axis"):
        tu.maybe_fix_3d_position_ids(data)
    assert nested.shape == before[0] and nested.private_writes == before[2]
    np.testing.assert_array_equal(nested.metadata.array, before[1])


@pytest.mark.parametrize("flag", (None, "0"))
def test_default_off_does_not_register_or_mutate(setup, monkeypatch, flag):
    adapter, _, tu, _, _ = setup
    if flag is None:
        monkeypatch.delenv(adapter.FLAG)
    else:
        monkeypatch.setenv(adapter.FLAG, flag)
    callbacks = []
    adapter.register(lambda *entry: callbacks.append(entry))
    assert callbacks == [] and adapter.patch_module(tu) is False
    with pytest.raises(ImportError, match="explicit opt-in"):
        adapter.prepare(tu)


@pytest.mark.parametrize("flag", ("yes", "true", "2", ""))
def test_invalid_flag_fails_closed(setup, monkeypatch, flag):
    adapter, _, _, _, _ = setup
    monkeypatch.setenv(adapter.FLAG, flag)
    with pytest.raises(ImportError, match="must be absent"):
        adapter.register(lambda *entry: pytest.fail("registered an invalid opt-in"))


@pytest.mark.parametrize("version", (None, "0.9.2", "1.0.0"))
def test_version_is_from_actual_root_not_unrelated_distribution(setup, version):
    adapter, root, tu, _, _ = setup
    root.__version__ = version
    with pytest.raises(ImportError, match="supports"):
        adapter.prepare(tu)


def test_exact_module_function_source_and_counts_are_enforced(setup, monkeypatch):
    adapter, _, tu, _, _ = setup
    filename = Path(tu.__file__)
    filename.write_text(filename.read_text() + "\n# changed fork\n")
    with pytest.raises(ImportError, match="module source SHA256"):
        adapter.prepare(tu)
    monkeypatch.setitem(adapter.MODULE_SHA256, tu.__name__, hashlib.sha256(filename.read_bytes()).hexdigest())
    filename.write_text(filename.read_text().replace("nested_tensor._ragged_idx = ragged_idx", "nested_tensor._ragged_idx = 1"))
    monkeypatch.setitem(adapter.MODULE_SHA256, tu.__name__, hashlib.sha256(filename.read_bytes()).hexdigest())
    with pytest.raises(ImportError, match="helper source SHA256"):
        adapter.prepare(tu)
    original = tu.nested_tensor_from_tensor_list
    monkeypatch.setitem(adapter.FUNCTION_SHA256, original.__name__, hashlib.sha256(importlib.import_module("inspect").getsource(original).encode()).hexdigest())
    with pytest.raises(ImportError, match="rewrite count"):
        adapter.prepare(tu)


def test_source_missing_foreign_owner_and_padding_order_fail_closed(setup):
    adapter, _, tu, padding, _ = setup
    with pytest.raises(ImportError, match="must precede"):
        adapter.prepare(padding)
    helper = padding.tu
    padding.tu = None
    with pytest.raises(ImportError, match="must precede"):
        adapter.prepare(padding)
    padding.tu = helper
    original = tu.maybe_fix_3d_position_ids
    tu.maybe_fix_3d_position_ids = lambda data: None
    with pytest.raises(ImportError, match="ownership"):
        adapter.prepare(tu)
    tu.maybe_fix_3d_position_ids = original
    Path(tu.__file__).unlink()
    with pytest.raises(ImportError, match="source is required"):
        adapter.prepare(tu)


def test_registration_occurs_after_root_and_uses_required_owned_callbacks(setup):
    adapter, root, tu, _, _ = setup
    callbacks = []
    adapter.register(lambda *entry: callbacks.append(entry))
    assert callbacks == [("verl", adapter.patch_root), (adapter.PADDING_MODULE, adapter.patch_module)]
    assert all(callback._jittor_required_patch for _, callback in callbacks)
    assert root.__version__ == "0.9.1" and tu.nested_tensor_from_tensor_list.__closure__ is None


def test_real_public_patch_ownership_alias_rollback_and_external_replacement(setup, monkeypatch):
    adapter, root, tu, padding, _ = setup
    package = types.ModuleType("_jagged_host_compat")
    package.__path__ = [str(ADAPTERS.parent / "compat")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    patcher = importlib.import_module(package.__name__ + ".module_patcher")
    handles = []

    def replace(owner, name, replacement, expected):
        handles.append(patcher.patch_method(owner, name, replacement, expected=expected))

    def aliases(prefix, name, original, replacement):
        for path, module in tuple(sys.modules.items()):
            if module is not None and (path == prefix or path.startswith(prefix + ".")):
                if vars(module).get(name) is original:
                    replace(module, name, replacement, original)

    monkeypatch.setattr(adapter, "replace", replace)
    monkeypatch.setattr(adapter, "replace_bound_aliases", aliases)
    alias = types.ModuleType("verl._host_bound_alias")
    original = tu.maybe_fix_3d_position_ids
    alias.maybe_fix_3d_position_ids = original
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    assert adapter.patch_root(root) is True
    assert alias.maybe_fix_3d_position_ids is tu.maybe_fix_3d_position_ids
    assert adapter.patch_module(padding) is True
    assert adapter.patch_root(root) is False
    external = lambda data: None
    alias.maybe_fix_3d_position_ids = external
    restored = [patcher.restore_method(handle) for handle in reversed(handles)]
    assert any(not outcome for outcome in restored)
    assert alias.maybe_fix_3d_position_ids is external
    assert tu.maybe_fix_3d_position_ids is original
    assert adapter.prepare(tu)


def test_idempotence_does_not_accept_a_forged_policy_marker(setup):
    adapter, _, tu, _, _ = setup
    apply_prepared(adapter, tu)
    current = tu.maybe_fix_3d_position_ids
    current.__code__ = (lambda captured: lambda data: captured)(object()).__code__
    with pytest.raises(ImportError, match="owned jagged replacement changed"):
        adapter.prepare(tu)


def test_pins_and_relocatable_public_boundary_have_negative_controls():
    tree = ast.parse(SOURCE.read_text())
    constants = {}
    names = {node.targets[0].id: node.value.value for node in tree.body
             if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
             and isinstance(node.value, ast.Constant)}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("MODULE_SHA256", "FUNCTION_SHA256"):
                value = ast.Dict(keys=[ast.Constant(names[key.id]) if isinstance(key, ast.Name) else key
                                       for key in node.value.keys], values=node.value.values)
                constants[node.targets[0].id] = ast.literal_eval(value)
    assert constants["MODULE_SHA256"] == {
        "verl.utils.tensordict_utils": "4dc4c321a51e256232751ddb5cb2a2be129076e7f02f36a685fcad6d0d98f5f1",
        "verl.workers.utils.padding": "4e29a4af7d6f7a2182cbd45536df5c3423745e925d0cd3bb151440dbfba8fbcc"}
    text = FIXTURE.read_text()
    lines = text.splitlines(keepends=True)
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            excerpt = "".join(lines[node.lineno - 1:node.end_lineno])
            assert hashlib.sha256(excerpt.encode()).hexdigest() == constants["FUNCTION_SHA256"][node.name]
    spec = importlib.util.spec_from_file_location("_jagged_boundary", ADAPTERS / "tests" / "_adapter_boundary.py")
    boundary = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(boundary)
    assert boundary.boundary_errors(SOURCE.read_text()) == []
    for negative in ("torch.nested._internal", "torch.Tensor = object",
                     "setattr(torch, 'version', 'fake')", "from jittor.compat.torch.nested import _NestedTensor"):
        assert boundary.boundary_errors(negative)
    # Exact rewritten definitions contain no private Tensor metadata access.
    stores = [node for node in ast.walk(tree) if isinstance(node, ast.Attribute)
              and isinstance(node.ctx, (ast.Store, ast.Del)) and node.attr == "_ragged_idx"]
    assert stores == []


def oracle_functions():
    """Extract actual production bootstrap definitions, not a private copy."""
    if not BOOTSTRAP.is_file():
        pytest.skip("agent/skills verl-torch-compat oracle fixture is not ported")
    names = {"_jagged_positions_enabled", "_register_jagged_positions", "_AfterLoader", "_OracleFinder",
             "_register_td_transport", "_restore_td_transport", "_restore_td_transport_binding"}
    tree = ast.parse(BOOTSTRAP.read_text())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in names]
    assert {node.name for node in selected} == names
    constants = [node for node in tree.body if isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in ("_TARGET", "_WORKER_TARGET", "_VALUEHEAD_TARGET",
                                            "_FSDP_ENGINE_TARGET", "_JAGGED_PADDING_TARGET",
                                            "_TD_TRANSPORT_TARGET", "_oracle_td_transport_bindings")]
    namespace = {"os": importlib.import_module("os"), "sys": sys, "importlib": importlib}
    exec(compile(ast.Module(body=constants + selected, type_ignores=[]), str(BOOTSTRAP), "exec"), namespace)
    return namespace


def test_native_registration_uses_the_same_prepare_without_candidate_runtime(setup, monkeypatch):
    adapter, root, tu, padding, _ = setup
    namespace = oracle_functions()
    alias = types.ModuleType("verl._oracle_bound_alias")
    original = tu.maybe_fix_3d_position_ids
    alias.maybe_fix_3d_position_ids = original
    monkeypatch.setitem(sys.modules, alias.__name__, alias)
    monkeypatch.setattr(adapter, "replace", lambda *args: pytest.fail("oracle called candidate patcher"))
    namespace["_register_jagged_positions"](root)
    namespace["_register_jagged_positions"](padding)
    assert alias.maybe_fix_3d_position_ids is tu.maybe_fix_3d_position_ids
    assert adapter.prepare(tu) == [] and adapter.prepare(padding) == []
    namespace["_register_jagged_positions"](root)
    Path(tu.__file__).write_text("# foreign source\n")
    with pytest.raises(SystemExit, match="source SHA256"):
        namespace["_register_jagged_positions"](root)


@pytest.mark.parametrize("value", ("0", "1", "yes"))
def test_native_finder_opt_in_lifecycle_and_invalid_flags(setup, monkeypatch, value):
    adapter, _, _, _, _ = setup
    namespace = oracle_functions()
    for flag in ("JITTOR_VERL_HF_ASYNC_ROLLOUT", "JITTOR_VERL_FSDP2_RESIDENT_REFERENCE", "VERL_DEVICE"):
        monkeypatch.delenv(flag, raising=False)
    monkeypatch.setenv(adapter.FLAG, value)
    calls = []

    class Loader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            calls.append(("load", module.__name__))

    monkeypatch.setattr(importlib.machinery.PathFinder, "find_spec",
                        lambda fullname, path=None, target=None:
                        importlib.machinery.ModuleSpec(fullname, Loader()))
    namespace["_register_jagged_positions"] = lambda module: calls.append(("patch", module.__name__))
    namespace["_register_td_transport"] = lambda module: calls.append(("transport", module.__name__))
    finder = namespace["_OracleFinder"]()
    if value == "yes":
        with pytest.raises(SystemExit, match="must be absent"):
            finder.find_spec("verl")
    elif value == "0":
        assert finder.find_spec("verl") is None
        assert finder.find_spec(adapter.PADDING_MODULE) is None
        assert finder.find_spec("tensordict") is None
        assert finder.find_spec(namespace["_TD_TRANSPORT_TARGET"]) is None
    else:
        for target in ("verl", adapter.PADDING_MODULE, "tensordict", namespace["_TD_TRANSPORT_TARGET"]):
            spec = finder.find_spec(target)
            assert isinstance(spec.loader, namespace["_AfterLoader"])
            spec.loader.exec_module(types.ModuleType(target))
        assert calls == [("load", "verl"), ("patch", "verl"),
                         ("load", adapter.PADDING_MODULE), ("patch", adapter.PADDING_MODULE),
                         ("load", "tensordict"), ("transport", "tensordict"),
                         ("load", namespace["_TD_TRANSPORT_TARGET"]),
                         ("transport", namespace["_TD_TRANSPORT_TARGET"])]


def test_native_default_off_callback_never_prepares_or_mutates(setup, monkeypatch):
    adapter, root, tu, _, _ = setup
    namespace = oracle_functions()
    monkeypatch.delenv(adapter.FLAG)
    monkeypatch.setattr(adapter, "prepare", lambda *args: pytest.fail("default-off prepared a policy"))
    original = tu.maybe_fix_3d_position_ids
    namespace["_register_jagged_positions"](root)
    assert tu.maybe_fix_3d_position_ids is original


def test_candidate_register_invokes_jagged_after_existing_root_callback():
    source = SOURCE.with_name("__init__.py").read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "register")
    calls = [ast.unparse(node) for node in function.body if isinstance(node, ast.Expr)
             and isinstance(node.value, ast.Call)]
    assert calls.index("register_module_patch('verl', patch_verl)") < calls.index(
        "register_jagged_positions(register_module_patch)")


def stale_batch():
    values = Tensor(np.arange(32, dtype=np.int64).reshape(4, 8))
    offsets = Tensor([0, 3, 8], dtype=np.int64)
    stale = Nested(values, offsets, jagged_dim=1)
    ids = Nested(Tensor(np.arange(8, dtype=np.int64)), Tensor([0, 3, 8]), jagged_dim=1)
    return {"position_ids": stale, "input_ids": ids}


def test_frontend_public_ragged_integer_marker_does_not_accept_bool_or_fixed_axes(setup):
    adapter, _, _, _, _ = setup
    frontend_types = types.SimpleNamespace(SymInt=int)
    assert adapter._ragged_dimension(-1, frontend_types)
    assert not adapter._ragged_dimension(False, frontend_types)
    assert not adapter._ragged_dimension(4, frontend_types)


@pytest.mark.parametrize("ragged", (SymInt("j5"), -1))
def test_strict_public_transport_rebuild_preserves_old_object_and_shared_buffers(setup, ragged):
    adapter, _, tu, _, torch = setup
    apply_prepared(adapter, tu)
    data = stale_batch()
    old = data["position_ids"]
    old.shape = (2, ragged, 8)
    data["input_ids"].shape = (2, ragged)
    before = (old.shape, old.values().array.copy(), old.offsets().array.copy(), old.private_writes)
    tu.maybe_fix_3d_position_ids(data)
    new = data["position_ids"]
    assert new is not old and new.shape == (2, 4, "j")
    assert torch.calls[-1] == ((4, 8), [0, 3, 8], 2)
    assert new.values() is old.values() and new.offsets() is old.offsets()
    assert old.shape == before[0] and old.private_writes == before[3] == new.private_writes == 0
    np.testing.assert_array_equal(old.values().array, before[1])
    np.testing.assert_array_equal(old.offsets().array, before[2])
    for index, row in enumerate(new.unbind()):
        start, stop = (0, 3) if index == 0 else (3, 8)
        np.testing.assert_array_equal(row.array, before[1][:, start:stop])
    call_count = len(torch.calls)
    tu.maybe_fix_3d_position_ids(data)
    assert data["position_ids"] is new and len(torch.calls) == call_count
    assert adapter.POLICY == "verl_v091_public_jagged_position_construction_v2"


@pytest.mark.parametrize("damage", (
    "missing_ids", "dense_ids", "float_positions", "bool_positions", "grad_true", "grad_none",
    "noncontiguous", "wrong_axes", "unknown_shape", "wrong_packed_width", "batch",
    "token_offsets", "token_count", "token_float", "token_grad", "token_device", "token_holes",
    "old_constructor", "position_holes", "lengths_device", "lengths_rank"))
def test_stale_rebuild_rejects_incomplete_or_inconsistent_contract_without_mutation(setup, damage):
    adapter, _, tu, _, torch = setup
    apply_prepared(adapter, tu)
    data = stale_batch()
    stale = data["position_ids"]
    if damage == "missing_ids":
        del data["input_ids"]
    elif damage == "dense_ids":
        data["input_ids"] = Tensor(np.zeros((2, 4), dtype=np.int64))
    elif damage == "float_positions":
        stale.payload = Tensor(stale.values().array.astype(np.float32))
    elif damage == "bool_positions":
        stale.payload = Tensor(stale.values().array.astype(bool))
    elif damage == "grad_true":
        stale.values().requires_grad = True
    elif damage == "grad_none":
        stale.values().requires_grad = None
    elif damage == "noncontiguous":
        stale.values().contiguous = False
    elif damage == "wrong_axes":
        stale.payload = Tensor(np.zeros((3, 8), dtype=np.int64))
    elif damage == "unknown_shape":
        stale.shape = (2, 7, 8)
    elif damage == "wrong_packed_width":
        stale.shape = (2, SymInt("j"), 9)
    elif damage == "batch":
        stale.shape = (3, SymInt("j"), 8)
    elif damage == "token_offsets":
        data["input_ids"].metadata = Tensor([0, 4, 8])
    elif damage == "token_count":
        data["input_ids"].payload = Tensor(np.arange(9, dtype=np.int64))
    elif damage == "token_float":
        data["input_ids"].payload = Tensor(np.arange(8, dtype=np.float32))
    elif damage == "token_grad":
        data["input_ids"].values().requires_grad = True
    elif damage == "token_device":
        data["input_ids"].values().device = "cuda:0"
        data["input_ids"].metadata.device = "cuda:0"
    elif damage == "token_holes":
        data["input_ids"].holes = Tensor([2, 5])
    elif damage == "old_constructor":
        stale = torch.nested.as_nested_tensor(rows((38, 38)), layout=torch.jagged)
        data["position_ids"] = stale
    elif damage == "lengths_device":
        stale.holes = Tensor([3, 5])
        stale.holes.device = "cuda:0"
    elif damage == "lengths_rank":
        stale.holes = Tensor([[3, 5]])
    else:
        stale.holes = Tensor([2, 5])
    before = (stale.shape, stale.values().array.copy(), stale.offsets().array.copy(), stale.private_writes)
    call_count = len(torch.calls)
    with pytest.raises(ValueError, match="jagged|multi-axis"):
        tu.maybe_fix_3d_position_ids(data)
    assert data["position_ids"] is stale and stale.shape == before[0] and stale.private_writes == before[3]
    assert len(torch.calls) == call_count
    np.testing.assert_array_equal(stale.values().array, before[1])
    np.testing.assert_array_equal(stale.offsets().array, before[2])


def test_transport_constructor_failure_does_not_replace_downstream_object(setup, monkeypatch):
    adapter, _, tu, _, torch = setup
    apply_prepared(adapter, tu)
    data = stale_batch()
    old = data["position_ids"]

    def broken(*args, **kwargs):
        raise RuntimeError("public nested constructor failed")

    monkeypatch.setattr(torch.nested, "nested_tensor_from_jagged", broken)
    with pytest.raises(RuntimeError, match="constructor failed"):
        tu.maybe_fix_3d_position_ids(data)
    assert data["position_ids"] is old and old.private_writes == 0
