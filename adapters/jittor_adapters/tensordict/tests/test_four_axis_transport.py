"""Pinned TD/verl restore host contracts, not tensor/device/PPO evidence."""
import ast
from contextlib import ExitStack
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

ADAPTERS = Path(__file__).resolve().parents[3]
ROOT = ADAPTERS.parent
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))
from _adapter_boundary import boundary_errors
from jittor_adapters.tensordict import consolidated_transport as adapter
from jittor_adapters._common import UnsupportedAdapterVersion

compat = types.ModuleType("_td_fouraxis_host_compat")
compat.__path__ = [os.environ.get("JITTOR_COMPAT_SOURCE", str(ROOT / "compat"))]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")
transaction = importlib.import_module(compat.__name__ + ".transaction")
FIXTURE = Path(__file__).with_name("_td_v010_transport_fixture.py")
GUARD = ROOT / "agent/skills/verl-torch-compat/runtime/oracle/sitecustomize.py"
requires_oracle = unittest.skipUnless(
    GUARD.is_file(),
    "agent/skills verl-torch-compat oracle fixture is not ported")


class Tensor:
    is_nested = False
    def __init__(self, shape, data=None, dtype="int64", device="cpu", grad=False, owner=None):
        self.shape, self.data, self.dtype, self.device = tuple(shape), data, dtype, device
        self.requires_grad = grad
        self.owner = self if owner is None else owner
    def dim(self): return len(self.shape)
    def detach(self): return Tensor(self.shape, self.data, self.dtype, self.device, owner=self.owner)
    def cpu(self): return Tensor(self.shape, self.data, self.dtype, "cpu", owner=self.owner)
    def tolist(self):
        if self.data is None: raise AssertionError("values must not be read by restore policy")
        return list(self.data)
    def numpy(self): raise AssertionError("host contracts do not materialize tensor values")
    def data_ptr(self): return id(self.owner)
    def view(self, what):
        if isinstance(what, str): return Tensor(self.shape, self.data, what, self.device, self.requires_grad, self.owner)
        return Tensor(what, self.data, self.dtype, self.device, self.requires_grad, self.owner)


class Nested:
    is_nested = True
    def __init__(self, values, offsets, lengths, axis):
        total = offsets.data[-1]
        if total > values.shape[axis-1]: raise ValueError("offset exceeds buffer")
        self._values, self._offsets, self._lengths = values, offsets, lengths
        shape = list(values.shape)
        shape[axis-1] = -1
        self.shape = (len(offsets.data)-1, *shape)
        self.requires_grad, self.dtype, self.device = values.requires_grad, values.dtype, values.device
    def values(self): return self._values
    def offsets(self): return self._offsets
    def lengths(self): return self._lengths
    def dim(self): return len(self.shape)


class Storage:
    def __init__(self, entries): self.entries = entries
    def __getitem__(self, key): return self.entries[key.start]


class Container:
    calls = 0
    @classmethod
    def _from_dict_validated(cls, data, **metadata):
        cls.calls += 1
        result = cls()
        result.data, result.metadata, result.locked = data, metadata, False
        return result
    def lock_(self): self.locked = True; return self


class FailAlias(types.ModuleType):
    reject = False
    def __setattr__(self, name, value):
        if self.reject and name == adapter.FUNCTION: raise RuntimeError("alias setter failure")
        super().__setattr__(name, value)


def oracle_scope():
    tree = ast.parse(GUARD.read_text())
    names = {"_jagged_positions_enabled", "_restore_td_transport_binding", "_restore_td_transport", "_register_td_transport"}
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = {"os": os, "sys": sys, "_TD_TRANSPORT_TARGET": adapter.MODULE,
             "_oracle_td_transport_bindings": []}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(GUARD), "exec"), scope)
    return scope


class TestFourAxisTransport(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(mock.patch.dict(os.environ, {adapter.FLAG: "1"}, clear=True))
        self.root = types.ModuleType("tensordict")
        self.root.__version__ = "0.10.0"
        self.root.__path__ = []
        self.verl = types.ModuleType("verl")
        self.verl.__version__ = "0.9.1"
        self.module = types.ModuleType(adapter.MODULE)
        self.module.__file__ = str(FIXTURE)
        exec(compile(FIXTURE.read_text(), str(FIXTURE), "exec"), vars(self.module))
        self.original = getattr(self.module, adapter.FUNCTION)
        self.alias = types.ModuleType("tensordict.base")
        setattr(self.alias, adapter.FUNCTION, self.original)
        self.calls = []
        def constructor(values, *, offsets, lengths, jagged_dim):
            self.calls.append((values, offsets, lengths, jagged_dim))
            return Nested(values, offsets, lengths, jagged_dim)
        self.module.torch = types.SimpleNamespace(Size=tuple, int64="int64", int32="int32", SymInt=type("SymInt",(),{}), nested=types.SimpleNamespace(nested_tensor_from_jagged=constructor))
        self.module._STR_DTYPE_TO_DTYPE = {name: name for name in ("int64", "int32", "int16", "bool", "float32")}
        self.module.CLS_MAP, self.module._is_tensorclass = {"Container": Container}, lambda cls: False
        self.stack.enter_context(mock.patch.dict(sys.modules, {"tensordict": self.root,
            adapter.MODULE: self.module, "tensordict.base": self.alias, "verl": self.verl,
            "jittor.compat.module_patcher": patcher}))
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        self.addCleanup(self.cleanup)
        Container.calls = 0

    def cleanup(self):
        patcher.release_module_patch_hooks()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)

    def payload(self, reverse=False):
        values, ids = Tensor((4,8)), Tensor((8,))
        pos_offs, id_offs = Tensor((3,), [0,3,8]), Tensor((3,), [0,3,8])
        entries = {0: values, 256: pos_offs, 280: ids, 344: id_offs}
        pos = [("<NJT_VALUES>position_ids", ("int64", [4,8], 0,256,0)),
               ("<NJT_OFFSETS>position_ids", ("int64", [3], 256,280,0))]
        ids_leaf = [("<NJT_VALUES>input_ids", ("int64", [8],280,344,0)),
                    ("<NJT_OFFSETS>input_ids", ("int64", [3],344,368,0))]
        leaves = dict((ids_leaf+pos) if reverse else (pos+ids_leaf))
        metadata = {"non_tensors": {}, "leaves": leaves, "cls": "Container",
                    "cls_metadata": {"batch_size": [2], "is_locked": True}}
        return metadata, Storage(entries)

    def replacement(self): return adapter.prepare(self.module)[0][2]

    def candidate_install(self):
        patcher.register_module_patch(adapter.MODULE, adapter.patch_deferred_transport)
        return patcher._apply_module_patches(self.module)

    def test_exact_source_pin(self):
        self.assertEqual(hashlib.sha256(inspect.getsource(self.original).encode()).hexdigest(), adapter.FUNCTION_SHA256)

    def test_candidate_registration_preserves_deferred_root_order_no_import(self):
        from jittor_adapters import tensordict as root_adapter
        before = set(sys.modules)
        calls = []
        root_adapter.register(lambda *args: calls.append(args))
        self.assertEqual(calls, [("tensordict.utils", root_adapter.patch_deferred_rebuild),
                                (adapter.MODULE, adapter.patch_deferred_transport),
                                ("tensordict", root_adapter.patch_tensordict)])
        self.assertEqual(set(sys.modules), before)

    def test_four_axis_leaf_orders_buffers_metadata_lock_and_live_globals(self):
        replacement = self.replacement()
        self.assertIs(replacement.__globals__, self.original.__globals__)
        self.assertEqual(inspect.signature(replacement), inspect.signature(self.original))
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                metadata, storage = self.payload(reverse)
                result = replacement(metadata, storage)
                pos = result.data["position_ids"]
                self.assertEqual(pos.shape, (2,4,-1))
                self.assertEqual(result.data["input_ids"].shape, (2,-1))
                self.assertIs(result._consolidated["storage"], storage)
                self.assertIs(result._consolidated["metadata"], metadata)
                self.assertTrue(result.locked)
                self.assertEqual(pos.values().data_ptr(), storage.entries[0].data_ptr())
                self.assertFalse(pos.values().requires_grad)
                self.assertEqual(pos.offsets().data_ptr(), storage.entries[256].data_ptr())
        self.module.torch.nested.nested_tensor_from_jagged = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("live global"))
        with self.assertRaisesRegex(RuntimeError, "live global"): replacement(*self.payload())

    def test_companion_int32_offsets_preserved_in_both_leaf_orders(self):
        # verl's unpad_input generates int32 cu_seqlens; its separate position
        # helper generates int64 offsets. Restore must preserve both buffers,
        # their own serialized dtype and equal offset values, without casting.
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                metadata, storage = self.payload(reverse)
                storage.entries[344].dtype = "int32"
                metadata["leaves"]["<NJT_OFFSETS>input_ids"] = ("int32", [3],344,356,0)
                result = self.replacement()(metadata, storage)
                ids, positions = result.data["input_ids"], result.data["position_ids"]
                self.assertEqual(ids.offsets().dtype, "int32")
                self.assertEqual(positions.offsets().dtype, "int64")
                self.assertEqual(ids.offsets().tolist(), positions.offsets().tolist())
                self.assertEqual(ids.offsets().data_ptr(), storage.entries[344].data_ptr())
                self.assertIs(result._consolidated["storage"], storage)
                self.assertIs(result._consolidated["metadata"], metadata)
                self.assertTrue(result.locked)

    def test_companion_offset_contract_remains_strict_and_position_int32_rejected(self):
        mutations = {
            "int16": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>input_ids", ("int16",[3],344,350,0)),
            "bool": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>input_ids", ("bool",[3],344,347,0)),
            "float32": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>input_ids", ("float32",[3],344,356,0)),
            "rank": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>input_ids", ("int64",[1,3],344,368,0)),
            "gradient": lambda m,s: setattr(s.entries[344], "requires_grad", True),
            "device": lambda m,s: setattr(s.entries[344], "device", "cuda:0"),
            "position_int32": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>position_ids", ("int32",[3],256,268,0)),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                Container.calls = 0
                metadata, storage = self.payload()
                mutation(metadata, storage)
                with self.assertRaises(ValueError): self.replacement()(metadata, storage)
                self.assertEqual(Container.calls, 0)

    def test_default_axis_one_and_non_position_fields_no_verl_dependency(self):
        for flag in (None, "0"):
            env = {} if flag is None else {adapter.FLAG: flag}
            with self.subTest(flag=flag), mock.patch.dict(os.environ, env, clear=True):
                with mock.patch.dict(sys.modules):
                    sys.modules.pop("verl", None)
                    replacement = self.replacement()
                    metadata, storage = self.payload()
                    metadata["leaves"] = {"<NJT_VALUES>n": ("int64", [8,4],0,256,0),
                                          "<NJT_OFFSETS>n": ("int64", [3],256,280,0)}
                    storage.entries[0].shape = (8,4)
                    self.assertEqual(replacement(metadata, storage).data["n"].shape, (2,-1,4))
                    self.assertEqual(self.calls[-1][-1], 1)
        with mock.patch.dict(sys.modules):
            sys.modules.pop("verl", None)
            metadata, storage = self.payload()
            del metadata["leaves"]["<NJT_VALUES>position_ids"]
            del metadata["leaves"]["<NJT_OFFSETS>position_ids"]
            self.replacement()(metadata, storage)

    def test_dense_position_not_intercepted(self):
        metadata, storage = self.payload()
        metadata["leaves"] = {"position_ids": ("int64", [4,8],0,256,0)}
        result = self.replacement()(metadata, storage)
        self.assertIsInstance(result.data["position_ids"], Tensor)
        self.assertEqual(self.calls, [])

    def test_named_field_preconditions_and_companion_fail_before_container(self):
        mutations = {
            "float_position": lambda m,s: m["leaves"].__setitem__("<NJT_VALUES>position_ids", ("float32",[4,8],0,256,0)),
            "static_axes": lambda m,s: m["leaves"].__setitem__("<NJT_VALUES>position_ids", ("int64",[3,8],0,256,0)),
            "position_grad": lambda m,s: setattr(s.entries[0], "requires_grad", True),
            "offset_grad": lambda m,s: setattr(s.entries[256], "requires_grad", True),
            "offset_rank": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>position_ids", ("int64",[1,3],256,280,0)),
            "offset_dtype": lambda m,s: m["leaves"].__setitem__("<NJT_OFFSETS>position_ids", ("float32",[3],256,280,0)),
            "offset_start": lambda m,s: setattr(s.entries[256], "data", [1,3,8]),
            "offset_order": lambda m,s: setattr(s.entries[256], "data", [0,9,8]),
            "offset_end": lambda m,s: setattr(s.entries[256], "data", [0,3,7]),
            "offset_device": lambda m,s: setattr(s.entries[256], "device", "cuda:0"),
            "missing_ids": lambda m,s: [m["leaves"].pop(k) for k in ("<NJT_VALUES>input_ids", "<NJT_OFFSETS>input_ids")],
            "ids_offsets": lambda m,s: setattr(s.entries[344], "data", [0,4,8]),
            "ids_dtype": lambda m,s: m["leaves"].__setitem__("<NJT_VALUES>input_ids", ("float32",[8],280,344,0)),
            "ids_device": lambda m,s: [setattr(s.entries[i], "device", "cuda:0") for i in (280,344)],
            "batch": lambda m,s: m["cls_metadata"].__setitem__("batch_size", [3]),
            "extra_batch_axis": lambda m,s: m["cls_metadata"].__setitem__("batch_size", [2,4]),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name):
                Container.calls = 0
                m,s = self.payload()
                mutation(m,s)
                with self.assertRaises(ValueError): self.replacement()(m,s)
                self.assertEqual(Container.calls, 0)

    def test_public_companion_gradient_is_rejected_but_upstream_detach_preserved(self):
        original_constructor = self.module.torch.nested.nested_tensor_from_jagged
        def bad_companion(values, *, offsets, lengths, jagged_dim):
            result = original_constructor(values, offsets=offsets, lengths=lengths, jagged_dim=jagged_dim)
            if values.dim() == 1:
                result.requires_grad = True
            return result
        with mock.patch.object(self.module.torch.nested, "nested_tensor_from_jagged", bad_companion):
            with self.assertRaisesRegex(ValueError, "gradient contract"): self.replacement()(*self.payload())
        # Reconstructed fields retain the upstream detach contract. A storage
        # fake's pre-detach flag is not the public restored field's gradient.
        m,s = self.payload()
        s.entries[280].requires_grad = True
        result = self.replacement()(m,s)
        self.assertFalse(result.data["input_ids"].requires_grad)

    def test_holes_rejected(self):
        m,s = self.payload()
        items = list(m["leaves"].items())
        items.insert(1, ("<NJT_LENGTHS>position_ids", ("int64",[2],400,416,0)))
        m["leaves"] = dict(items)
        s.entries[400] = Tensor((2,), [2,4])
        with self.assertRaisesRegex(ValueError, "without holes"): self.replacement()(m,s)
        self.assertEqual(Container.calls, 0)

    def test_nested_named_field_rejected(self):
        m,s = self.payload()
        parent = {"non_tensors":{}, "leaves":{}, "cls":"Container", "cls_metadata":{"batch_size":[2]}, "child":m}
        with self.assertRaisesRegex(ValueError, "only root"): self.replacement()(parent,s)

    def test_prepare_early_but_actual_position_requires_loaded_verl_version(self):
        for value in (None, "missing", "0.9.2"):
            with self.subTest(value=value), mock.patch.dict(sys.modules):
                if value is None: sys.modules.pop("verl", None)
                elif value == "missing": sys.modules["verl"] = types.ModuleType("verl")
                else:
                    other = types.ModuleType("verl"); other.__version__ = value; sys.modules["verl"] = other
                replacement = self.replacement()  # TD can complete before verl.
                with self.assertRaises(UnsupportedAdapterVersion): replacement(*self.payload())
                self.assertEqual(Container.calls, 0)

    def test_bad_flag_version_source_owner_and_reserved_names_rejected(self):
        for flag in ("", "true", "2"):
            with self.subTest(flag=flag), mock.patch.dict(os.environ, {adapter.FLAG:flag}):
                with self.assertRaises(UnsupportedAdapterVersion): adapter.prepare(self.module)
        with mock.patch.object(self.root, "__version__", "0.11.0"):
            with self.assertRaises(UnsupportedAdapterVersion): adapter.prepare(self.module)
        with mock.patch.object(adapter, "FUNCTION_SHA256", "0"*64):
            with self.assertRaises(UnsupportedAdapterVersion): adapter.prepare(self.module)
        with mock.patch.dict(sys.modules, {adapter.MODULE:types.ModuleType(adapter.MODULE)}):
            with self.assertRaises(UnsupportedAdapterVersion): adapter.prepare(self.module)
        setattr(self.module, adapter.FUNCTION, lambda *a: None)
        with self.assertRaises(UnsupportedAdapterVersion): adapter.prepare(self.module)
        source = inspect.getsource(self.original)
        for changed in (source.replace("result = cls._from_dict_validated", "other = cls._from_dict_validated"),
                        source.replace("lengths=nested_lengths", "lengths=None"),
                        source.replace("consolidated =", "_td_reserved =")):
            with self.subTest(source=changed), self.assertRaises(UnsupportedAdapterVersion):
                adapter._rewrite(self.original, changed, True)

    def test_candidate_owned_idempotence_restore_and_copied_marker(self):
        self.candidate_install()
        current = getattr(self.module, adapter.FUNCTION)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), current)
        self.assertEqual(adapter.prepare(self.module), [])
        self.assertEqual(self.candidate_install()[0].status, "unchanged")
        def foreign(*args, **kw): return None
        foreign.__dict__.update(current.__dict__)
        setattr(self.module, adapter.FUNCTION, foreign)
        with self.assertRaises(UnsupportedAdapterVersion): adapter.prepare(self.module)
        setattr(self.module, adapter.FUNCTION, current)
        patcher.release_module_patch_hooks()
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    def test_candidate_alias_failure_rolls_back(self):
        failing = FailAlias("tensordict.failure_alias")
        setattr(failing, adapter.FUNCTION, self.original)
        failing.reject = True
        with mock.patch.dict(sys.modules, {failing.__name__:failing}):
            with self.assertRaisesRegex(RuntimeError, "alias setter failure"): self.candidate_install()
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    def test_profile_change_and_constructor_noop_fail_closed(self):
        replacement = self.replacement()
        with mock.patch.dict(os.environ, {adapter.FLAG:"0"}):
            with self.assertRaises(UnsupportedAdapterVersion): replacement(*self.payload())
        with mock.patch.object(self.module.torch.nested, "nested_tensor_from_jagged", return_value=Nested(Tensor((4,8)),Tensor((3,),[0,3,4]),None,1)):
            with self.assertRaises(ValueError): replacement(*self.payload())

    def test_public_relocatable_boundary_and_shared_prepare_no_jittor_import(self):
        self.assertEqual(boundary_errors(Path(adapter.__file__).read_text()), [])
        before = set(sys.modules)
        adapter.prepare(self.module)
        self.assertFalse([n for n in set(sys.modules)-before if n == "jittor" or n.startswith("jittor.")])

    @requires_oracle
    def test_oracle_same_prepare_aliases_lifecycle_and_no_jittor(self):
        scope = oracle_scope()
        before = set(sys.modules)
        scope["_register_td_transport"](self.root)
        current = getattr(self.module, adapter.FUNCTION)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), current)
        self.assertEqual(adapter.prepare(self.module), [])
        self.assertEqual(current(*self.payload()).data["position_ids"].shape, (2,4,-1))
        scope["_register_td_transport"](self.module)
        self.assertEqual(len(scope["_oracle_td_transport_bindings"]), 2)
        self.assertFalse([n for n in set(sys.modules)-before if n == "jittor" or n.startswith("jittor.")])
        scope["_restore_td_transport"]()
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)

    @requires_oracle
    def test_oracle_defers_early_root_and_default_off_no_prepare(self):
        scope = oracle_scope()
        del self.root.__version__
        scope["_register_td_transport"](self.module)
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.root.__version__ = "0.10.0"
        scope["_register_td_transport"](self.root)
        scope["_restore_td_transport"]()
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(adapter, "prepare", side_effect=AssertionError("default-off prepare")):
            scope["_register_td_transport"](self.module)

    @requires_oracle
    def test_oracle_adapter_import_failure_is_fail_closed_but_missing_target_does_not_import(self):
        scope = oracle_scope()
        with mock.patch("builtins.__import__", side_effect=ImportError("adapter unavailable")):
            with self.assertRaisesRegex(SystemExit, "adapter unavailable"):
                scope["_register_td_transport"](self.root)
        with mock.patch.dict(sys.modules):
            sys.modules.pop(adapter.MODULE)
            with mock.patch("builtins.__import__", side_effect=AssertionError("forced optional import")):
                scope["_register_td_transport"](self.root)

    @requires_oracle
    def test_oracle_source_failure_and_partial_alias_failure_rollback(self):
        scope = oracle_scope()
        with mock.patch.object(adapter, "FUNCTION_SHA256", "0"*64):
            with self.assertRaises(SystemExit): scope["_register_td_transport"](self.root)
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        failing = FailAlias("tensordict.failure_alias")
        setattr(failing, adapter.FUNCTION, self.original)
        failing.reject = True
        with mock.patch.dict(sys.modules, {failing.__name__:failing}):
            with self.assertRaisesRegex(SystemExit, "alias setter failure"): scope["_register_td_transport"](self.root)
        self.assertIs(getattr(self.module, adapter.FUNCTION), self.original)
        self.assertIs(getattr(self.alias, adapter.FUNCTION), self.original)
        self.assertEqual(scope["_oracle_td_transport_bindings"], [])

    @requires_oracle
    def test_oracle_external_binding_preserved_and_finder_two_stage(self):
        scope = oracle_scope()
        scope["_register_td_transport"](self.root)
        current = getattr(self.alias, adapter.FUNCTION)
        foreign = lambda *a: None
        setattr(self.alias, adapter.FUNCTION, foreign)
        with self.assertRaises(SystemExit): scope["_restore_td_transport"]()
        self.assertIs(getattr(self.alias, adapter.FUNCTION), foreign)
        setattr(self.alias, adapter.FUNCTION, current)
        scope["_restore_td_transport"]()
        tree = ast.parse(GUARD.read_text())
        finder = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_OracleFinder")
        text = ast.unparse(finder)
        self.assertIn("'tensordict': _register_td_transport", text)
        self.assertIn("_TD_TRANSPORT_TARGET: _register_td_transport", text)


if __name__ == "__main__": unittest.main()
