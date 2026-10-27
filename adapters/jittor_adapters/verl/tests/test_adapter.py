"""Pinned upstream control flow, fake host leaves, real patch lifecycle.

No Jittor import/JIT, tensor arithmetic, or claim of hardware coverage.
"""
import ast
import builtins
import hashlib
import importlib
import inspect
import os
from pathlib import Path
import sys
import tempfile
import traceback
import types
import unittest
from unittest import mock

ADAPTERS = Path(__file__).resolve().parents[3]
ROOT = ADAPTERS.parent
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))
from jittor_adapters import verl as adapter
from jittor_adapters._common import UnsupportedAdapterVersion
from _adapter_boundary import boundary_errors

compat = types.ModuleType("_verl_adapter_host_compat")
compat.__path__ = [os.environ.get("JITTOR_COMPAT_SOURCE", str(ROOT / "compat"))]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")
transaction = importlib.import_module(compat.__name__ + ".transaction")
FIXTURE = Path(__file__).with_name("_verl_v091_fixture.py")


class Dense:
    is_nested = False

    def __init__(self, rows):
        self.rows, self.shape = list(rows), (len(rows),)

    def size(self, dim):
        return self.shape[dim]

    def dim(self):
        return len(self.shape)

    def __iter__(self):
        return iter(self.rows)

    def __getitem__(self, indices):
        return Dense([self.rows[i] for i in indices])

    def contiguous(self):
        return self


class Jagged:
    is_nested, layout = True, "jagged"

    def __init__(self, rows):
        self.rows, self.shape = list(rows), (len(rows), -1)
        self._contiguous = True

    def size(self, dim):
        return self.shape[dim]

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return self

    def unbind(self, dim=0):
        assert dim == 0
        return tuple(self.rows)

    def numpy(self):
        raise AssertionError("Nested must never be converted to NumPy")

    def to_padded_tensor(self, padding):
        raise AssertionError("Host routing must not densify a Nested leaf")


class NonTensorData:
    shape = ()

    def __init__(self, data):
        self.data = data


class NonTensorStack(list):
    @classmethod
    def from_list(cls, values):
        return cls(values)

    @property
    def shape(self):
        return (len(self),)

    def tolist(self):
        return [value.data if isinstance(value, NonTensorData) else value for value in self]

    def __getitem__(self, key):
        if isinstance(key, Dense):
            return NonTensorStack([list.__getitem__(self, i) for i in key])
        return super().__getitem__(key)


class Array(list):
    pass


class TensorDict(dict):
    def __init__(self, source, batch_size, device=None):
        super().__init__(source)
        self.batch_size = [batch_size] if isinstance(batch_size, int) else list(batch_size)
        self.device = device

    def __len__(self):
        return self.batch_size[0] if self.batch_size else 0

    def chunk(self, chunks):
        width = len(self) // chunks
        return [TensorDict({key: value if isinstance(value, NonTensorData) else
                            value[Dense(range(i * width, (i + 1) * width))]
                            for key, value in self.items()}, [width]) for i in range(chunks)]

    @staticmethod
    def cat(items, dim):
        assert dim == 0
        result = {}
        for key in items[0]:
            values = [item[key] for item in items]
            if isinstance(values[0], Dense):
                result[key] = Dense([row for value in values for row in value.rows])
            elif isinstance(values[0], NonTensorStack):
                result[key] = NonTensorStack([row for value in values for row in value])
            else:
                result[key] = values[0]
        return TensorDict(result, [sum(len(item) for item in items)])


@unittest.skipIf(sys.version_info < (3, 10), "verl 0.9.1 upstream source requires Python >= 3.10")
class TestVerlAdapter(unittest.TestCase):
    def setUp(self):
        root = types.ModuleType("verl")
        root.__version__, root.__path__ = "0.9.1", []
        self.module = types.ModuleType("verl.utils.tensordict_utils")
        self.module.__file__ = str(FIXTURE)
        self.torch = types.SimpleNamespace(Tensor=Dense, jagged="jagged", tensor=Dense)
        vars(self.module).update(torch=self.torch, np=types.SimpleNamespace(ndarray=Array),
                                 TensorDict=TensorDict, NonTensorStack=NonTensorStack,
                                 NonTensorData=NonTensorData, Any=object)
        exec(compile(FIXTURE.read_text(), str(FIXTURE), "exec"), vars(self.module))
        self.module.nested_tensor_from_tensor_list = lambda values, ragged_idx=None: Jagged(values)
        self.module.concat_nested_tensors = lambda values: Jagged([row for value in values for row in value.rows])
        self.module.concat_tensordict_with_none_bsz = lambda values: values[0]
        self.originals = {name: getattr(self.module, name) for name in adapter.FUNCTIONS}
        self.alias = types.ModuleType("verl.workers.alias_fixture")
        self.alias.get_tensordict = self.originals["get_tensordict"]
        frontend = types.ModuleType("jittor.compat.torch")
        frontend.nested_tensor_type = lambda: Jagged
        self.modules = mock.patch.dict(sys.modules, {
            "verl": root, self.module.__name__: self.module, self.alias.__name__: self.alias,
            "jittor": types.ModuleType("jittor"), "jittor.compat": compat,
            "jittor.compat.module_patcher": patcher, "jittor.compat.torch": frontend,
        })
        self.modules.start()
        self.saved_registry = dict(patcher._REGISTRY)
        patcher._REGISTRY.clear()
        # The checked-in fixture contains the five exact upstream functions,
        # not the rest of the 0.9.1 file. Only its file digest differs; every
        # production function digest and predicate-count guard remains enabled.
        self.digest = mock.patch.object(adapter, "SOURCE_SHA256", hashlib.sha256(FIXTURE.read_bytes()).hexdigest())
        self.digest.start()
        self.original_isinstance = builtins.isinstance
        self.addCleanup(self.cleanup)

    def cleanup(self):
        patcher.uninstall_module_patches()
        patcher._REGISTRY.clear()
        patcher._REGISTRY.update(self.saved_registry)
        self.digest.stop()
        self.modules.stop()
        self.assertIs(builtins.isinstance, self.original_isinstance)
        self.assertIs(self.torch.Tensor, Dense)

    def install(self):
        adapter.register(patcher.register_module_patch)
        return patcher.install_module_patches(load_entry_points=False)

    def test_original_rejects_nested_then_patch_preserves_actual_leaf(self):
        nested = Jagged([[1], [2, 3]])
        with self.assertRaises(AssertionError):
            self.originals["get_tensordict"]({"ids": nested})
        self.install()
        td = self.module.get_tensordict({"ids": nested, "dense": Dense([4, 5]), "labels": ["a", "b"]}, {"meta": 7})
        self.assertIs(td["ids"], nested)
        self.assertNotIsInstance(nested, Dense)
        self.assertEqual(td.batch_size, [2])
        self.assertIsInstance(td["labels"], NonTensorStack)
        self.assertEqual(td["meta"].data, 7)
        self.assertIs(self.alias.get_tensordict, self.module.get_tensordict)
        for name, original in self.originals.items():
            changed = getattr(self.module, name)
            self.assertIs(changed.__globals__, vars(self.module))
            self.assertEqual(inspect.signature(changed), inspect.signature(original))

    def test_source_locations_and_validation_tracebacks_are_preserved(self):
        self.install()
        invalid_indices = Dense([0])
        invalid_indices.shape = (1, 1)
        cases = {
            "get_tensordict": ({"bad": object()},),
            "get": (TensorDict({"bad": object()}, [1]), "bad"),
            "chunk_tensordict": (TensorDict({}, [3]), 2),
            "concat_tensordict": ([],),
            "index_select_tensor_dict": (TensorDict({}, [1]), invalid_indices),
        }
        for name, arguments in cases.items():
            original, replacement = self.originals[name], getattr(self.module, name)
            with self.subTest(helper=name):
                self.assertEqual(replacement.__code__.co_filename, original.__code__.co_filename)
                self.assertEqual(replacement.__code__.co_firstlineno, original.__code__.co_firstlineno)
                locations = []
                for function in (original, replacement):
                    try:
                        function(*arguments)
                    except AssertionError as error:
                        frame = traceback.extract_tb(error.__traceback__)[-1]
                        locations.append((frame.filename, frame.lineno, frame.name))
                    else:
                        self.fail("original validation assertion did not run")
                self.assertEqual(locations[0], locations[1])

    def test_get_accepts_only_actual_nested_and_preserves_other_paths(self):
        nested = Jagged([[1], [2, 3]])
        dense = Dense([4, 5])
        metadata = {"experiment": "fixture"}
        stack = NonTensorStack.from_list([NonTensorData("a"), NonTensorData("b")])
        data = TensorDict({"nested": nested, "dense": dense, "stack": stack,
                           "meta": NonTensorData(metadata), "none": NonTensorData(None)}, [2])
        original_get = self.module.get
        with self.assertRaises(AssertionError):
            original_get(data, "nested")
        self.install()
        self.assertIs(self.module.get(data, "nested"), nested)
        self.assertNotIsInstance(nested, Dense)
        self.assertIs(self.module.get(data, "dense"), dense)
        self.assertEqual(self.module.get(data, "stack"), ["a", "b"])
        self.assertIs(self.module.get(data, "meta"), metadata)
        sentinel = object()
        self.assertIs(self.module.get(data, "absent", sentinel), sentinel)
        self.assertIsNone(self.module.get(data, "absent"))
        self.assertIsNone(self.module.get(data, "none", sentinel))
        for key in ("dense", "stack", "meta", "none", "absent"):
            self.assertEqual(self.module.get(data, key), original_get(data, key))
        # Unformatted lists and arbitrary duck types still fail exactly as
        # upstream: only NonTensorStack has list-unwrapping semantics.
        for invalid in (["a", "b"], types.SimpleNamespace(is_nested=True, shape=(2, -1))):
            malformed = TensorDict({"value": invalid}, [2])
            with self.assertRaises(AssertionError):
                original_get(malformed, "value")
            with self.assertRaises(AssertionError):
                self.module.get(malformed, "value")

    def test_all_original_validation_paths_remain(self):
        self.install()
        nested = Jagged([[1], [2]])
        bad_layout = Jagged([[1], [2]])
        bad_layout.layout = "strided"
        noncontiguous = Jagged([[1], [2]])
        noncontiguous._contiguous = False
        duck = types.SimpleNamespace(is_nested=True, shape=(2, -1), layout="jagged")
        cases = [({"x": noncontiguous}, None), ({"x": bad_layout}, None),
                 ({"x": nested, "y": Dense([1])}, None),
                 ({"x": nested, "y": NonTensorStack([1])}, None),
                 ({"x": [Dense([1])]}, None), ({"x": [nested]}, None),
                 ({"x": duck}, None), ({"x": nested}, {"x": "duplicate"})]
        for values, metadata in cases:
            with self.subTest(values=values), self.assertRaises(AssertionError):
                self.module.get_tensordict(values, metadata)
        stack = NonTensorStack([NonTensorData("a"), NonTensorData("b")])
        for values in ({"x": Dense([1, 2])}, {"x": Array(["a", "b"])}, {"x": stack}, {}):
            result = self.module.get_tensordict(values)
            expected = self.originals["get_tensordict"](values)
            self.assertEqual(result.batch_size, expected.batch_size)
            self.assertEqual(set(result), set(expected))
            self.assertEqual(type(result.get("x")), type(expected.get("x")))

    def test_chunk_index_concat_take_existing_nested_branches(self):
        self.install()
        nested = Jagged([[1], [2, 3], [4], [5, 6]])
        batch = self.module.get_tensordict({"ids": nested, "dense": Dense([1, 2, 3, 4])})
        chunks = self.module.chunk_tensordict(batch, 2)
        self.assertEqual([part["ids"].rows for part in chunks], [nested.rows[:2], nested.rows[2:]])
        selected = self.module.index_select_tensor_dict(batch, [3, 1])
        self.assertEqual(selected["ids"].rows, [nested.rows[3], nested.rows[1]])
        joined = self.module.concat_tensordict(chunks)
        self.assertEqual(joined["ids"].rows, nested.rows)
        with self.assertRaises(AssertionError):
            self.module.chunk_tensordict(batch, 3)
        with self.assertRaises(AssertionError):
            self.module.concat_tensordict([chunks[0], TensorDict({"dense": Dense([1, 2])}, [2])])
        bad_indices = Dense([0]); bad_indices.shape = (1, 1)
        with self.assertRaises(AssertionError):
            self.module.index_select_tensor_dict(batch, bad_indices)
        # Original globals must remain live, not frozen at clone time.
        replacement = mock.Mock(return_value=Jagged([["live-global"]]))
        self.module.nested_tensor_from_tensor_list = replacement
        self.module.index_select_tensor_dict(batch, [0])
        replacement.assert_called_once()

    def test_contiguous_accepts_actual_nested_leaf(self):
        nested = Jagged([[1], [2, 3]])
        data = TensorDict({"ids": nested, "dense": Dense([4, 5])}, [2])
        with self.assertRaises(AssertionError):
            self.originals["contiguous"](data)
        self.install()
        result = self.module.contiguous(data)
        self.assertIs(result["ids"], nested)
        self.assertEqual(result["dense"].rows, [4, 5])

    def test_uninstall_idempotence_and_external_patch_protection(self):
        self.install()
        first = self.module.get_tensordict
        self.install()
        self.assertIs(self.module.get_tensordict, first)
        patcher.uninstall_module_patches()
        self.assertIs(self.module.get_tensordict, self.originals["get_tensordict"])
        self.assertIs(self.alias.get_tensordict, self.originals["get_tensordict"])
        self.install()
        external = lambda value: value
        self.module.get_tensordict = external
        with self.assertRaises(transaction.TransactionConflict):
            patcher.uninstall_module_patches()
        self.assertIs(self.module.get_tensordict, external)

    def test_version_source_and_function_changes_fail_before_mutation(self):
        for target, name, value in ((sys.modules["verl"], "__version__", "0.9.2"),
                                     (adapter, "SOURCE_SHA256", "0" * 64),
                                     (self.module, "__file__", "/missing/verl-source.py"),
                                     (self.module, "index_select_tensor_dict", lambda *args: None)):
            with self.subTest(name=name), mock.patch.object(target, name, value):
                with self.assertRaises(UnsupportedAdapterVersion):
                    self.install()
            for helper, original in self.originals.items():
                self.assertIs(getattr(self.module, helper), original)

    def test_real_finder_waits_for_root_version_published_last(self):
        symbols = types.ModuleType("_verl_host_symbols")
        for name in ("torch", "np", "TensorDict", "NonTensorData", "NonTensorStack", "Any"):
            setattr(symbols, name, vars(self.module)[name])
        prefix = "from _verl_host_symbols import torch, np, TensorDict, NonTensorData, NonTensorStack, Any\n"
        helper_source = prefix + FIXTURE.read_text()
        with tempfile.TemporaryDirectory(prefix="verl-late-version-host-") as directory:
            package = Path(directory) / "verl"
            (package / "utils").mkdir(parents=True)
            (package / "utils/__init__.py").write_text("")
            (package / "utils/tensordict_utils.py").write_text(helper_source)
            (package / "__init__.py").write_text(
                "from .utils import tensordict_utils\n"
                "from .utils.tensordict_utils import get_tensordict\n"
                "__version__ = '0.9.1'\n")
            with mock.patch.dict(sys.modules, {symbols.__name__: symbols}), mock.patch.object(
                    adapter, "SOURCE_SHA256", hashlib.sha256(helper_source.encode()).hexdigest()):
                for name in tuple(sys.modules):
                    if name == "verl" or name.startswith("verl."):
                        sys.modules.pop(name)
                adapter.register(patcher.register_module_patch)
                patcher.install_module_patches(load_entry_points=False)
                sys.path.insert(0, directory)
                try:
                    imported = importlib.import_module("verl")
                    self.assertEqual(imported.__version__, "0.9.1")
                    helper = imported.tensordict_utils
                    self.assertIs(imported.get_tensordict, helper.get_tensordict)
                    nested = Jagged([[1], [2, 3]])
                    data = helper.get_tensordict({"value": nested})
                    self.assertIs(helper.get(data, "value"), nested)
                    original = helper.get._jittor_verl_original
                    patcher.uninstall_module_patches()
                    self.assertIs(helper.get, original)
                finally:
                    sys.path.remove(directory)

    def test_unrecognized_predicate_and_wrong_count_fail_closed(self):
        tree = ast.parse("isinstance(value, (torch.Tensor, list))")
        with self.assertRaises(UnsupportedAdapterVersion):
            adapter._ExpandTensorPredicate().visit(tree)
        digest, count = adapter.FUNCTIONS["get_tensordict"]
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "count"):
            adapter._clone_helper(self.originals["get_tensordict"], Jagged, digest, count + 1)

    def test_root_missing_expected_helper_fails_without_importing_it(self):
        sys.modules.pop(self.module.__name__)
        with self.assertRaisesRegex(UnsupportedAdapterVersion, "did not load"):
            self.install()
        self.assertNotIn(self.module.__name__, sys.modules)


class TestVerlRelocatableBoundary(unittest.TestCase):
    def test_adapter_uses_only_public_framework_access(self):
        owner = Path(adapter.__file__)
        for path in owner.parent.rglob("*.py"):
            relative = path.relative_to(owner.parent)
            if "tests" not in relative.parts:
                self.assertEqual(boundary_errors(path.read_text()), [], str(path))

    def test_gate_has_negative_controls(self):
        for source in ("from jittor.compat.torch.nested import _NestedTensor",
                       "torch.Tensor = object", "setattr(jittor, 'x', 1)", "torch.nested._internal"):
            self.assertTrue(boundary_errors(source), source)
        self.assertEqual(boundary_errors("from jittor.compat.torch import nested_tensor_type"), [])


if __name__ == "__main__":
    unittest.main()
