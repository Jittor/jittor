"""Canonical codec tests with real files/NumPy bits and a host-only tensor double."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import pickle
import pickletools
import struct
from threading import Event
import sys
import tempfile
import types
import typing
import unittest
from unittest import mock
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BITS = np.asarray([0, 0x8000, 0x3F80, 0xBF80, 0x7FC1, 0x7F81,
                   0xFFC5, 0x7FFF, 0xFF81, 0x7F80, 0x0001, 0x8001], dtype=np.uint16)


class HostTensor:
    def __init__(self, array, dtype=None, requires_grad=False):
        self.array = np.array(array, copy=True)
        self.dtype = dtype or self.array.dtype.name
        self.requires_grad = requires_grad
        self.device = 'cpu'

    @property
    def shape(self):
        return self.array.shape

    @property
    def ndim(self):
        return self.array.ndim

    def numpy(self):
        return (np.left_shift(self.array.astype(np.uint32), np.uint32(16)).view(np.float32)
                if self.dtype == 'bfloat16' else self.array.copy())

    def contiguous(self):
        return self

    def stop_grad(self):
        self.requires_grad = False
        return self

    def requires_grad_(self, value):
        self.requires_grad = bool(value)
        return self

    def to(self, *, device):
        if device != 'cpu':
            raise AssertionError('host tensor double cannot claim accelerator placement')
        return self

    def int64(self):
        return HostTensor(self.array.astype(np.int64))


class HostParameter(HostTensor):
    def __init__(self, value, requires_grad=True):
        super().__init__(value.array, value.dtype, requires_grad)


def fragment(value):
    return pickletools.optimize(pickle.dumps(value, protocol=2))[2:-1]


def torch_bf16_file(path, bits, shape, stride=None):
    stride = stride if stride is not None else ((1,) if shape else ())
    storage = (b'(' + fragment('storage') + b'ctorch\nBFloat16Storage\n' +
               fragment('0') + fragment('cpu') + fragment(bits.size) + b'tQ')
    data = (b'\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(' + storage +
            fragment(0) + fragment(shape) + fragment(stride) + b'\x89NtR.')
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('archive/data.pkl', data)
        archive.writestr('archive/data/0', bits.tobytes())


class BF16CodecHostContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.scopes = []
        self.numeric_bf16_casts = 0
        self.native_views = 0
        self.modules = mock.patch.dict(sys.modules, {})
        self.modules.start()
        self.native = types.ModuleType('jittor')
        self.native.__path__ = []
        self.native.Var = HostTensor
        self.native.array = lambda array, dtype=None: HostTensor(
            np.asarray(array, dtype=dtype), dtype)
        self.native.concat = lambda values, dim: HostTensor(np.concatenate(
            [value.array for value in values], axis=dim), values[0].dtype)
        self.native.fetch = lambda *args: args[-1](*(value.numpy() for value in args[:-1]))
        self.native.sync_all = lambda *args: None
        self.native.sync = lambda roots, **kwargs: None
        core_package = types.ModuleType("jittor._core")
        core_package.__path__ = []
        core_var = types.ModuleType("jittor._core.var")
        def fetch(operands, callback):
            callback(*(value.array.copy() for value in operands))
            return object()
        core_var.core = types.SimpleNamespace(
            ops=types.SimpleNamespace(fetch=fetch),
            submit_pending_fetches=lambda: None)
        sys.modules[core_package.__name__] = core_package
        sys.modules[core_var.__name__] = core_var

        def reinterpret(value, shape, dtype):
            self.assertEqual(value.dtype, 'uint16')
            self.assertEqual(dtype, 'bfloat16')
            self.native_views += 1
            return HostTensor(value.array.reshape(shape), dtype)

        self.native.reinterpret_view = reinterpret

        def tensor(array, dtype=None, device=None, requires_grad=False):
            self.assertIn(device, (None, 'cpu'))
            if dtype == 'bfloat16':
                self.numeric_bf16_casts += 1
                words = (np.asarray(array, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
                nan = ((words & 0x7F80) == 0x7F80) & ((words & 0x007F) != 0)
                words = np.where(nan, (words & 0x8000) | 0x7FC0, words).astype(np.uint16)
                return HostTensor(words, dtype, requires_grad)
            return HostTensor(np.asarray(array, dtype=dtype), dtype, requires_grad)

        self.frontend = types.SimpleNamespace(Var=HostTensor, Tensor=HostTensor, tensor=tensor,
            nn=types.SimpleNamespace(Parameter=HostParameter), Size=tuple)
        self.context = types.SimpleNamespace(target_namespace=self.frontend, state={'core_native_api': {'load': None}})
        self.get_context = lambda _: self.context
        for name in ('jittor', 'jittor.compat', 'jittor.compat.torch', 'jittor.compat.torch.serialization'):
            package = self.native if name == 'jittor' else types.ModuleType(name)
            package.__path__ = []
            sys.modules[name] = package
        state = types.ModuleType('jittor.compat.torch.tensor_state')
        state.compatibility_owner = lambda _: self.frontend
        sys.modules[state.__name__] = state
        frontend = types.ModuleType('jittor.compat.torch.frontend')

        @contextlib.contextmanager
        def scope(kind, *, device=None):
            self.scopes.append(device)
            yield

        frontend.tensor_frontend = scope
        sys.modules[frontend.__name__] = frontend
        self.nested = self.load('nested', 'compat/torch/nested.py')
        security = self.load('serialization.security', 'compat/torch/serialization/security.py')
        self.archive = self.load('serialization.torch_archive', 'compat/torch/serialization/torch_archive.py',
            _resolve_global=security._resolve_global)
        self.portable = self.load('serialization.portable', 'compat/torch/serialization/portable.py',
            _portable_pickle_load=security._portable_pickle_load, _load_torch_pt=self.archive._load_torch_pt)
        self.safe = self.load('serialization.safetensors', 'compat/torch/serialization/safetensors.py',
            _snapshot_tensors=self.portable._snapshot_tensors)

    def tearDown(self):
        self.modules.stop()
        self.temporary.cleanup()

    def load(self, suffix, file, **extra):
        name = 'jittor.compat.torch.' + suffix
        module = types.ModuleType(name)
        module.__package__ = name.rsplit('.', 1)[0]
        module.__dict__.update(_Event=Event, np=np, _np_pt=np, jt=self.native, cast=typing.cast, Any=typing.Any,
            get_install_context=self.get_context, _jittor_dtype_name=lambda value: str(value).removeprefix('torch.'),
            _pickle=pickle, _io=io, _zipfile=zipfile, _os_pickle=os, os=os, json=json, struct=struct,
            _SafetensorError=ValueError, EXPECTED=(AttributeError, TypeError, ValueError, OSError), swallowed=lambda *a: None,
            _make_cpu_resident=lambda value, **kw: value, _make_cuda_resident=lambda *a, **kw: None,
            _move_to_cuda_index=lambda *a: None, _set_meta_placeholder=lambda value: value,
            MappingProxyType=types.MappingProxyType)
        if hasattr(self, 'nested'):
            module._rebuild_var_from_numpy = self.nested._rebuild_var_from_numpy
            module._numpy_bfloat16_storage = getattr(self.nested, '_numpy_bfloat16_storage', None)
        module.__dict__.update(extra)
        sys.modules[name] = module
        tree = ast.parse((ROOT / file).read_text())
        nodes = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / file), 'exec'), vars(module))
        return module

    def assert_bits(self, value, wanted):
        self.assertEqual(value.dtype, 'bfloat16')
        np.testing.assert_array_equal(value.array, wanted)

    def test_portable_raw_and_legacy_payloads_keep_parameter_and_storage_bits(self):
        source = HostParameter(HostTensor(BITS, 'bfloat16'), requires_grad=True)
        stream = io.BytesIO()
        self.portable.save({'parameter': source}, stream)
        wire = pickle.loads(stream.getvalue())
        self.assertEqual(wire['parameter']['data'].dtype, np.uint16)
        stream.seek(0)
        result = self.portable.load(stream, map_location='cpu', weights_only=True)['parameter']
        self.assertIsInstance(result, HostParameter)
        self.assertTrue(result.requires_grad)
        self.assert_bits(result, BITS)
        old = {'__jt_var__': True, 'data': source.numpy(), 'dtype': 'bfloat16', 'device': 'cpu'}
        self.assert_bits(self.portable.load(io.BytesIO(pickle.dumps(old)), weights_only=True), BITS)
        self.assertEqual(self.numeric_bf16_casts, 0)

    def test_existing_torch_archive_file_decoder_uses_same_raw_owner(self):
        path = self.directory / 'native.pt'
        torch_bf16_file(path, BITS, (4, 3), (1, 4))
        value = self.archive._load_torch_pt(path, weights_only=True)
        self.assert_bits(value, BITS.reshape(3, 4).T)
        self.assertEqual(self.numeric_bf16_casts, 0)
        self.assertIn('cpu', self.scopes)

    def test_safetensors_file_read_slice_write_use_same_raw_owner(self):
        path = self.directory / 'tensor.safetensors'
        raw = BITS.tobytes()
        header = json.dumps({'x': {'dtype': 'BF16', 'shape': [3, 4], 'data_offsets': [0, len(raw)]}}).encode()
        path.write_bytes(struct.pack('<Q', len(header)) + header + raw)
        with self.safe._PySafeOpen(path) as handle:
            self.assert_bits(handle.get_tensor('x'), BITS.reshape(3, 4))
            self.assert_bits(handle.get_slice('x')[:, 1:], BITS.reshape(3, 4)[:, 1:])
        encoded = self.safe._save_dict({'x': HostTensor(BITS, 'bfloat16')})
        size = struct.unpack('<Q', encoded[:8])[0]
        np.testing.assert_array_equal(np.frombuffer(encoded[8 + size:], dtype=np.uint16), BITS)
        self.assertEqual(self.numeric_bf16_casts, 0)

    def test_scalar_empty_noncontiguous_and_nested_pickle_legacy_compatibility(self):
        for words in (BITS[4].reshape(()), BITS[:0], BITS.reshape(3, 4).T):
            source = HostTensor(words, 'bfloat16')
            for payload in (source.array, source.numpy()):
                rebuilt = self.nested._rebuild_var_from_numpy(payload, 'bfloat16')
                self.assert_bits(rebuilt, words)
        nested = self.nested._NestedTensor.from_tensors([HostTensor(BITS[:4], 'bfloat16'), HostTensor(BITS[4:], 'bfloat16')])
        restored = pickle.loads(pickle.dumps(nested))
        for value, wanted in zip(restored.unbind(), (BITS[:4], BITS[4:])):
            self.assert_bits(value, wanted)
        stream = io.BytesIO();self.portable.save(nested, stream);stream.seek(0)
        restored = self.portable.load(stream, weights_only=True)
        for value, wanted in zip(restored.unbind(), (BITS[:4], BITS[4:])):
            self.assert_bits(value, wanted)

    def test_other_dtype_values_and_restricted_loading_are_unchanged(self):
        value = HostTensor(np.asarray([1.25, -3.5], dtype=np.float32))
        stream = io.BytesIO();self.portable.save(value, stream);stream.seek(0)
        np.testing.assert_array_equal(self.portable.load(stream, weights_only=True).array, value.array)
        with self.assertRaises(pickle.UnpicklingError):
            self.portable.load(io.BytesIO(pickle.dumps(eval)), weights_only=True)
        security = sys.modules['jittor.compat.torch.serialization.security']
        with self.assertRaises(pickle.UnpicklingError):
            security._resolve_global('jittor.compat.torch.nested', '_numpy_bfloat16_storage', True)

    def test_unsupported_bf16_storage_fails_explicitly(self):
        with self.assertRaises(TypeError):
            self.nested._rebuild_var_from_numpy(np.asarray([1], dtype=np.int16), 'bfloat16')

    def test_all_public_codecs_import_the_existing_nested_rebuild_owner(self):
        for file in ('portable.py', 'torch_archive.py', 'safetensors.py'):
            tree = ast.parse((ROOT / 'compat/torch/serialization' / file).read_text())
            imports = [node for node in tree.body if isinstance(node, ast.ImportFrom)
                       and node.module == 'nested' and node.level == 2]
            self.assertTrue(any('_rebuild_var_from_numpy' in {name.name for name in node.names} for node in imports), file)


if __name__ == '__main__':
    unittest.main()
