"""BF16 codec round trips preserve storage words, including NaN payloads."""
import io
import json
from pathlib import Path
import pickle
import pickletools
import struct
import tempfile
import zipfile

import numpy as np
import pytest
import torch
import jittor as jt
from jittor._runtime.fallback import forbid_backend_fallbacks
from jittor.compat.torch.nested import _NestedTensor
from jittor.compat.torch.serialization import safetensors as safe_owner
from _helpers.device_types import instantiate_device_type_tests, onlyCPU

WORDS = np.asarray([0, 0x8000, 0x3F80, 0xBF80, 0x7FC1, 0x7F81,
                    0xFFC5, 0x7FFF, 0xFF81, 0x7F80, 0x0001, 0x8001], dtype=np.uint16)


def value(words, device):
    words = np.asarray(words, dtype=np.uint16)
    carrier = torch.tensor(words, dtype=torch.uint16, device=device)
    return jt.reinterpret_view(carrier, tuple(words.shape), 'bfloat16').stop_grad()


def bits(tensor):
    dense = tensor.detach().contiguous()
    raw = jt.reinterpret_view(dense, tuple(dense.shape), 'uint16')
    return np.array(raw.cpu().numpy(), copy=True)


def assert_value(tensor, words, device):
    assert tensor.dtype is torch.bfloat16
    assert tuple(tensor.shape) == tuple(np.asarray(words).shape)
    backend, _ = jt.core.dispatch_context([tensor])
    assert backend == {'cpu': 'cpu', 'cuda': 'cuda', 'npu': 'acl'}[device]
    np.testing.assert_array_equal(bits(tensor), words)


def fragment(item):
    return pickletools.optimize(pickle.dumps(item, protocol=2))[2:-1]


def archive_bytes(words, shape, stride):
    storage = (b'(' + fragment('storage') + b'ctorch\nBFloat16Storage\n' +
               fragment('0') + fragment('cpu') + fragment(words.size) + b'tQ')
    data = (b'\x80\x02ctorch._utils\n_rebuild_tensor_v2\n(' + storage +
            fragment(0) + fragment(shape) + fragment(stride) + b'\x89NtR.')
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr('archive/version', '3\n')
        archive.writestr('archive/byteorder', 'little')
        archive.writestr('archive/data.pkl', data)
        archive.writestr('archive/data/0', words.tobytes())
    stream.seek(0)
    return stream


class TestBF16SerializationRawBits:
    def test_portable_parameter_scalar_empty_and_view(self, device):
        with forbid_backend_fallbacks():
            dense = value(WORDS.reshape(3, 4), device)
            values = {'dense': dense, 'view': dense.transpose(0, 1),
                      'parameter': torch.nn.Parameter(value(WORDS, device), requires_grad=True),
                      'scalar': value(WORDS[4].reshape(()), device), 'empty': value(WORDS[:0], device)}
            pointer = dense.data_ptr()
            stream = io.BytesIO();torch.save(values, stream)
            wire = pickle.loads(stream.getvalue())
            assert all(entry['data'].dtype == np.uint16 for entry in wire.values())
            stream.seek(0)
            restored = torch.load(stream, weights_only=True, map_location=device)
            assert_value(restored['dense'], WORDS.reshape(3, 4), device)
            assert_value(restored['view'], WORDS.reshape(3, 4).T, device)
            assert_value(restored['parameter'], WORDS, device)
            assert isinstance(restored['parameter'], torch.nn.Parameter)
            assert restored['parameter'].requires_grad
            assert_value(restored['scalar'], WORDS[4].reshape(()), device)
            assert_value(restored['empty'], WORDS[:0], device)
            assert dense.data_ptr() == pointer
            np.testing.assert_array_equal(bits(dense), WORDS.reshape(3, 4))

    def test_legacy_float32_portable_payload_keeps_all_bits(self, device):
        with forbid_backend_fallbacks():
            for words in (WORDS, WORDS[4].reshape(()), WORDS[:0], WORDS.reshape(3, 4).T):
                widened = np.left_shift(words.astype(np.uint32), np.uint32(16)).view(np.float32)
                payload = {'__jt_var__': True, 'data': widened, 'dtype': 'bfloat16',
                           'requires_grad': False, 'parameter': False, 'device': 'cpu'}
                restored = torch.load(io.BytesIO(pickle.dumps(payload)), weights_only=True, map_location=device)
                assert_value(restored, words, device)

    def test_torch_zip_storage_keeps_scalar_empty_and_strided_bits(self, device):
        with forbid_backend_fallbacks():
            for words, shape, stride, expected in (
                (WORDS, (12,), (1,), WORDS), (WORDS, (4, 3), (1, 4), WORDS.reshape(3, 4).T),
                (WORDS[4:5], (), (), WORDS[4].reshape(())), (WORDS[:0], (0,), (1,), WORDS[:0])):
                restored = torch.load(archive_bytes(words, shape, stride), weights_only=True, map_location=device)
                assert_value(restored, expected, device)

    def test_safetensors_file_and_slice_keep_raw_bits(self, device):
        with forbid_backend_fallbacks(), tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.safetensors'
            source = value(WORDS.reshape(3, 4), device)
            safe_owner._save_file({'value': source}, path)
            encoded = path.read_bytes()
            header_size = struct.unpack('<Q', encoded[:8])[0]
            np.testing.assert_array_equal(np.frombuffer(encoded[8 + header_size:], dtype=np.uint16), WORDS)
            restored = safe_owner._load_file(path, device=device)['value']
            assert_value(restored, WORDS.reshape(3, 4), device)
            with safe_owner._PySafeOpen(path, device=device) as handle:
                assert_value(handle.get_slice('value')[:, 1:], WORDS.reshape(3, 4)[:, 1:], device)

    @onlyCPU
    def test_nested_pickle_keeps_raw_and_legacy_payloads(self, device):
        with forbid_backend_fallbacks():
            source = _NestedTensor.from_tensors([value(WORDS[:4], device), value(WORDS[4:], device)])
            restored = pickle.loads(pickle.dumps(source))
            for tensor, wanted in zip(restored.unbind(), (WORDS[:4], WORDS[4:])):
                assert_value(tensor, wanted, device)
            stream = io.BytesIO();torch.save(source, stream);stream.seek(0)
            restored = torch.load(stream, weights_only=True)
            for tensor, wanted in zip(restored.unbind(), (WORDS[:4], WORDS[4:])):
                assert_value(tensor, wanted, device)

    def test_raw_codec_does_not_expand_restricted_pickle_globals(self, device):
        with pytest.raises(pickle.UnpicklingError):
            torch.load(io.BytesIO(pickle.dumps(eval)), weights_only=True, map_location=device)
        invalid = {'__jt_var__': True, 'data': np.asarray([1], dtype=np.int16), 'dtype': 'bfloat16'}
        with pytest.raises(TypeError, match='BF16 checkpoint storage'):
            torch.load(io.BytesIO(pickle.dumps(invalid)), weights_only=True, map_location=device)


instantiate_device_type_tests(TestBF16SerializationRawBits, globals(), only_for=('cpu', 'cuda', 'npu'))
