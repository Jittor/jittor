"""Real CUDA identity is needed when a trainer connects to a separate server."""
import ctypes
from types import SimpleNamespace
import uuid

import pytest
import torch
from jittor.compat.torch.installers.cuda import api


def test_uuid_bytes_and_string_match_torch_contract():
    raw = bytes(range(16))
    value = api._CUuuid(raw)
    assert value.bytes == list(range(16))
    assert str(value) == "00010203-0405-0607-0809-0a0b0c0d0e0f"
    value.bytes[0] = 255
    assert value.bytes[0] == 0
    with pytest.raises(ValueError, match="16 bytes"):
        api._CUuuid(b"invalid")


@pytest.mark.parametrize("legacy", [False, True])
def test_uuid_queries_requested_visible_ordinal(monkeypatch, legacy):
    calls = []
    def get_device(pointer, ordinal):
        calls.append(ordinal)
        pointer._obj.value = ordinal + 10
        return 0
    def get_uuid(pointer, device):
        calls.append(device.value)
        for index in range(16):
            pointer._obj[index] = index
        return 0
    driver = SimpleNamespace(cuDeviceGet=get_device)
    setattr(driver, "cuDeviceGetUuid" if legacy else "cuDeviceGetUuid_v2", get_uuid)
    monkeypatch.setattr(api, "_cuda_driver", lambda: (driver, ctypes))
    assert api._cuda_device_uuid(3).bytes == list(range(16))
    assert calls == [3, 13]


@pytest.mark.parametrize("failure", ["driver", "ordinal", "query", "symbol"])
def test_uuid_fails_without_a_real_identity(monkeypatch, failure):
    driver = SimpleNamespace(cuDeviceGet=lambda *args: 101 if failure == "ordinal" else 0)
    if failure != "symbol":
        driver.cuDeviceGetUuid_v2 = lambda *args: 999
    monkeypatch.setattr(api, "_cuda_driver", lambda: (None, None) if failure == "driver" else (driver, ctypes))
    with pytest.raises(RuntimeError):
        api._cuda_device_uuid(0)


@pytest.mark.cuda
def test_cuda_property_is_stable_and_has_real_uuid_bytes():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    first = torch.cuda.get_device_properties(0).uuid
    again = torch.cuda.get_device_properties("cuda:0").uuid
    assert len(first.bytes) == 16 and any(first.bytes)
    assert first.bytes == again.bytes
    assert str(first) == str(uuid.UUID(bytes=bytes(first.bytes)))


@pytest.mark.cuda
def test_cuda_uuid_rejects_an_invalid_ordinal():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    with pytest.raises(RuntimeError, match="cuDeviceGet"):
        api._cuda_device_uuid(torch.cuda.device_count())
