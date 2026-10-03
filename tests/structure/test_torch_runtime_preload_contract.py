"""A discoverable Torch package is not an oracle until it owns the process."""

from unittest import mock

from _helpers import torch_runtime


def test_unclaimed_torch_package_is_unavailable_to_native_parity(monkeypatch):
    monkeypatch.delenv("REAL_TORCH_SITE", raising=False)
    monkeypatch.delitem(torch_runtime.sys.modules, "torch", raising=False)
    with mock.patch.object(torch_runtime.importlib.util, "find_spec", return_value=object()):
        assert not torch_runtime.modules_available("torch.nn")
