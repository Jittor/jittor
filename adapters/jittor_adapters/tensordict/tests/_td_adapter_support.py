"""Host support shared by the TensorDict adapter tests.

The fake leaves and the fake 0.10 package are not numerical tensors; they
preserve object identity and reject NumPy conversion. Kept out of the test
modules so pytest can collect every adapter suite in one session without one
test module importing another.
"""
import importlib
import os
from pathlib import Path
import sys
import types
from unittest import mock


ADAPTERS = Path(__file__).resolve().parents[3]
REPOSITORY = ADAPTERS.parent
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))

compat = types.ModuleType("_tensordict_adapter_host_compat")
compat.__path__ = [os.environ.get("JITTOR_COMPAT_SOURCE", str(REPOSITORY / "compat"))]
sys.modules[compat.__name__] = compat
patcher = importlib.import_module(compat.__name__ + ".module_patcher")
transaction = importlib.import_module(compat.__name__ + ".transaction")


class Dense:
    def __init__(self, shape=(2, 3), device=None):
        self.shape, self.device = shape, device


class Jagged:
    is_nested = True

    def __init__(self, shape=(2, -1), device=None, payload=None):
        self.shape, self.device = shape, device
        self.payload = object() if payload is None else payload
        self.moves = []

    def numpy(self):
        raise AssertionError("jagged payload must never become a NumPy/dense leaf")

    def to(self, device, non_blocking=False):
        self.moves.append((device, non_blocking))
        return Jagged(self.shape, device, self.payload)


def fake_package(version="0.10.0"):
    root = types.ModuleType("tensordict")
    root.__version__, root.__path__ = version, []
    base = types.ModuleType("tensordict.base")
    base._ACCEPTED_CLASSES = (Dense,)
    base._device_recorder = types.SimpleNamespace(marked=False, record_transfer=mock.Mock())
    base._register_tensor_class = lambda cls: None

    class TensorDictBase:
        batch_size, batch_dims, device = (2,), 1, None

        def _convert_to_tensor(self, value):
            return value.numpy() if hasattr(value, "numpy") else ("converted", value)

        def _validate_value_generic(self, value, non_blocking=False, *, check_shape=True):
            if isinstance(value, Dense):
                is_tc = False
            elif not isinstance(value, base._ACCEPTED_CLASSES):
                return self._convert_to_tensor(value)
            # Faithful 0.10 failure: registered non-Tensor leaves don't assign is_tc.
            if check_shape and not is_tc:
                return value
            return value

        def _validate_value_devicefree(self, value, non_blocking=False, *, check_shape=True):
            return self._validate_value_generic(value, non_blocking, check_shape=check_shape)

        def _validate_value_batchfree(self, value, non_blocking=False, *, check_shape=True):
            if not isinstance(value, base._ACCEPTED_CLASSES):
                return self._convert_to_tensor(value)
            return value

        def _validate_value_batchfree_devicefree(self, value, non_blocking=False, *, check_shape=True):
            return self._validate_value_batchfree(value, non_blocking, check_shape=check_shape)

    base.TensorDictBase = TensorDictBase
    alias = types.ModuleType("tensordict._td")
    alias._ACCEPTED_CLASSES = (Dense, TensorDictBase)
    root.base = base
    return root, base, alias
