"""Host support shared by the verl adapter tests.

Fake host leaves and the pinned fixture path, kept out of the test modules so
pytest can collect every adapter suite in one session without one test module
importing another. No Jittor import, tensor arithmetic, or hardware claim.
"""
import importlib
import os
from pathlib import Path
import sys
import types


ADAPTERS = Path(__file__).resolve().parents[3]
ROOT = ADAPTERS.parent
sys.path.insert(0, str(ADAPTERS))
sys.path.insert(0, str(ADAPTERS / "tests"))

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
