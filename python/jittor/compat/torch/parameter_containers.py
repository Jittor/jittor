"""Installation-owned parameter containers without mutating source tensors."""

from collections import OrderedDict
from collections.abc import Mapping


def make_parameter_containers(module_type, parameter_type, tensor_base):
    def parameter(value):
        if isinstance(value, parameter_type):
            return value
        if isinstance(value, tensor_base):
            return parameter_type(value)
        return value

    class ParameterList(module_type):
        def __init__(self, values=None):
            super().__init__()
            self._values = []
            if values is not None:
                self.extend(values)

        def _var_attrs(self):
            return [(str(index), value) for index, value in enumerate(self._values)
                    if isinstance(value, tensor_base)]

        def __len__(self):
            return len(self._values)

        def __iter__(self):
            return iter(self._values)

        def __getitem__(self, index):
            if isinstance(index, slice):
                return type(self)(self._values[index])
            return self._values[index]

        def __getattr__(self, name):
            values = vars(self).get("_values", ())
            if name.isdigit() and int(name) < len(values):
                return values[int(name)]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            values = vars(self).get("_values")
            if values is not None and name.isdigit() and int(name) < len(values):
                values[int(name)] = parameter(value)
                return
            super().__setattr__(name, value)

        def __setitem__(self, index, value):
            if isinstance(index, slice):
                raise TypeError("ParameterList assignment requires an integer index")
            self._values[index] = parameter(value)

        def append(self, value):
            self._values.append(parameter(value))
            return self

        def extend(self, values):
            if isinstance(values, tensor_base):
                raise TypeError("ParameterList.extend expects an iterable of parameters")
            for value in values:
                self.append(value)
            return self

        def insert(self, index, value):
            self._values.insert(index, parameter(value))
            return self

        def __iadd__(self, values):
            return self.extend(values)

    class ParameterDict(module_type):
        def __init__(self, values=None):
            super().__init__()
            self._values = OrderedDict()
            if values is not None:
                self.update(values)

        def _var_attrs(self):
            return [(key, value) for key, value in self._values.items()
                    if isinstance(value, tensor_base)]

        def __len__(self):
            return len(self._values)

        def __iter__(self):
            return iter(self._values)

        def __contains__(self, key):
            return key in self._values

        def __getitem__(self, key):
            return self._values[key]

        def __getattr__(self, name):
            values = vars(self).get("_values", {})
            if name in values:
                return values[name]
            raise AttributeError(name)

        def __setattr__(self, name, value):
            values = vars(self).get("_values")
            if values is not None and name in values:
                values[name] = parameter(value)
                return
            super().__setattr__(name, value)

        def __setitem__(self, key, value):
            if not isinstance(key, str):
                raise TypeError("ParameterDict keys must be strings")
            if not key or "." in key:
                raise KeyError("ParameterDict keys must be nonempty and contain no dots")
            if key not in self._values and hasattr(self, key):
                raise KeyError("ParameterDict key conflicts with an existing attribute: " + key)
            self._values[key] = parameter(value)

        def __delitem__(self, key):
            del self._values[key]

        def keys(self):
            return self._values.keys()

        def values(self):
            return self._values.values()

        def items(self):
            return self._values.items()

        def get(self, key, default=None):
            return self._values.get(key, default)

        def update(self, values):
            entries = values.items() if isinstance(values, (Mapping, ParameterDict)) else values
            for key, value in entries:
                self[key] = value

        def clear(self):
            self._values.clear()

        def pop(self, key):
            return self._values.pop(key)

        def setdefault(self, key, default=None):
            if key not in self:
                self[key] = default
            return self[key]

        def copy(self):
            return type(self)(self.items())

    for cls in (ParameterList, ParameterDict):
        cls.__module__ = "torch.nn.modules.parameter"
        cls.__qualname__ = cls.__name__
    return ParameterList, ParameterDict
