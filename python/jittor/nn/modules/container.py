"""Neural-network module containers."""

import collections
import types

import jittor as jt


class Sequential(jt.Module):
    def __init__(self, *args):
        self.layers = collections.OrderedDict()
        for mod in args:
            if mod is None:
                continue  # torch: ModuleList(None) -> empty
            if isinstance(mod, collections.OrderedDict):
                for key, module in mod.items():
                    self.add_module(key, module)
            elif isinstance(mod, (list, tuple, types.GeneratorType)) or (
                hasattr(mod, "__iter__") and not isinstance(mod, jt.Module)
            ):
                # torch's ModuleList accepts ANY iterable of modules (incl. a
                # generator, e.g. DINO: ModuleList(build(l) for l in layers)).
                for module in mod:
                    self.append(module)
            else:
                self.append(mod)

    def __getitem__(self, idx):
        if isinstance(idx, slice) or idx not in self.layers:
            return list(self.layers.values())[idx]
        return self.layers[idx]

    def __iter__(self):
        return self.layers.values().__iter__()

    def keys(self):
        return self.layers.keys()

    def values(self):
        return self.layers.values()

    def items(self):
        return self.layers.items()

    def execute(self, x):
        for key, layer in self.layers.items():
            x = layer(x)
        return x

    def dfs(self, parents, k, callback, callback_leave, recurse=True):
        n_children = len(self.layers)
        ret = callback(parents, k, self, n_children)
        if ret == False:  # noqa: E712
            return
        parents.append(self)
        if recurse:
            for key, value in self.layers.items():
                if isinstance(value, jt.Module):
                    value.dfs(parents, key, callback, callback_leave)
        parents.pop()
        if callback_leave:
            callback_leave(parents, k, self, n_children)

    def append(self, mod):
        # torch's ModuleList stores None children (e.g. HRNet's _make_fuse_layers
        # appends None for the identity/same-resolution path and checks `is not None`
        # in forward). Accept None as a placeholder instead of asserting.
        if mod is None:
            self.layers[str(len(self.layers))] = None
            return self
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), "Module is not a type"
        self.layers[str(len(self.layers))] = mod
        return self

    def extend(self, mods):
        # torch.nn.ModuleList.extend: append every module from an iterable
        # (mmdet PVT does `layers.extend([...])` when assembling backbone stages).
        for module in mods:
            self.append(module)
        return self

    def insert(self, index, mod):
        # torch.nn.ModuleList.insert: insert before `index`, shifting the
        # (string-keyed) tail. Rebuild the OrderedDict with contiguous int keys.
        assert callable(mod) and not isinstance(mod, type)
        values = list(self.layers.values())
        count = len(values)
        if index < 0:
            index += count
        index = max(0, min(index, count))
        values.insert(index, mod)
        self.layers = collections.OrderedDict((str(i), value) for i, value in enumerate(values))
        return self

    def add_module(self, name, mod):
        assert callable(mod), f"Module <{type(mod)}> is not callable"
        assert not isinstance(mod, type), "Module is not a type"
        self.layers[str(name)] = mod

    def __len__(self):
        return len(self.layers)

    def named_children(self):
        return list(self.layers.items())

    @property
    def _modules(self):
        return self.layers

    def __setattr__(self, key, value) -> None:
        if isinstance(key, str) and key.isdigit():
            if int(key) < len(self.layers):
                self.add_module(key, value)
            else:
                super().__setattr__(key, value)
        else:
            super().__setattr__(key, value)

    def __getattr__(self, key):
        if "layers" in self.__dict__ and key in self.__dict__["layers"]:
            return self.__dict__["layers"][key]
        return super().__getattr__(key)


__all__ = ["Sequential"]
