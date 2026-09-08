"""Identity lookup for live Python holders; graph semantics belong to the core."""

import weakref


_ABSENT = object()


class HolderRegistry(dict):
    """Weak independent Tensor entries, with explicit legacy Var ownership.

    Native legacy Vars have no weakref slot. Their existing pruning remains a
    compatibility path; an independent Tensor must never silently take it.
    Keys are identities solely to avoid Tensor equality in this holder index.
    """

    def __setitem__(self, key, value):
        if getattr(type(value), "_frontend_backend", None) is not None:
            registry = weakref.ref(self)

            def expired(reference):
                owner = registry()
                if owner is not None and dict.get(owner, key) is reference:
                    dict.__delitem__(owner, key)

            value = weakref.ref(value, expired)
        dict.__setitem__(self, key, value)

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        if isinstance(value, weakref.ReferenceType):
            value = value()
            if value is None:
                raise KeyError(key)
        return value

    def is_weak(self, key):
        return isinstance(dict.get(self, key), weakref.ReferenceType)

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def items(self):
        result = []
        for key in list(dict.keys(self)):
            value = self.get(key, _ABSENT)
            if value is not _ABSENT:
                result.append((key, value))
        return result

    def keys(self):
        return [key for key, value in self.items()]

    def __iter__(self):
        return iter(self.keys())

    def values(self):
        return [value for key, value in self.items()]

    def copy(self):
        return dict(self.items())

    def update(self, other=(), **kwargs):
        for key, value in (other.items() if hasattr(other, "items") else other):
            self[key] = value
        for key, value in kwargs.items():
            self[key] = value

    def pop(self, key, default=_ABSENT):
        value = self.get(key, _ABSENT)
        if value is _ABSENT:
            if default is _ABSENT:
                raise KeyError(key)
            return default
        dict.__delitem__(self, key)
        return value

    def setdefault(self, key, default=None):
        value = self.get(key, _ABSENT)
        if value is _ABSENT:
            self[key] = value = default
        return value

    def clear_legacy(self):
        for key in list(dict.keys(self)):
            if not self.is_weak(key):
                dict.__delitem__(self, key)
