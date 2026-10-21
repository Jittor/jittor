"""Boundary trees and isolated callable bindings for activation replay."""

import dataclasses
import functools
import types
from contextlib import contextmanager

import jittor as jt
from jittor._core.module import Module, _walk


def flatten(value, tensors, memo):
    if isinstance(value, jt.Var):
        key = value.id
        if key not in memo:
            memo[key] = len(tensors)
            tensors.append(value)
        return ("tensor", memo[key])
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        fields = tuple(f.name for f in dataclasses.fields(value) if f.init)
        return ("dataclass", type(value), fields,
                tuple(flatten(getattr(value, f), tensors, memo) for f in fields))
    if isinstance(value, dict):
        return ("dict", type(value), tuple(value),
                tuple(flatten(v, tensors, memo) for v in value.values()))
    if isinstance(value, (tuple, list)):
        return ("sequence", type(value), tuple(flatten(v, tensors, memo) for v in value))
    return ("constant", value)


def unflatten(spec, tensors):
    kind = spec[0]
    if kind == "tensor":
        return tensors[spec[1]]
    if kind == "constant":
        return spec[1]
    if kind == "dict":
        return spec[1](zip(spec[2], (unflatten(s, tensors) for s in spec[3])))
    if kind == "dataclass":
        return spec[1](**dict(zip(spec[2], (unflatten(s, tensors) for s in spec[3]))))
    values = [unflatten(s, tensors) for s in spec[2]]
    if hasattr(spec[1], "_fields"):
        return spec[1](*values)
    return spec[1](values)


def structure(spec):
    """Compare structure without invoking tensor equality on constant leaves."""
    if spec[0] in ("tensor", "constant"):
        return spec[0] if spec[0] == "constant" else spec
    return spec[:-1] + (tuple(structure(s) for s in spec[-1]),)


def _contains_tensor(value, replacements, seen):
    # Keep recursion at module scope: a local recursive function closes over
    # itself and replacements, retaining replay tensors until cyclic GC.
    if isinstance(value, jt.Var):
        return value.id in replacements
    if id(value) in seen:
        return False
    seen.add(id(value))
    if isinstance(value, dict):
        return any(_contains_tensor(v, replacements, seen) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_tensor(v, replacements, seen) for v in value)
    return False


class Capture:
    def __init__(self, function, args, kwargs):
        self.function = function
        self.values = []
        self.memo = {}
        self.spec = flatten((args, kwargs), self.values, self.memo)
        self.slots = []
        self.buffer_slots = []
        self._seen = set()
        self._slot_seen = set()
        self._visit(function)

    def _add(self, value):
        flatten(value, self.values, self.memo)

    def _visit(self, value):
        if id(value) in self._seen:
            return
        self._seen.add(id(value))
        if isinstance(value, jt.Var):
            self._add(value)
        elif isinstance(value, Module):
            for _, owner in _walk(value):
                for name, tensor, role in owner._var_roles():
                    key = (id(owner), name)
                    if key in self._slot_seen:
                        continue
                    self._slot_seen.add(key)
                    if role == "parameter":
                        self._add(tensor)
                        self.slots.append((owner, name, tensor.id))
                    elif role in ("buffer", "non_persistent_buffer"):
                        self.buffer_slots.append((owner, name, tensor))
        elif isinstance(value, functools.partial):
            self._visit(value.func)
            self._visit(value.args)
            self._visit(value.keywords)
        elif isinstance(value, types.MethodType):
            self._visit(value.__self__)
            self._visit(value.__func__)
        elif isinstance(value, types.FunctionType):
            self._visit(value.__defaults__)
            self._visit(value.__kwdefaults__)
            for cell in value.__closure__ or ():
                self._visit(cell.cell_contents)
            for name in value.__code__.co_names:
                candidate = value.__globals__.get(name)
                if isinstance(candidate, (jt.Var, Module, tuple, list, dict)):
                    self._visit(candidate)
        elif isinstance(value, dict):
            for item in value.values():
                self._visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                self._visit(item)

    def prepare_buffers(self):
        # A materialized arithmetic copy owns storage even on native clone's
        # sharing path. The replay copy must not alias a mutable live buffer.
        self.buffers = {}
        with jt.no_grad():
            for _, _, value in self.buffer_slots:
                if value.id not in self.buffers:
                    copy = value + jt.zeros_like(value)
                    copy.sync()
                    self.buffers[value.id] = copy

    def _rebind(self, value, replacements, memo):
        if isinstance(value, jt.Var):
            return replacements.get(value.id, value)
        if id(value) in memo:
            return memo[id(value)]
        if isinstance(value, Module):
            return value
        if isinstance(value, (dict, list, tuple)):
            if not _contains_tensor(value, replacements, set()):
                return value
        if isinstance(value, types.MethodType):
            return types.MethodType(self._rebind(value.__func__, replacements, memo), value.__self__)
        if isinstance(value, functools.partial):
            return functools.partial(self._rebind(value.func, replacements, memo),
                                     *self._rebind(value.args, replacements, memo),
                                     **self._rebind(value.keywords or {}, replacements, memo))
        if isinstance(value, types.FunctionType):
            namespace = value.__globals__
            changed = {name: self._rebind(namespace[name], replacements, memo)
                       for name in value.__code__.co_names
                       if name in namespace and isinstance(namespace[name], (jt.Var, tuple, list, dict))}
            if changed:
                namespace = dict(namespace, **changed)
            def cell(item):
                return (lambda: item).__closure__[0]
            closure = tuple(cell(self._rebind(c.cell_contents, replacements, memo))
                            for c in value.__closure__ or ()) or None
            result = types.FunctionType(value.__code__, namespace, value.__name__,
                                        self._rebind(value.__defaults__, replacements, memo), closure)
            result.__kwdefaults__ = self._rebind(value.__kwdefaults__, replacements, memo)
            memo[id(value)] = result
            return result
        if isinstance(value, dict):
            result = type(value)()
            memo[id(value)] = result
            result.update((k, self._rebind(v, replacements, memo)) for k, v in value.items())
            return result
        if isinstance(value, (tuple, list)):
            parts = [self._rebind(v, replacements, memo) for v in value]
            return type(value)(*parts) if hasattr(value, "_fields") else type(value)(parts)
        return value

    @contextmanager
    def replay(self, values):
        replacements = {old.id: new for old, new in zip(self.values, values)}
        saved = []
        try:
            for owner, name, key in self.slots:
                saved.append((owner, name, vars(owner)[name]))
                vars(owner)[name] = replacements[key]
            # Restore original objects afterwards, including any once-updated
            # running buffers from the first forward. Replay changes only copies.
            with jt.no_grad():
                buffers = {key: value + jt.zeros_like(value) for key, value in self.buffers.items()}
            for owner, name, original in self.buffer_slots:
                saved.append((owner, name, vars(owner)[name]))
                vars(owner)[name] = buffers[original.id]
            yield self._rebind(self.function, replacements, {})
        finally:
            for owner, name, previous in reversed(saved):
                vars(owner)[name] = previous
