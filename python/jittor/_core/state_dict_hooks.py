"""Module state hooks, independent of tensor placement and frontend policy."""

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import itertools

from .hooks import _RemovableHandle


_enabled = ContextVar("jittor_module_state_hooks", default=True)
_serial = itertools.count()


@contextmanager
def scope(enabled):
    token = _enabled.set(enabled)
    try:
        yield
    finally:
        _enabled.reset(token)


def register(module, table_name, hook, **options):
    if not callable(hook):
        raise TypeError("state hook must be callable")
    table = module.__dict__.setdefault(table_name, OrderedDict())
    info = module.__dict__.setdefault("_state_dict_hook_options", {})
    key = next(_serial)
    table[key] = hook
    info[table_name, key] = options

    def remove():
        table.pop(key, None)
        info.pop((table_name, key), None)

    return _RemovableHandle(remove)


def _hooks(module, table_name):
    info = module.__dict__.get("_state_dict_hook_options", {})
    return [(hook, info.get((table_name, key), {})) for key, hook in
            tuple(module.__dict__.get(table_name, {}).items())]


def _postorder(root, recurse):
    from .module import _walk

    pending = []
    for name, module in _walk(root, recurse):
        depth = name.count(".") + 1 if name else 0
        while pending and pending[-1][0] >= depth:
            _, old_name, old_module = pending.pop()
            yield old_name, old_module
        pending.append((depth, name, module))
    while pending:
        _, name, module = pending.pop()
        yield name, module


def _metadata(state, prefix, module):
    metadata = getattr(state, "_metadata", {})
    return metadata.get(prefix[:-1], {"version": getattr(module, "_version", 1)})


def apply_post(root, state, prefix, recurse):
    if not _enabled.get():
        return state
    for name, module in _postorder(root, recurse):
        path = prefix + name + ("." if name else "")
        for hook, options in _hooks(module, "_state_dict_hooks"):
            result = hook(module, state, path, _metadata(state, path, module))
            if result is not None:
                if not options.get("allow_return", True):
                    raise RuntimeError("state_dict post-hook must return None")
                if not isinstance(result, Mapping):
                    raise TypeError("state_dict hook must return a mapping or None")
                state = result
    return state


def prepare_load(root, state, strict):
    if not _enabled.get():
        return state, [], [], []
    from .module import _walk

    copied = OrderedDict(state)
    if hasattr(state, "_metadata"):
        copied._metadata = state._metadata
    missing, unexpected, errors = [], [], []
    for name, module in _walk(root):
        prefix = name + "." if name else ""
        for hook, options in _hooks(module, "_load_state_dict_pre_hooks"):
            args = (copied, prefix, _metadata(copied, prefix, module),
                    strict, missing, unexpected, errors)
            if options.get("with_module", False):
                hook(module, *args)
            else:
                hook(*args)
    return copied, missing, unexpected, errors
