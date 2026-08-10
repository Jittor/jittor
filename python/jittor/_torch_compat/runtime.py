"""Runtime binding for torch-compat implementation modules.

Implementation modules are imported while :mod:`jittor` is still composing its
public namespace.  A bound proxy avoids adding more import-time edges back to
the partially initialized package.
"""


class _JittorRuntimeProxy:
    def __init__(self):
        self._module = None

    def bind(self, module):
        self._module = module

    def __getattr__(self, name):
        if self._module is None:
            raise RuntimeError("torch compatibility runtime is not bound")
        return getattr(self._module, name)


jt = _JittorRuntimeProxy()


def bind_runtime(module):
    jt.bind(module)


def preserve_facade_origins(symbols, source_module=None):
    """Keep reflection and pickle metadata stable after implementation moves."""
    pending = list(symbols)
    visited = set()
    while pending:
        symbol = pending.pop()
        if id(symbol) in visited:
            continue
        visited.add(id(symbol))
        module_name = getattr(symbol, "__module__", None)
        if source_module is None:
            should_update = (
                isinstance(module_name, str)
                and module_name.startswith("jittor._torch_compat")
            )
        else:
            should_update = module_name == source_module
        if should_update:
            symbol.__module__ = "jittor.torch_compat"
            if isinstance(symbol, type):
                for member in vars(symbol).values():
                    if isinstance(member, (classmethod, staticmethod)):
                        pending.append(member.__func__)
                    elif isinstance(member, property):
                        pending.extend(
                            value for value in (member.fget, member.fset, member.fdel)
                            if value is not None
                        )
                    elif callable(member):
                        pending.append(member)
