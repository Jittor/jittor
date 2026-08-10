"""Runtime binding shared by private ``jittor.nn`` implementation modules."""


class _JittorRuntimeProxy:
    def __init__(self):
        self._module = None

    def bind(self, module):
        self._module = module

    def __getattr__(self, name):
        if self._module is None:
            raise RuntimeError("nn runtime is not bound")
        return getattr(self._module, name)


jt = _JittorRuntimeProxy()


def bind_runtime(module):
    jt.bind(module)


def preserve_facade_origins(symbols):
    """Keep reflection and pickle metadata stable after implementation moves."""
    pending = list(symbols)
    visited = set()
    while pending:
        symbol = pending.pop()
        if id(symbol) in visited:
            continue
        visited.add(id(symbol))
        module_name = getattr(symbol, "__module__", None)
        if isinstance(module_name, str) and module_name.startswith("jittor._nn"):
            symbol.__module__ = "jittor.nn"
        wrapped = getattr(symbol, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
