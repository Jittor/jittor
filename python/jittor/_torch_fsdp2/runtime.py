"""Runtime bindings shared by private FSDP2 compatibility modules."""


class _RuntimeProxy:
    def __init__(self, name):
        self._name = name
        self._module = None

    def bind(self, module):
        self._module = module

    def __getattr__(self, name):
        if self._module is None:
            raise RuntimeError("FSDP2 %s runtime is not bound" % self._name)
        return getattr(self._module, name)


jt = _RuntimeProxy("jittor")
nn = _RuntimeProxy("nn")
fsdp = _RuntimeProxy("facade")
facade = fsdp


def bind_runtime(jittor_module, nn_module=None, facade_module=None):
    """Bind the late-imported runtime modules used by private implementations."""
    jt.bind(jittor_module)
    if nn_module is None:
        nn_module = getattr(jittor_module, "nn", None)
    if nn_module is not None:
        nn.bind(nn_module)
    if facade_module is not None:
        fsdp.bind(facade_module)


def preserve_facade_origins(symbols, source_module=None):
    """Keep reflection and pickle metadata on the historical public module."""
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
                and module_name.startswith("jittor._torch_fsdp2")
            )
        else:
            should_update = module_name == source_module
        if should_update:
            try:
                symbol.__module__ = "jittor.torch_fsdp2_compat"
            except (AttributeError, TypeError):
                pass

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

        wrapped = getattr(symbol, "__wrapped__", None)
        if wrapped is not None:
            pending.append(wrapped)
        function = getattr(symbol, "__func__", None)
        if function is not None:
            pending.append(function)
