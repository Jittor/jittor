"""Per-construction graph rewriting; external objects are never rewritten."""
from jittor._core.dtypes import dtype_name
from typing import Any


class ChildAdoption:
    def __init__(self, owner, module, external):
        self.owner = owner
        self.external = external
        self.protected = set(external)
        self.memo = {id(module): module}

    def adapt_value(self, value):
        if id(value) in self.external:
            return value
        if id(value) in self.memo:
            return self.memo[id(value)]
        owner = self.owner
        if isinstance(value, owner.native_module):
            # Shared native children retain all tensor identities, even when
            # those tensors are also exposed directly on the enclosing module.
            self.protected.update(owner.external_objects((value,), {}))
            if isinstance(value, owner.Module):
                return value
            native_type = type(value)
            if not native_type.__module__.startswith("jittor.nn.modules."):
                return value
            result: Any = object.__new__(owner.adapt_class(native_type))
            self.memo[id(value)] = result
            for name, item in vars(value).items():
                setattr(result, name, self.adapt_value(item))
            return result
        if isinstance(value, dict):
            rebuilt: Any = value.copy()
            self.memo[id(value)] = rebuilt
            for name, item in value.items():
                rebuilt[name] = self.adapt_value(item)
            return rebuilt
        if isinstance(value, list):
            result = []
            self.memo[id(value)] = result
            result.extend(self.adapt_value(item) for item in value)
            return result
        if isinstance(value, tuple):
            items = tuple(self.adapt_value(item) for item in value)
            if all(item is original for item, original in zip(items, value)):
                return value
            if type(value) is tuple:
                result = items
            elif hasattr(value, "_fields"):
                result = type(value)(*items)
            else:
                return value
            self.memo[id(value)] = result
            return result
        return value


class ParameterRewrite:
    def __init__(self, protected, replacements):
        self.protected = protected
        self.replacements = replacements
        self.memo = {}

    def replace(self, value):
        if id(value) in self.protected:
            return value
        if id(value) in self.replacements:
            return self.replacements[id(value)]
        if id(value) in self.memo:
            return self.memo[id(value)]
        if isinstance(value, dict):
            rebuilt: Any = value.copy()
            self.memo[id(value)] = rebuilt
            for key, item in value.items():
                rebuilt[key] = self.replace(item)
            return rebuilt
        if isinstance(value, list):
            rebuilt = []
            self.memo[id(value)] = rebuilt
            rebuilt.extend(self.replace(item) for item in value)
            return rebuilt
        if isinstance(value, tuple):
            items = tuple(self.replace(item) for item in value)
            if all(item is old for item, old in zip(items, value)):
                return value
            if type(value) is tuple:
                rebuilt = items
            elif hasattr(value, "_fields"):
                rebuilt = type(value)(*items)
            else:
                return value
            self.memo[id(value)] = rebuilt
            return rebuilt
        return value


def _adapt_batchnorm_counter(owner, module):
    """Torch owns a persistent scalar int64 counter; native keeps its format.

    Adapted children can be built without calling their initializer. Apply
    this to every owned frontend module after graph adoption, while leaving
    supplied native modules and the original native child objects untouched.
    """
    native = getattr(owner.backend.nn, "BatchNorm", None)
    if (native is None or not isinstance(module, owner.Module)
            or not isinstance(module, native)
            or not getattr(module, "track_running_stats", False)):
        return
    counter = module.num_batches_tracked.cast("int64").reshape(()).stop_grad()
    # ChildAdoption retains sets by identity. Role-name updates on the adapted
    # copy must not change the source native module's checkpoint persistence.
    for name in ("_buffer_names", "_non_persistent_buffer_names",
                 "_parameter_names", "_non_parameter_names"):
        if name in vars(module):
            object.__setattr__(module, name, set(vars(module)[name]))
    module.register_buffer("num_batches_tracked", counter, persistent=True)


def adopt_owned_children(owner, module, external, frozen=False):
    adoption = ChildAdoption(owner, module, external)
    for name, value in tuple(vars(module).items()):
        replacement = adoption.adapt_value(value)
        if replacement is not value:
            setattr(module, name, replacement)
    # Role names may identify ParameterList entries, not Python attributes.
    # Replace by object identity through the owned container graph.
    roles = tuple(module._var_roles())
    adoption.protected.update(id(value) for _, value, role in roles
                              if role in ("buffer", "non_persistent_buffer"))
    replacements = {}
    for _, parameter, role in roles:
        if (role != "parameter" or id(parameter) in adoption.protected
                or not isinstance(parameter, owner.tensor_type)
                or isinstance(parameter, owner.Parameter)):
            continue
        if id(parameter) not in replacements:
            differentiable = dtype_name(parameter.dtype) in (
                "float16", "bfloat16", "float32", "float64", "complex64", "complex128")
            replacements[id(parameter)] = owner.Parameter(
                parameter, requires_grad=not frozen and differentiable)
    if replacements:
        rewrite = ParameterRewrite(adoption.protected, replacements)
        for name, value in tuple(vars(module).items()):
            replacement = rewrite.replace(value)
            if replacement is not value:
                setattr(module, name, replacement)
    for owned in tuple(adoption.memo.values()):
        _adapt_batchnorm_counter(owner, owned)
