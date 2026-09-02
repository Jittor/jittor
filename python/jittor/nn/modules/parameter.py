"""Torch-compatible parameter helpers and containers."""

import collections
import types

import jittor as jt


class ParameterList(jt.Module):
    def __init__(self, *args):
        self.params = collections.OrderedDict()
        for var in args:
            if isinstance(var, (collections.OrderedDict, dict)):
                for key, value in var.items():
                    self.add_param(key, value)
            elif isinstance(var, list):
                for value in var:
                    self.append(value)
            else:
                self.append(var)

    def __getitem__(self, idx):
        if idx not in self.params:
            return list(self.params.values())[idx]
        return self.params[idx]

    def __iter__(self):
        return self.params.values().__iter__()

    def keys(self):
        return self.params.keys()

    def values(self):
        return self.params.values()

    def items(self):
        return self.params.items()

    def execute(self, x):
        raise NotImplementedError("Parameters is not executable")

    def append(self, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        var._is_torch_parameter = True
        self.params[len(self.params)] = var

    def add_param(self, name, var):
        assert isinstance(var, jt.Var), f"argument <{type(var)}> is not jittor var"
        var._is_torch_parameter = True
        self.params[name] = var

    def __setitem__(self, name, var):
        self.add_param(name, var)

    def __len__(self):
        return len(self.params)


def _make_parameter(data, requires_grad=True):
    """Torch-compatible Parameter wrapper.

    Jittor treats a Var assigned to a Module as a parameter, so wrapping an
    existing Var only needs to set the trainable flag. Do not clone here:
    PyTorch's Parameter is a lightweight wrapper over the supplied tensor data,
    while cloning can force materialization/JIT work and makes large pretrained
    model construction unnecessarily slow.
    """
    if not isinstance(data, jt.Var):
        data = jt.array(data)
    data.requires_grad = requires_grad
    data._is_torch_parameter = True
    return data


def _make_buffer(data, persistent=True):
    """Torch-compatible Buffer wrapper.

    Marking the Var is enough to give it the buffer role: ``Module.__setattr__`` keeps a Var
    tagged ``is_buffer`` out of the parameter set, and ``named_buffers`` reports it even when
    the attribute name was never passed to ``register_buffer``. As with Parameter, the wrapper
    shares the supplied data instead of cloning it.
    """
    if data is None:
        data = jt.empty(0)
    if not isinstance(data, jt.Var):
        data = jt.array(data)
    data.is_buffer = True
    data.persistent = bool(persistent)
    data._is_torch_parameter = False
    return data


def _run_subclass_init(cls, var, args, kwargs):
    """Run a Parameter subclass's ``__init__`` and keep the state it sets.

    ``self`` cannot be the Var. A zero-argument ``super()`` inside ``__init__``
    checks the real type of its first argument, and that check does not go
    through ``__instancecheck__`` -- so passing the Var raises before the body
    runs. Instead ``__init__`` runs against a bare instance of the class, where
    ``super()`` resolves normally, and the attributes it set are copied over
    afterwards.

    An ``__init__`` that needs its object to behave like a tensor cannot be run
    this way; then the keyword arguments are attached directly, which is what a
    parameter subclass carrying loader metadata wants in any case.
    """
    initializer = cls.__init__
    if initializer is object.__init__:
        return
    try:
        stand_in = object.__new__(cls)
        initializer(stand_in, *args, **kwargs)
    except Exception:
        for name, value in kwargs.items():
            var.__dict__.setdefault(name, value)
        return
    var.__dict__.update(stand_in.__dict__)


def _adopt_class_members(cls, var):
    """Give a constructed parameter its class's methods and properties.

    A Parameter here is a plain Var carrying a marker, not an instance of the
    class, so the usual attribute lookup never reaches the class body. Methods
    are bound onto the Var; properties are read once, after ``__init__`` has had
    its chance to set whatever they are computed from. Both are best-effort --
    a property that needs state this object does not have is simply skipped, and
    anything ``__init__`` already stored keeps precedence.

    Deliberately not solved with a ``__getattr__`` on Var: that fires on every
    failed attribute lookup on every Var in the process, and a decode step does
    tens of thousands of them.
    """
    for klass in cls.__mro__:
        if klass in (Parameter, object):
            continue
        for name, member in vars(klass).items():
            if name.startswith("__") or name in var.__dict__:
                continue
            try:
                if isinstance(member, property):
                    if member.fget is not None:
                        var.__dict__[name] = member.fget(var)
                elif callable(member):
                    var.__dict__[name] = types.MethodType(member, var)
            except Exception:
                continue


class _ParameterMeta(type):
    def __instancecheck__(cls, obj):
        if not isinstance(obj, jt.Var):
            return False
        if not bool(getattr(obj, "_is_torch_parameter", False)):
            return False
        if cls is Parameter:
            return True
        # A typed subclass must not answer for every parameter: vLLM routes
        # weight loading on `isinstance(param, SomeParameterSubclass)`, and a
        # blanket True there sends plain weights down a quantised path.
        built = obj.__dict__.get("_torch_parameter_class")
        return built is not None and issubclass(built, cls)

    def __call__(cls, *args, **kwargs):
        # Run the construction protocol rather than short-circuiting it: torch
        # code subclasses Parameter, gives the subclass its own __new__/__init__
        # and extra keyword arguments, and expects both to run. Overriding this
        # instead of relying on type.__call__ buys the step after __init__,
        # where class members can be adopted onto the finished Var.
        var = cls.__new__(cls, *args, **kwargs)
        if not isinstance(var, jt.Var) or cls is Parameter:
            return var
        _run_subclass_init(cls, var, args, kwargs)
        _adopt_class_members(cls, var)
        return var


class Parameter(metaclass=_ParameterMeta):
    """Semantic parameter role backed by a marked :class:`jittor.Var`."""

    def __new__(cls, data=None, requires_grad=True, **kwargs):
        var = _make_parameter(data, requires_grad=requires_grad)
        if cls is not Parameter:
            var.__dict__["_torch_parameter_class"] = cls
        return var

    def __init__(self, *args, **kwargs):
        """Accept whatever __new__ accepted; the Var is already built."""


class _BufferMeta(type):
    def __instancecheck__(cls, obj):
        return isinstance(obj, jt.Var) and bool(getattr(obj, "is_buffer", False))


class Buffer(metaclass=_BufferMeta):
    """Semantic buffer role backed by a marked :class:`jittor.Var`.

    Assigning one to a module is the declarative form of ``register_buffer``: state that moves
    and serialises with the module but is never handed to an optimizer.
    """

    def __new__(cls, data=None, *, persistent=True):
        return _make_buffer(data, persistent=persistent)

    def __init__(self, *args, **kwargs):
        """Accept whatever __new__ accepted; the Var is already built."""


__all__ = ["Buffer", "Parameter", "ParameterList"]
