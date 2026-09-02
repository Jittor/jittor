"""Torch operator-library schema compatibility.

This module owns metadata-only helpers used by the dynamically published
``torch.library`` module.  Operator execution and dispatch remain owned by the
installer that publishes the active Torch namespace.
"""

from __future__ import absolute_import

import collections
import enum
import inspect
import types
import typing
from ..diagnostics import EXPECTED, swallowed


_EMPTY = inspect.Parameter.empty
_UNKNOWN_MUTATES = "unknown"


class Tag(enum.Enum):
    """Operator metadata tags exposed by ``torch.Tag``."""

    core = 0
    cudagraph_unsafe = 1
    data_dependent_output = 2
    dynamic_output_shape = 3
    flexible_layout = 4
    generated = 5
    inplace_view = 6
    maybe_aliasing_or_mutating = 7
    needs_contiguous_strides = 8
    needs_exact_strides = 9
    needs_fixed_stride_order = 10
    nondeterministic_bitwise = 11
    nondeterministic_seeded = 12
    pointwise = 13
    pt2_compliant_tag = 14
    reduction = 15
    view_copy = 16


# Dispatch keys, in the order torch's dispatcher would consider them for a real
# (non-meta) tensor of each residency.  "Meta" is deliberately absent from both
# lists: a meta kernel returns a correctly-shaped tensor holding NOTHING, so
# serving a real call from it produces fake numbers silently.
_CUDA_DISPATCH_ORDER = (
    "CUDA", "AutogradCUDA", "PrivateUse1", "AutogradPrivateUse1",
    "CompositeExplicitAutograd", "CompositeExplicitAutogradNonFunctional",
    "CompositeImplicitAutograd", "Autograd", "BackendSelect", "",
)
_CPU_DISPATCH_ORDER = (
    "CPU", "AutogradCPU",
    "CompositeExplicitAutograd", "CompositeExplicitAutogradNonFunctional",
    "CompositeImplicitAutograd", "Autograd", "BackendSelect", "",
)
_FAKE_ONLY_KEYS = ("Meta", "FakeTensor", "SparseCPU", "SparseCUDA")


def _argument_residency(args, kwargs):
    """"cuda" or "cpu", from the first tensor argument -- as torch dispatches."""
    import jittor as jt

    from .types import _var_is_cpu_resident

    def walk(value):
        if isinstance(value, jt.Var):
            return "cpu" if _var_is_cpu_resident(value) else "cuda"
        if isinstance(value, (list, tuple)):
            for item in value:
                found = walk(item)
                if found is not None:
                    return found
        return None

    for value in list(args) + list((kwargs or {}).values()):
        found = walk(value)
        if found is not None:
            return found
    return "cuda" if getattr(jt.flags, "use_cuda", 0) else "cpu"


class _AutogradContext:
    """The ``ctx`` handed to a torch.library register_autograd backward."""

    def __init__(self):
        self._saved_tensors = ()
        self._saved_versions = ()

    def save_for_backward(self, *tensors):
        import jittor as jt

        self._saved_tensors = tuple(tensors)
        self._saved_versions = tuple(
            (t.id if isinstance(t, jt.Var) else None) for t in tensors)

    @property
    def saved_tensors(self):
        import jittor as jt

        for tensor, version in zip(self._saved_tensors, self._saved_versions):
            if isinstance(tensor, jt.Var) and version is not None \
                    and tensor.id != version:
                raise RuntimeError(
                    "one of the variables needed for gradient computation has "
                    "been modified by an inplace operation")
        return self._saved_tensors


def _call_with_registered_autograd(op, function, args, kwargs):
    """Run the forward inside a jt.Function so the registered backward runs.

    ``register_autograd`` used to write ``op._backward`` and ``op._setup_context``
    and nothing in the tree ever read them again.  A custom operator whose
    forward leaves the tape (any op implemented with numpy, a C++ extension, or
    an explicit stop_grad) therefore produced gradients of exactly zero, with
    no error -- the model trained, and that operator never learned.
    """
    import jittor as jt

    if getattr(jt.flags, "no_grad", 0):
        return function(*args, **kwargs)
    slots = [i for i, value in enumerate(args)
             if isinstance(value, jt.Var) and value.requires_grad]
    if not slots:
        return function(*args, **kwargs)

    class _LibraryAutograd(jt.Function):
        def execute(self, *taped):
            full = list(args)
            for slot, value in zip(slots, taped):
                full[slot] = value
            self._ctx = _AutogradContext()
            output = function(*full, **kwargs)
            if op._setup_context is not None:
                op._setup_context(self._ctx, tuple(full), output)
            return output

        def grad(self, *grads):
            produced = op._backward(self._ctx, *grads)
            if not isinstance(produced, (tuple, list)):
                produced = (produced,)
            picked = []
            for slot in slots:
                picked.append(produced[slot] if slot < len(produced) else None)
            return tuple(picked)

    return _LibraryAutograd.apply(*[args[slot] for slot in slots])


class _RegisteredOp:
    def __init__(self, namespace, name):
        self.namespace = namespace
        self.name = name
        self.default = self
        self._schema = None
        self._tags = ()
        self._implementations = {}
        self._fake_impl = None
        self._backward = None
        self._setup_context = None
        self._overridden_by_integration = None

    def register_impl(self, dispatch_key, function, allow_override=False):
        key = str(dispatch_key or "CompositeExplicitAutograd")
        if key in self._implementations and not allow_override:
            raise RuntimeError(
                "operator %s::%s already has an implementation for %s"
                % (self.namespace, self.name, key)
            )
        self._implementations[key] = function

    def select_impl(self, args=(), kwargs=None):
        """Pick the kernel torch's dispatcher would pick for these arguments.

        This used to be ``next(reversed(list(self._implementations.values())))``
        -- the LAST registration, whatever key it carried.  So registering
        "CPU" and then "CUDA" ran the CUDA kernel on CPU tensors, and the very
        common ``impl(..., ("CPU", "CUDA", "Meta"))`` put the *meta* kernel
        last, which meant every real call returned an empty fake tensor.
        """
        impls = self._implementations
        if not impls:
            raise NotImplementedError(
                "operator %s::%s has no Jittor implementation"
                % (self.namespace, self.name))
        residency = _argument_residency(args, kwargs)
        order = _CUDA_DISPATCH_ORDER if residency == "cuda" else _CPU_DISPATCH_ORDER
        for key in order:
            if key in impls:
                return impls[key]
        registered = sorted(k or "<default>" for k in impls)
        if all(k in _FAKE_ONLY_KEYS for k in impls):
            raise NotImplementedError(
                "operator %s::%s only has %s kernel(s) registered. Those "
                "produce correctly-shaped tensors with meaningless contents; "
                "running one for a real %s tensor would silently return fake "
                "numbers. Register a %s (or CompositeExplicitAutograd) kernel."
                % (self.namespace, self.name, ", ".join(registered),
                   residency, residency.upper()))
        raise NotImplementedError(
            "operator %s::%s has no kernel for %s tensors; registered keys are "
            "%s" % (self.namespace, self.name, residency, ", ".join(registered)))

    def __call__(self, *args, **kwargs):
        function = self.select_impl(args, kwargs)
        if self._backward is None:
            return function(*args, **kwargs)
        return _call_with_registered_autograd(self, function, args, kwargs)

    def overloads(self):
        return ["default"]

    def __repr__(self):
        return "%s.%s" % (self.namespace, self.name)


class _OpNamespace:
    def __init__(self, namespace):
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(self, "_ops", {})

    def get_or_create(self, name):
        ops = object.__getattribute__(self, "_ops")
        if name not in ops:
            ops[name] = _RegisteredOp(object.__getattribute__(self, "_namespace"), name)
        return ops[name]

    def __getattr__(self, name):
        ops = object.__getattribute__(self, "_ops")
        if name in ops:
            return ops[name]
        raise AttributeError(
            "torch.ops.%s has no op '%s'" % (object.__getattribute__(self, "_namespace"), name)
        )


class _OpsDispatcher:
    def __init__(self, base):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_namespaces", {})

    def get_or_create(self, namespace, name):
        namespaces = object.__getattribute__(self, "_namespaces")
        module = namespaces.setdefault(namespace, _OpNamespace(namespace))
        return module.get_or_create(name)

    def __getattr__(self, name):
        namespaces = object.__getattribute__(self, "_namespaces")
        if name in namespaces:
            return namespaces[name]
        base = object.__getattribute__(self, "_base")
        if base is not None:
            try:
                return getattr(base, name)
            except AttributeError as exc:
                swallowed("torch/library.py __getattr__: return getattr(base, name)", exc)
        namespace = _OpNamespace(name)
        namespaces[name] = namespace
        return namespace


def _operator_name(schema_or_name):
    name = str(schema_or_name).split("(", 1)[0]
    if "::" in name:
        _, name = name.split("::", 1)
    return name.split(".", 1)[0].strip()


def _integration_custom_op_overrides():
    """Integration-supplied replacements for ops downstream libraries register.

    Kept out of this module on purpose: a generic registration API must not
    know any model's operator names.
    """
    try:
        from ..integrations import custom_op_overrides

        return custom_op_overrides()
    except EXPECTED as exc:
        swallowed("torch/library.py _integration_custom_op_overrides: from ..integrations import custom_op_overrides", exc)
        return {}


def install_torch_library(torch_module, modules):
    """Publish one executable ``torch.library`` and ``torch.ops`` surface."""
    library_module = types.ModuleType("torch.library")
    dispatcher = getattr(torch_module, "ops", None)
    if not isinstance(dispatcher, _OpsDispatcher):
        dispatcher = _OpsDispatcher(dispatcher)

    class Library:
        def __init__(self, namespace, kind, dispatch_key=""):
            self.ns = str(namespace)
            self.kind = str(kind)
            self.dispatch_key = str(dispatch_key)

        def _op(self, op_name):
            return dispatcher.get_or_create(self.ns, _operator_name(op_name))

        def define(self, schema, alias_analysis="", *, tags=()):
            if "(" not in str(schema):
                raise ValueError("operator schema must contain an argument list")
            op = self._op(schema)
            op._schema = "%s::%s" % (self.ns, schema)
            op._tags = tuple(tags)
            return op.name

        def impl(self, op_name, fn, dispatch_key="", *, with_keyset=False, allow_override=False):
            if not callable(fn):
                raise TypeError("Library.impl expects a callable implementation")
            key = dispatch_key or self.dispatch_key
            self._op(op_name).register_impl(key, fn, allow_override)
            return None

        def _register_fake(self, op_name, fn, _stacklevel=1, *, allow_override=False):
            op = self._op(op_name)
            if op._fake_impl is not None and not allow_override:
                raise RuntimeError(
                    "operator %s::%s already has a fake implementation" % (self.ns, op.name)
                )
            op._fake_impl = fn
            return None

    def custom_op(name=None, fn=None, *args, **kwargs):
        """torch.library.custom_op -- a generic registration API.

        It used to carry a hard-coded branch comparing ``name`` against one
        specific downstream library's operator, throwing the caller's
        implementation away and substituting the shim's own.  A model-specific
        special case inside a general-purpose registration API silently
        overrides any library that registers that exact name.
        Integration-supplied replacements now come from
        jittor.compat.integrations, keyed by name, and the substitution is
        recorded on the operator as ``_overridden_by_integration``.
        """
        def decorator(implementation):
            if isinstance(name, str) and "::" in name:
                namespace, op_name = name.split("::", 1)
                override = _integration_custom_op_overrides().get(name)
                op = dispatcher.get_or_create(namespace, op_name)
                if override is not None:
                    op._overridden_by_integration = name
                op.register_impl(
                    "CompositeExplicitAutograd", override or implementation,
                    allow_override=True
                )
            return implementation

        return decorator(fn) if fn is not None else decorator

    def register_fake(op, func=None, *, lib=None, **kwargs):
        def decorator(function):
            if isinstance(op, _RegisteredOp):
                op._fake_impl = function
            elif isinstance(op, str) and "::" in op:
                namespace, name = op.split("::", 1)
                target = lib or Library(namespace, "FRAGMENT")
                target._register_fake(name, function, **kwargs)
            return function

        return decorator(func) if func is not None else decorator

    def impl(qualname, dispatch_types, func=None, *, lib=None):
        def decorator(function):
            if "::" not in qualname:
                raise ValueError("operator name must have the form namespace::name")
            namespace, name = qualname.split("::", 1)
            target = lib or Library(namespace, "FRAGMENT")
            keys = (dispatch_types,) if isinstance(dispatch_types, str) else tuple(dispatch_types)
            for key in keys:
                target.impl(name, function, dispatch_key=key)
            return function

        return decorator(func) if func is not None else decorator

    def register_autograd(op, backward, *, setup_context=None, lib=None):
        if not callable(backward):
            raise TypeError("register_autograd expects a callable backward")
        if isinstance(op, _RegisteredOp):
            target = op
        elif isinstance(op, str) and "::" in op:
            namespace, name = op.split("::", 1)
            target = dispatcher.get_or_create(namespace, _operator_name(name))
        else:
            raise TypeError("register_autograd expects an operator or namespace::name")
        target._backward = backward
        target._setup_context = setup_context
        return None

    library_module.Library = Library
    library_module.Tag = Tag
    library_module.custom_op = custom_op
    library_module.infer_schema = make_infer_schema(torch_module)
    library_module.register_fake = register_fake
    library_module.register_kernel = impl
    library_module.impl = impl
    library_module.register_autograd = register_autograd
    library_module.register_torch_dispatch = lambda *a, **k: lambda f: f
    library_module.register_vmap = lambda *a, **k: lambda f: f
    from ..stub_policy import unimplemented_callable as _unimplemented_callable
    library_module.opcheck = _unimplemented_callable(
        "torch.library.opcheck",
        "return None from every operator-correctness check, so a user's "
        "custom-op test suite passes unconditionally whatever the operator does",
        "Test the operator directly against a reference implementation.")
    library_module.get_ctx = lambda: None

    ops_module = types.ModuleType("torch._ops")
    ops_module.OpOverload = _RegisteredOp
    ops_module.OpOverloadPacket = _RegisteredOp
    ops_module.HigherOrderOperator = type("HigherOrderOperator", (), {})
    ops_module.__all__ = [
        "OpOverload", "OpOverloadPacket", "HigherOrderOperator"
    ]

    modules["torch.library"] = library_module
    modules["torch._ops"] = ops_module
    torch_module.library = library_module
    torch_module._ops = ops_module
    torch_module.ops = dispatcher
    torch_module.Tag = Tag
    torch_c = modules.get("torch._C")
    if torch_c is not None:
        torch_c.Tag = Tag
    return library_module


def _typing_origin(annotation):
    get_origin = getattr(typing, "get_origin", None)
    if get_origin is not None:
        return get_origin(annotation)
    return getattr(annotation, "__origin__", None)


def _typing_args(annotation):
    get_args = getattr(typing, "get_args", None)
    if get_args is not None:
        return get_args(annotation)
    return getattr(annotation, "__args__", ())


def _evaluate_annotation(annotation, function, torch_module, error):
    if isinstance(annotation, typing.ForwardRef):
        annotation = annotation.__forward_arg__
    if not isinstance(annotation, str):
        return annotation
    localns = {
        "torch": torch_module,
        "Tensor": torch_module.Tensor,
        "device": torch_module.device,
        "dtype": torch_module.dtype,
        "List": typing.List,
        "Optional": typing.Optional,
        "Sequence": typing.Sequence,
        "Tuple": typing.Tuple,
        "Union": typing.Union,
    }
    try:
        return eval(annotation, function.__globals__, localns)
    except EXPECTED as exc:
        swallowed("torch/library.py _evaluate_annotation: return eval(annotation, function.__globals__, localns)", exc)
        error("Unsupported type annotation %s. It is not a type." % annotation)


def _is_union(origin):
    if origin is typing.Union:
        return True
    try:
        import types

        return origin is types.UnionType
    except (ImportError, AttributeError):
        return False


def _schema_type(annotation, function, torch_module, error):
    annotation = _evaluate_annotation(annotation, function, torch_module, error)
    tensor_type = torch_module.Tensor
    direct = {
        tensor_type: "Tensor",
        int: "SymInt",
        float: "float",
        bool: "bool",
        str: "str",
        torch_module.dtype: "ScalarType",
        torch_module.device: "Device",
    }
    if annotation in direct:
        return direct[annotation]

    origin = _typing_origin(annotation)
    args = _typing_args(annotation)
    if _is_union(origin):
        number_members = set((int, float, bool))
        if set(args) == number_members:
            return "Scalar"
        non_none = tuple(arg for arg in args if arg is not type(None))
        if len(non_none) == 1 and len(non_none) != len(args):
            return _schema_type(non_none[0], function, torch_module, error) + "?"
        error("Unsupported type annotation %s." % (annotation,))

    sequence_origins = (
        list,
        typing.List,
        typing.Sequence,
        collections.abc.Sequence,
    )
    if origin in sequence_origins:
        if len(args) != 1:
            error("Unsupported collection type annotation %s." % (annotation,))
        element = _schema_type(args[0], function, torch_module, error)
        if element.endswith("?"):
            return element[:-1] + "?[]"
        return element + "[]"

    # torch 2.11 lets an op take an opaque value type -- a plain python object
    # the graph carries rather than inspects. Serving stacks use one to pass a
    # layer name without baking it in as a constant. Nothing here dispatches on
    # the schema, so the name of the class is description enough; refusing it
    # would only stop the op from being registered at all.
    if isinstance(annotation, type) and _is_opaque_type(annotation, torch_module):
        return "__torch__.torch.classes." + annotation.__name__

    error("Unsupported type annotation %s." % (annotation,))


def _is_opaque_type(annotation, torch_module):
    opaque_base = getattr(torch_module, "_opaque_base", None)
    base = getattr(opaque_base, "OpaqueBase", None)
    if base is None or base is object:
        return False
    try:
        return issubclass(annotation, base)
    except TypeError:
        return False


def _return_schema(annotation, function, torch_module, error):
    annotation = _evaluate_annotation(annotation, function, torch_module, error)
    if annotation is None or annotation is type(None):
        return "()"
    if annotation is _EMPTY:
        error("No return type annotation was provided. Please add one.")

    origin = _typing_origin(annotation)
    if origin not in (tuple, typing.Tuple):
        result = _schema_type(annotation, function, torch_module, error)
        if result not in ("Tensor", "Tensor[]", "SymInt", "float", "bool", "Scalar"):
            error("Return has unsupported type %s." % (annotation,))
        return result

    args = _typing_args(annotation)
    if len(args) == 2 and args[1] is Ellipsis:
        error("Return has unsupported variadic tuple type %s." % (annotation,))
    results = [_return_schema(arg, function, torch_module, error) for arg in args]
    if any(result.startswith("(") for result in results):
        error("Nested tuple returns are not supported.")
    output = ", ".join(results)
    if len(results) == 1:
        output = "(" + output + ")"
    return "(" + output + ")"


def _default_schema(value, torch_module, error):
    if value is None or isinstance(value, (int, float, bool)):
        return str(value)
    # The compatibility dtype intentionally subclasses ``str`` so Jittor's
    # native dtype dispatch accepts it.  Match the semantic type first.
    if isinstance(value, torch_module.dtype):
        return value.name
    if isinstance(value, str):
        return '"%s"' % value
    if isinstance(value, torch_module.device):
        return '"%s"' % value
    error("Unsupported default value type %s." % type(value))


def _infer_schema(function, mutates_args, op_name, torch_module):
    signature = inspect.signature(function)

    def error(what):
        raise ValueError("infer_schema(func): %s Got func with signature %s)" % (what, signature))

    if isinstance(mutates_args, str) and mutates_args != _UNKNOWN_MUTATES:
        raise ValueError("mutates_args must be a sequence of argument names or 'unknown'.")
    if not isinstance(mutates_args, str):
        try:
            mutable_names = set(mutates_args)
        except TypeError:
            raise ValueError("mutates_args must be a sequence of argument names or 'unknown'.")
    else:
        mutable_names = None

    parameters = []
    seen = set()
    keyword_only = False
    for index, (name, parameter) in enumerate(signature.parameters.items()):
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            error("We do not support positional-only args, varargs, or varkwargs.")
        if parameter.kind == inspect.Parameter.KEYWORD_ONLY and not keyword_only:
            parameters.append("*")
            keyword_only = True
        if parameter.annotation is _EMPTY:
            error("Parameter %s must have a type annotation." % name)

        schema_type = _schema_type(parameter.annotation, function, torch_module, error)
        mutated = (mutates_args == _UNKNOWN_MUTATES and schema_type.startswith("Tensor")) or (
            mutable_names is not None and name in mutable_names
        )
        if mutated:
            if not schema_type.startswith("Tensor"):
                error(
                    "Parameter %s is mutable but only Tensors or collections "
                    "of Tensors can be mutated." % name
                )
            suffix = schema_type[len("Tensor") :]
            schema_type = "Tensor(a%d!)%s" % (index, suffix)

        item = "%s %s" % (schema_type, name)
        if parameter.default is not _EMPTY:
            item += "=" + _default_schema(parameter.default, torch_module, error)
        parameters.append(item)
        seen.add(name)

    if mutable_names is not None:
        missing = mutable_names - seen
        if missing:
            error("%s in mutates_args were not found in the function signature." % sorted(missing))

    result = _return_schema(signature.return_annotation, function, torch_module, error)
    prefix = op_name if op_name is not None else ""
    return "%s(%s) -> %s" % (prefix, ", ".join(parameters), result)


def make_infer_schema(torch_module):
    """Bind ``infer_schema`` to the active Torch-compatible type objects."""

    def infer_schema(prototype_function, *, mutates_args, op_name=None):
        return _infer_schema(prototype_function, mutates_args, op_name, torch_module)

    infer_schema.__module__ = __name__
    return infer_schema


__all__ = ["Tag", "install_torch_library", "make_infer_schema"]
