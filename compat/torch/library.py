"""Torch operator-library schema compatibility.

This module owns the stable public registration objects and native custom-op
bridge. Mutable dispatch state belongs to the active installation context.
"""

from __future__ import absolute_import

import enum
import re
from typing import Any, Optional, Tuple, cast
import jittor as jt
from ..diagnostics import EXPECTED, swallowed
from ..transaction import InstallTransaction, current_transaction, _MISSING
from .library_metadata import _infer_schema, _install_dispatch_key_surface


def _set_item(mapping, key, value):
    """Publish one registry slot into the enclosing installation/runtime hook."""
    transaction = current_transaction()
    if transaction is not None:
        transaction.record(mapping, key, mapping.get(key, _MISSING), value)
    mapping[key] = value


def _set_attr(target, key, value):
    transaction = current_transaction()
    if transaction is not None:
        transaction.mutate_attr(target, key, value)
    else:
        setattr(target, key, value)


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
_DEVICE_DISPATCH_KEYS = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "default": "CompositeExplicitAutograd",
}


def _dispatch_order(residency):
    order = _CUDA_DISPATCH_ORDER if residency == "cuda" else _CPU_DISPATCH_ORDER

    # Autocast is a thread-local dtype policy and its dispatch key outranks the
    # ordinary backend key in torch.  Query the policy for the tensor's actual
    # residency: a CPU autocast context must not affect a CUDA registration (or
    # vice versa), and an enabled=False nested context returns None here.
    from .amp import autocast_dtype

    if autocast_dtype(residency) is None:
        return order
    key = "AutocastCUDA" if residency == "cuda" else "AutocastCPU"
    return (key,) + order


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
        self._saved_tensors: Tuple[Any, ...] = ()
        self._saved_versions: Tuple[Optional[Tuple[int, int, int]], ...] = ()

    def save_for_backward(self, *tensors):
        import jittor as jt

        self._saved_tensors = tuple(tensors)
        self._saved_versions = cast(Tuple[Optional[Tuple[int, int, int]], ...], tuple(
            (tuple(t._saved_storage_version()) if isinstance(t, jt.Var) else None) for t in tensors)
        )

    @property
    def saved_tensors(self):
        import jittor as jt

        for tensor, version in zip(self._saved_tensors, self._saved_versions):
            if isinstance(tensor, jt.Var) and version is not None \
                    and tuple(tensor._saved_storage_version()) != version:
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

    return _LibraryAutograd(op, function, args, kwargs, slots)(
        *[args[slot] for slot in slots])


class _LibraryAutograd(jt.Function):
    """One native taped call; configuration is copied into its call context."""

    def __init__(self, op, function, args, kwargs, slots):
        self.op, self.function = op, function
        self.args, self.kwargs, self.slots = args, kwargs, slots

    def execute(self, *taped):
        full = list(self.args)
        for slot, value in zip(self.slots, taped):
            full[slot] = value
        self._ctx = _AutogradContext()
        output = self.function(*full, **self.kwargs)
        if self.op._setup_context is not None:
            self.op._setup_context(self._ctx, tuple(full), output)
        # A registered backward, rather than the kernel's internal graph,
        # owns differentiation. Under explicit-requires-grad policy a detached
        # forward result stays stopped unless it is enabled before native tape
        # construction. Integer/bool outputs remain non-differentiable.
        from jittor._core.dtypes import dtype_name
        outputs = output if isinstance(output, (tuple, list)) else (output,)
        for value in outputs:
            if isinstance(value, jt.Var) and dtype_name(value.dtype).startswith(
                    ("float", "bfloat", "complex")):
                value.start_grad()
        return output

    def grad(self, *grads):
        produced = self.op._backward(self._ctx, *grads)
        if not isinstance(produced, (tuple, list)):
            produced = (produced,)
        return tuple(produced[slot] if slot < len(produced) else None
                     for slot in self.slots)


class _RegisteredOp:
    def __init__(self, namespace, name, namespace_obj=None):
        self.namespace = namespace
        self.name = name
        # `ns.op` is the packet its overloads hang off; `ns.op.overload` is a
        # `_RegisteredOp` too, created and stored under the full name.
        self._namespace_obj = namespace_obj
        self.default = self
        self._schema = None
        self._tags = ()
        self._implementations = {}
        self._fake_impl = None
        self._backward = None
        self._setup_context = None
        self._overridden_by_integration = None

    def register_impl(self, dispatch_key, function, allow_override=False):
        with InstallTransaction._lock:
            key = str(dispatch_key or "CompositeExplicitAutograd")
            key = _DEVICE_DISPATCH_KEYS.get(key, key)
            if key in self._implementations and not allow_override:
                raise RuntimeError(
                    "operator %s::%s already has an implementation for %s"
                    % (self.namespace, self.name, key)
                )
            _set_item(self._implementations, key, function)

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
        for key in _dispatch_order(residency):
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

    def __getattr__(self, name):
        # `torch.ops.ns.op.overload`: the overload is registered as its own
        # operator under the full name, so resolve it through the namespace.
        namespace = object.__getattribute__(self, "_namespace_obj")
        if namespace is not None:
            ops = object.__getattribute__(namespace, "_ops")
            full = "%s.%s" % (object.__getattribute__(self, "name"), name)
            if full in ops:
                return ops[full]
        raise AttributeError(
            "torch.ops.%s.%s has no overload '%s'"
            % (object.__getattribute__(self, "namespace"),
               object.__getattribute__(self, "name"), name))

    def overloads(self):
        names = ["default"]
        namespace = self._namespace_obj
        if namespace is not None:
            ops = object.__getattribute__(namespace, "_ops")
            prefix = self.name + "."
            names += sorted(key[len(prefix):] for key in ops
                            if key.startswith(prefix))
        return names

    def __repr__(self):
        return "%s.%s" % (self.namespace, self.name)


class _AtenOpProxy:
    """An unregistered `torch.ops.aten.<name>` answered by `torch.<name>`.

    Libraries reference aten overloads (`aten.empty_like.default`) while
    defining wrapper subclasses, and only ever call them; the eager function
    of the same name is the implementation. Overload attribute access returns
    this proxy itself, so `.default`, `.Tensor` and `.out` all dispatch to the
    same function. Method-style ops (`aten.new_zeros.default(t, size)`) have no
    module-level counterpart and dispatch through the first tensor instead.
    """

    def __init__(self, name, function):
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_function", function)

    def __call__(self, *args, **kwargs):
        function = object.__getattribute__(self, "_function")
        if function is not None:
            return function(*args, **kwargs)
        name = object.__getattribute__(self, "_name")
        if not args or not hasattr(args[0], name):
            raise AttributeError(
                "torch.ops.aten.%s has no eager function and no method on its "
                "first argument" % name)
        return getattr(args[0], name)(*args[1:], **kwargs)

    def __getattr__(self, overload):
        return self

    def __repr__(self):
        return "torch.ops.aten.%s" % object.__getattribute__(self, "_name")


class _OpNamespace:
    def __init__(self, namespace):
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(self, "_ops", {})

    def get_or_create(self, name):
        with InstallTransaction._lock:
            ops = object.__getattribute__(self, "_ops")
            if name not in ops:
                _set_item(ops, name, _RegisteredOp(
                    object.__getattribute__(self, "_namespace"), name, self))
            return ops[name]

    def __getattr__(self, name):
        ops = object.__getattribute__(self, "_ops")
        if name in ops:
            return ops[name]
        if object.__getattribute__(self, "_namespace") == "aten":
            torch_module = _library_state()["torch_module"]
            function = getattr(torch_module, name, None)
            return _AtenOpProxy(name, function if callable(function) else None)
        raise AttributeError(
            "torch.ops.%s has no op '%s'" % (object.__getattribute__(self, "_namespace"), name)
        )


class _OpsDispatcher:
    def __init__(self, base):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "_namespaces", {})

    def get_or_create(self, namespace, name):
        with InstallTransaction._lock:
            namespaces = object.__getattribute__(self, "_namespaces")
            if namespace not in namespaces:
                _set_item(namespaces, namespace, _OpNamespace(namespace))
            return namespaces[namespace].get_or_create(name)

    def load_library(self, path, *args, **kwargs):
        """Report a missing compiled extension the way torch does: ImportError.

        `torch.ops.load_library` loads a libtorch C++ extension, an ABI this
        compat layer does not implement. It used to fall through `__getattr__`
        to an empty `_OpNamespace`, so calling it raised `TypeError: object is
        not callable`; libraries that treat a compiled extension as optional
        only recover from ImportError/OSError. torchvision's `extension.py`
        is the case that matters: its `try` block catches those two and falls
        back to the pure-Python implementation instead of failing the whole
        import.
        """
        base = object.__getattribute__(self, "_base")
        if base is not None:
            loader = getattr(base, "load_library", None)
            if loader is not None:
                return loader(path, *args, **kwargs)
        raise ImportError(
            "torch.ops.load_library(%r): jittor-compat has no libtorch C++ "
            "extension ABI, so a compiled PyTorch extension cannot be loaded. "
            "Libraries that keep a pure-Python fallback treat this as an "
            "optional extension and continue without their compiled ops."
            % (path,))

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
        with InstallTransaction._lock:
            if name not in namespaces:
                _set_item(namespaces, name, _OpNamespace(name))
            return namespaces[name]


def _operator_name(schema_or_name):
    name = str(schema_or_name).split("(", 1)[0]
    if "::" in name:
        _, name = name.split("::", 1)
    # Keep the overload suffix. In torch `ns.op` and `ns.op.overload` are
    # *distinct* operators, each with its own schema and kernels; dropping the
    # suffix collapsed them onto one operator, so a library that registers an
    # inplace overload (vLLM's `fused_add_rms_norm.maybe_inplace`, on
    # `CompositeExplicitAutograd`, exactly like the base op) hit "already has an
    # implementation". `_RegisteredOp.__getattr__` resolves `ns.op.overload`.
    return name.strip()


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


def _library_state():
    import jittor as jt
    from .context import get_install_context
    return get_install_context(jt).state["library_api"]


class Library:
    _transactional_registry = True

    def __init__(self, namespace, kind, dispatch_key=""):
        self._dispatcher = _library_state()["dispatcher"]
        self.ns = str(namespace)
        self.kind = str(kind)
        self.dispatch_key = str(dispatch_key)

    def _op(self, op_name):
        return self._dispatcher.get_or_create(self.ns, _operator_name(op_name))

    def define(self, schema, alias_analysis="", *, tags=()):
        if "(" not in str(schema):
            raise ValueError("operator schema must contain an argument list")
        op = self._op(schema)
        _set_attr(op, "_schema", "%s::%s" % (self.ns, schema))
        _set_attr(op, "_tags", tuple(tags))
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
        _set_attr(op, "_fake_impl", fn)
        return None


def define(qualname, schema, *, lib=None, tags=()):
    """Register a named operator schema with the active dispatcher."""
    if not isinstance(qualname, str) or qualname.count("::") != 1:
        raise ValueError("operator name must have the form namespace::name")
    namespace, name = qualname.split("::")
    if not namespace or not name or not re.fullmatch(r"\(.*\) -> .*", schema):
        raise ValueError("expected a qualified operator name and a nameless schema")
    target = lib if lib is not None else Library(namespace, "FRAGMENT")
    tags = (tags,) if isinstance(tags, Tag) else tuple(tags)
    with InstallTransaction._lock:
        if target._op(name)._schema is not None:
            raise RuntimeError(f"operator {target.ns}::{name} already has a schema")
        target.define(name + schema, tags=tags)


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
        namespace = op_name = None
        if isinstance(name, str) and "::" in name:
            namespace, op_name = name.split("::", 1)
            override = _integration_custom_op_overrides().get(name)
            op = _library_state()["dispatcher"].get_or_create(namespace, op_name)
            if override is not None:
                _set_attr(op, "_overridden_by_integration", name)
            op.register_impl(
                "CompositeExplicitAutograd", override or implementation,
                allow_override=True
            )
        # torch 2.6+ returns a definition object whose methods register the
        # fake/kernel/autograd parts (`@op.register_fake`). Callers still call
        # the definition as the eager implementation.
        return _CustomOpDef(implementation, namespace, op_name)

    return decorator(fn) if fn is not None else decorator


class _CustomOpDef:
    """The object `torch.library.custom_op` returns from torch 2.6 on."""

    def __init__(self, implementation, namespace=None, name=None):
        object.__setattr__(self, "_implementation", implementation)
        object.__setattr__(self, "_namespace", namespace)
        object.__setattr__(self, "_name", name)

    def __call__(self, *args, **kwargs):
        return object.__getattribute__(self, "_implementation")(*args, **kwargs)

    def __getattr__(self, attribute):
        return getattr(object.__getattribute__(self, "_implementation"), attribute)

    def _qualname(self):
        namespace = object.__getattribute__(self, "_namespace")
        name = object.__getattribute__(self, "_name")
        return "%s::%s" % (namespace, name) if namespace and name else None

    def register_fake(self, fn):
        qualname = self._qualname()
        if qualname is not None:
            register_fake(qualname, fn)
        return fn

    def register_kernel(self, *args, **kwargs):
        # register_kernel(dispatch_key) -> decorator, or (dispatch_key, fn).
        fn = args[-1] if args and callable(args[-1]) else None
        if fn is not None and len(args) >= 2:
            qualname = self._qualname()
            if qualname is not None:
                impl(qualname, args[0], fn)
            return fn

        def decorator(function):
            qualname = self._qualname()
            if qualname is not None:
                impl(qualname, args[0], function)
            return function

        return decorator

    def register_autograd(self, fn):
        qualname = self._qualname()
        if qualname is not None:
            register_autograd(qualname, fn)
        return fn


def register_fake(op, func=None, *, lib=None, **kwargs):
    def decorator(function):
        if isinstance(op, _RegisteredOp):
            _set_attr(op, "_fake_impl", function)
        elif isinstance(op, str) and "::" in op:
            namespace, name = op.split("::", 1)
            target = lib or Library(namespace, "FRAGMENT")
            target._register_fake(name, function, **kwargs)
        return function

    return decorator(func) if func is not None else decorator


def impl(qualname, dispatch_types, func=None, *, lib=None):
    if not isinstance(qualname, str):
        # torch 2.6+ call form: `torch.library.impl(lib, name, dispatch_key)`,
        # used as a decorator. The older form names the library inside the
        # qualname; torchao registers Meta kernels with the newer one.
        target_lib, qualname, dispatch_types, func = qualname, dispatch_types, func, None
        lib = target_lib

    def decorator(function):
        if "::" in qualname:
            namespace, name = qualname.split("::", 1)
            target = lib or Library(namespace, "FRAGMENT")
        elif lib is not None:
            target = lib
            name = qualname
        else:
            raise ValueError("operator name must have the form namespace::name")
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
        target = _library_state()["dispatcher"].get_or_create(namespace, _operator_name(name))
    else:
        raise TypeError("register_autograd expects an operator or namespace::name")
    _set_attr(target, "_backward", backward)
    _set_attr(target, "_setup_context", setup_context)
    return None



class HigherOrderOperator:
    """Metadata-only higher-order operator placeholder."""


def _annotation_identity(function):
    return function


def register_torch_dispatch(*args, **kwargs):
    return _annotation_identity


def register_vmap(*args, **kwargs):
    return _annotation_identity


def get_ctx():
    return None


_OPCHECK_EFFECT = (
    "return None from every operator-correctness check, so a user's "
    "custom-op test suite passes unconditionally whatever the operator does")
_OPCHECK_HINT = "Test the operator directly against a reference implementation."


def opcheck(*args, **kwargs):
    from ..stub_policy import unimplemented
    return unimplemented("torch.library.opcheck", _OPCHECK_EFFECT, _OPCHECK_HINT)


setattr(cast(Any, opcheck), "_jittor_unimplemented", "torch.library.opcheck")


def infer_schema(prototype_function, *, mutates_args, op_name=None):
    return _infer_schema(prototype_function, mutates_args, op_name,
                         _library_state()["torch_module"])


def register_kernel(op, device_types, func=None, *, lib=None):
    if isinstance(op, _RegisteredOp):
        op = f"{op.namespace}::{op.name}"
    dispatch_types = "CompositeExplicitAutograd" if device_types is None else device_types
    return impl(op, dispatch_types, func, lib=lib)


def wrap_triton(tensor):
    """torch.library.wrap_triton returns its argument in eager execution.

    The wrapper exists so the compiler can see a triton kernel's tensor
    arguments; there is no tracing here, so the value is already the answer.
    """
    return tensor


def install_torch_library(torch_module, modules):
    """Publish stable API objects and the installation-owned dispatcher."""
    from ..functional_collectives import install_dispatcher as install_collective_dispatcher
    from .context import get_install_context
    from .fidelity import Fidelity, register_api_bindings
    from ..stub_policy import record_unimplemented
    ctx = get_install_context(torch_module)
    record_unimplemented("torch.library.opcheck", _OPCHECK_EFFECT, _OPCHECK_HINT)
    state = ctx.state.get("library_api")
    if state is None:
        dispatcher = getattr(torch_module, "ops", None)
        if not isinstance(dispatcher, _OpsDispatcher):
            dispatcher = _OpsDispatcher(dispatcher)
        state = {"dispatcher": dispatcher, "torch_module": torch_module}
        ctx.state["library_api"] = state
    library_module = ctx.registry.ensure("torch.library")
    for name in ("Library", "Tag", "define", "custom_op", "infer_schema", "register_fake",
                 "register_kernel", "impl", "register_autograd",
                 "register_torch_dispatch", "register_vmap", "opcheck", "get_ctx",
                 "wrap_triton"):
        setattr(library_module, name, globals()[name])
    ops_module = ctx.registry.ensure("torch._ops")
    ops_module.OpOverload = _RegisteredOp
    ops_module.OpOverloadPacket = _RegisteredOp
    ops_module.HigherOrderOperator = HigherOrderOperator
    ops_module.__all__ = ["OpOverload", "OpOverloadPacket", "HigherOrderOperator"]
    modules["torch.library"] = library_module
    modules["torch._ops"] = ops_module
    torch_module.library = library_module
    torch_module._ops = ops_module
    torch_module.ops = state["dispatcher"]
    install_collective_dispatcher(state["dispatcher"])
    torch_module.Tag = Tag
    torch_c = modules.get("torch._C")
    if torch_c is None:
        torch_c = ctx.registry.ensure("torch._C")
        torch_module._C = torch_c
    torch_c.Tag = Tag
    _install_dispatch_key_surface(torch_c)
    # torch._custom_ops: torchvision's meta registrations call `get_ctx()` from
    # this module. The names map to the library API above.
    custom_ops_module = ctx.registry.ensure("torch._custom_ops")
    for _name in ("get_ctx", "infer_schema", "custom_op", "register_fake", "impl"):
        setattr(custom_ops_module, _name, globals()[_name])
    torch_module._custom_ops = custom_ops_module
    modules["torch._custom_ops"] = custom_ops_module
    register_api_bindings(library_module, "torch.library",
                          ("Library", "Tag", "define", "custom_op", "infer_schema", "register_fake",
                           "register_kernel", "impl", "register_autograd"),
                          Fidelity.APPROXIMATE,
                          "Context-owned dispatcher uses actual tensor CPU/CUDA residency and "
                          "native Function/Var gradient graph; fake kernels are metadata only. "
                          "Schema inference supports the documented Python annotation subset.")
    register_api_bindings(library_module, "torch.library",
                          ("register_torch_dispatch", "register_vmap", "opcheck", "get_ctx"),
                          Fidelity.UNIMPLEMENTED,
                          "Dispatch/vmap annotations do not install transforms; no fake context. "
                          "opcheck is unsupported under the configured stub policy.")
    register_api_bindings(ops_module, "torch._ops", ("OpOverload", "OpOverloadPacket"),
                          Fidelity.APPROXIMATE, "One default overload backed by the native execution graph.")
    register_api_bindings(ops_module, "torch._ops", ("HigherOrderOperator",),
                          Fidelity.UNIMPLEMENTED, "Metadata placeholder; no higher-order dispatch.")
    return library_module


def make_infer_schema(torch_module):
    """Legacy factory spelling; return the one active schema implementation."""
    from .context import get_install_context
    get_install_context(torch_module)
    return infer_schema


__all__ = ["Tag", "Library", "custom_op", "infer_schema", "register_fake",
           "register_kernel", "impl", "register_autograd", "register_torch_dispatch",
           "register_vmap", "opcheck", "get_ctx", "install_torch_library", "make_infer_schema"]
