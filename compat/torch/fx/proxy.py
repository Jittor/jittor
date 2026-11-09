"""Proxy values and the symbolic tracer.

``symbolic_trace`` feeds proxy arguments through a module's forward; operations
on them record graph nodes. Module calls and parameter reads are intercepted
by patching ``_dispatch_call`` and ``Module.__getattribute__`` for the duration
of the trace and restoring them afterwards.
"""

from __future__ import absolute_import

import functools
import inspect
import math
import operator
from collections import OrderedDict

import jittor as jt

from ...diagnostics import EXPECTED, swallowed
from .graph import Graph
from .node import Node, TraceError


# ---------------------------------------------------------------------------
# Proxy and symbolic tracing
# ---------------------------------------------------------------------------


class Proxy(object):
    """A symbolic stand-in for a value while a function is being traced.

    Every operation performed on a proxy records a node instead of computing.
    Attribute access is lazy: ``x.shape`` only becomes a ``getattr`` node when
    the result is used, and ``x.relu()`` becomes one ``call_method`` node.
    """

    def __init__(self, node, tracer=None):
        if node is None:
            raise TraceError("Proxy with no node")
        if tracer is None:
            tracer = GraphAppendingTracer(node.graph)
        self.tracer = tracer
        self.node = node

    def __repr__(self):
        return "Proxy(%s)" % self.node.name

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            # Python looks up protocol hooks on the type, not the instance, so
            # reaching here means code explicitly asked for a dunder; returning
            # an Attribute would make copy/pickle recurse forever.
            raise AttributeError(k)
        return Attribute(self, k)

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", "__call__", (self,) + args, kwargs)

    def __iter__(self):
        raise TraceError(
            "Proxy object cannot be iterated. A traced value's contents are "
            "not known at trace time; unroll the loop or use concrete_args.")

    def __bool__(self):
        raise TraceError(
            "symbolically traced variables cannot be used as inputs to "
            "control flow")

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        # Without this, copy.deepcopy would walk __getattr__ and recurse.
        new = Proxy.__new__(Proxy)
        for k, v in self.__dict__.items():
            try:
                new.__dict__[k] = __import__("copy").deepcopy(v, memo)
            except EXPECTED as exc:
                swallowed("Proxy.__deepcopy__: copy.deepcopy(value)", exc)
                new.__dict__[k] = v
        return new


class Attribute(Proxy):
    """The result of ``proxy.attr``; materialized lazily when used."""

    def __init__(self, root, attr):
        self.root = root
        self.attr = attr
        self.tracer = root.tracer
        self._node = None

    @property
    def node(self):
        if self._node is None:
            self._node = self.tracer.create_proxy(
                "call_function", getattr, (self.root, self.attr), {}).node
        return self._node

    def __call__(self, *args, **kwargs):
        return self.tracer.create_proxy(
            "call_method", self.attr, (self.root,) + args, kwargs)


def _proxy_magic(method):
    def impl(*args, **kwargs):
        tracer = args[0].tracer
        target = getattr(operator, method)
        return tracer.create_proxy("call_function", target, args, kwargs)
    impl.__name__ = method
    return impl


# ``method`` is the operator-module name; the dunder drops only the trailing
# underscore of names like ``and_``.  Methods are installed after class
# creation so defining ``__eq__`` does not clear ``Proxy.__hash__``.
for _method in (
        "eq", "ne", "lt", "gt", "le", "ge", "pos", "neg", "invert",
        "add", "sub", "mul", "floordiv", "truediv", "mod", "pow",
        "lshift", "rshift", "and_", "or_", "xor", "getitem", "matmul"):
    setattr(Proxy, "__%s__" % _method.rstrip("_"), _proxy_magic(_method))


def _reflected_magic(method):
    def impl(*args, **kwargs):
        tracer = args[0].tracer
        target = getattr(operator, method)
        # Python hands us (self, other) for a reflected call; the operator
        # target expects (other, self).
        return tracer.create_proxy(
            "call_function", target, (args[1], args[0]) + args[2:], kwargs)
    impl.__name__ = "r" + method
    return impl


for _method in (
        "add", "sub", "mul", "floordiv", "truediv", "mod", "pow",
        "lshift", "rshift", "and_", "or_", "xor", "matmul"):
    setattr(Proxy, "__r%s__" % _method.rstrip("_"),
            _reflected_magic(_method))


def _find_proxy(value):
    """Return the first ``Proxy`` reachable from ``value``, or ``None``."""
    if isinstance(value, Proxy):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _find_proxy(item)
            if found is not None:
                return found
    elif isinstance(value, dict):
        for item in value.values():
            found = _find_proxy(item)
            if found is not None:
                return found
    return None


def wrap(f):
    """Mark a function so calls with proxies record a ``call_function`` node.

    Torch additionally rewrites references in the traced function's globals;
    here the returned wrapper is what must be called (or assigned back to the
    name in the module).  Calls without proxies run the original function.
    """
    if isinstance(f, (list, tuple)):
        return [wrap(item) for item in f]
    if isinstance(f, str):
        module_name, _, attr = f.rpartition(".")
        module = __import__(module_name, fromlist=[attr])
        return wrap(getattr(module, attr))

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        proxy = _find_proxy((args, kwargs))
        if proxy is not None:
            return proxy.tracer.create_proxy(
                "call_function", f, args, kwargs)
        return f(*args, **kwargs)

    wrapped.__wrapped__ = f
    return wrapped


class GraphAppendingTracer(object):
    """Tracer bound to an existing graph, for wrapping a raw :class:`Node`."""

    def __init__(self, graph):
        self.graph = graph

    def create_proxy(self, kind, target, args, kwargs, name=None,
                     type_expr=None):
        node = self.graph.create_node(kind, target, args, kwargs, name,
                                      type_expr)
        return Proxy(node, self)


class Tracer(object):
    """Symbolically execute a module or function and return its graph.

    Module calls reach ``_dispatch_call`` (installed by the torch shim) and
    parameter reads reach ``Module.__getattribute__``; both are patched for the
    duration of the trace and restored afterwards.  Leaf modules -- modules
    whose class lives in ``torch.nn``/``jittor.nn``, excluding containers --
    become ``call_module`` nodes; everything else is traced through.

    ``autowrap_modules``/``autowrap_functions`` are accepted for signature
    compatibility but not applied; ``torch.fx.wrap`` is the supported way to
    opt a function into tracing.
    """

    def __init__(self, autowrap_modules=(math,), autowrap_functions=(),
                 param_shapes_constant=False):
        self.root = None
        self.graph = None
        self.tensor_attrs = OrderedDict()
        self.submodule_paths = None
        self.param_shapes_constant = param_shapes_constant
        self.traced_func_name = "forward"
        self._tracing = False
        self._parameter_names = {}
        self._parameter_proxies = {}
        self._orig_getattribute = None
        self._patched_dispatch = []

    # -- IR helpers -------------------------------------------------------
    def create_arg(self, a):
        """Lower a Python value seen during tracing into a graph argument."""
        if isinstance(a, Proxy):
            return a.node
        if isinstance(a, Node):
            return a
        if isinstance(a, tuple):
            if hasattr(a, "_fields"):
                return type(a)(*(self.create_arg(x) for x in a))
            return tuple(self.create_arg(x) for x in a)
        if isinstance(a, list):
            return [self.create_arg(x) for x in a]
        if isinstance(a, dict):
            return {k: self.create_arg(v) for k, v in a.items()}
        if isinstance(a, slice):
            return slice(self.create_arg(a.start), self.create_arg(a.stop),
                         self.create_arg(a.step))
        if isinstance(a, range):
            return range(self.create_arg(a.start), self.create_arg(a.stop),
                         self.create_arg(a.step))
        if isinstance(a, jt.Var):
            # A tensor constant.  Prefer an attribute that already exists on
            # the module hierarchy; otherwise stash it on the root so the
            # generated GraphModule can retrieve it with get_attr.
            for name, value in self.tensor_attrs.items():
                if value is a:
                    return self.get_attr_arg(name)
            name = "_tensor_constant%d" % len(self.tensor_attrs)
            while hasattr(self.root, name):
                name += "_"
            setattr(self.root, name, a)
            self.tensor_attrs[name] = a
            return self.get_attr_arg(name)
        return a

    def get_attr_proxy(self, name):
        proxy = self._parameter_proxies.get(name)
        if proxy is None:
            proxy = self.create_proxy("get_attr", name, (), {})
            self._parameter_proxies[name] = proxy
        return proxy

    def get_attr_arg(self, name):
        """A get_attr *Node* for use as an argument (``create_arg`` contract)."""
        return self.get_attr_proxy(name).node

    def create_proxy(self, kind, target, args, kwargs, name=None,
                     type_expr=None):
        node = self.create_node(kind, target, args, kwargs, name, type_expr)
        return Proxy(node, self)

    def create_node(self, kind, target, args, kwargs, name=None,
                    type_expr=None):
        args = tuple(self.create_arg(a) for a in args)
        kwargs = {k: self.create_arg(v) for k, v in kwargs.items()}
        return self.graph.create_node(kind, target, args, kwargs, name,
                                      type_expr)

    # -- module hierarchy -------------------------------------------------
    def path_of_module(self, mod):
        if self.submodule_paths is not None:
            path = self.submodule_paths.get(mod)
            if path is None:
                raise NameError("module is not installed as a submodule")
            return path
        for name, module in self.root.named_modules():
            if module is mod:
                return name
        raise NameError("module is not installed as a submodule")

    def is_leaf_module(self, m, module_qualified_name):
        module = type(m).__module__ or ""
        is_framework = module.startswith(("torch.nn", "jittor.nn"))
        return is_framework and type(m).__name__ not in (
            "Sequential", "ModuleList", "ModuleDict", "ParameterList",
            "ParameterDict")

    def call_module(self, m, forward, args, kwargs):
        name = self.path_of_module(m)
        if not self.is_leaf_module(m, name):
            return forward(*args, **kwargs)
        return self.create_proxy("call_module", name, args, kwargs)

    # -- tracing ----------------------------------------------------------
    def _trace_entry(self, root):
        if isinstance(root, jt.nn.Module):
            self.root = root
            self.submodule_paths = {module: name
                                    for name, module in root.named_modules()}
            cls = type(root)
            if "forward" in cls.__dict__:
                return cls.forward, True
            if "execute" in cls.__dict__:
                return cls.execute, True
            forward = getattr(cls, "forward", None)
            if forward is not None:
                return forward, True
            raise RuntimeError(
                "%s has neither forward nor execute" % cls.__name__)
        self.root = jt.nn.Module()
        return root, False

    def _install_hooks(self):
        # The shim has two module bases: the native ``jittor._core.module.
        # Module`` that jittor layers inherit, and the adapted ``torch.nn.
        # Module`` the torch frontend publishes, which defines its own
        # ``_dispatch_call``.  A module instance finds the first definition in
        # its MRO, so every class in the traced hierarchy that owns the hook
        # has to be patched.
        tracer = self
        native_module = jt.nn.Module
        self._orig_getattribute = native_module.__getattribute__
        orig_getattribute = self._orig_getattribute
        parameter_names = self._parameter_names
        self._patched_dispatch = []
        dispatch_classes = [native_module]
        for _, module in self.root.named_modules():
            for cls in type(module).__mro__:
                if cls is object:
                    break
                if "_dispatch_call" in cls.__dict__ \
                        and cls not in dispatch_classes:
                    dispatch_classes.append(cls)

        def trace_getattribute(module, name):
            value = orig_getattribute(module, name)
            if tracer._tracing and isinstance(value, jt.Var):
                qualname = parameter_names.get(id(value))
                if qualname is not None:
                    proxy = tracer._parameter_proxies.get(qualname)
                    if proxy is None:
                        proxy = tracer.create_proxy(
                            "get_attr", qualname, (), {})
                        tracer._parameter_proxies[qualname] = proxy
                    return proxy
            return value

        for cls in dispatch_classes:
            original = cls.__dict__.get("_dispatch_call")
            if original is None:
                continue

            def make_dispatch(original):
                def dispatch(module, *args, **kwargs):
                    def forward(*a, **k):
                        return original(module, *a, **k)
                    return tracer.call_module(module, forward, args, kwargs)
                return dispatch

            self._patched_dispatch.append((cls, original,
                                           make_dispatch(original)))
            cls._dispatch_call = self._patched_dispatch[-1][2]
        native_module.__getattribute__ = trace_getattribute

    def _remove_hooks(self):
        for cls, original, _ in self._patched_dispatch:
            cls._dispatch_call = original
        self._patched_dispatch = []
        jt.nn.Module.__getattribute__ = self._orig_getattribute

    def trace(self, root, concrete_args=None):
        """Trace ``root`` and return the resulting :class:`Graph`."""
        fn, is_module = self._trace_entry(root)
        try:
            fn = inspect.unwrap(fn)
        except (TypeError, ValueError) as exc:
            swallowed("Tracer.trace: inspect.unwrap", exc)
        self.graph = Graph(tracer_cls=type(self))

        if is_module:
            self._parameter_names = {
                id(param): name
                for name, param in self.root.named_parameters()}

            # torch's collect_tensor_attrs: every tensor attribute on the
            # hierarchy, so constants are emitted as get_attr rather than
            # baked into args.
            def collect(module, prefix):
                for key, value in module.__dict__.items():
                    if isinstance(value, jt.Var):
                        self.tensor_attrs[".".join(prefix + [key])] = value
                for child_name, child in module.named_children():
                    collect(child, prefix + [child_name])

            collect(self.root, [])

        try:
            signature = inspect.signature(fn)
            parameters = list(signature.parameters.values())
        except (TypeError, ValueError):
            raise TraceError("cannot inspect the signature of %r" % (fn,))
        if is_module:
            if not parameters:
                raise RuntimeError("module forward has no self parameter")
            parameters = parameters[1:]
        if concrete_args is None:
            concrete_args = {}
        proxy_args = []
        for parameter in parameters:
            if parameter.kind in (parameter.VAR_POSITIONAL,
                                  parameter.VAR_KEYWORD):
                raise TraceError(
                    "variadic signatures (*args/**kwargs) are not supported "
                    "by the Jittor FX tracer")
            if parameter.kind == parameter.KEYWORD_ONLY:
                raise TraceError(
                    "keyword-only parameters are not supported by the Jittor "
                    "FX tracer")
            if parameter.name in concrete_args:
                proxy_args.append(concrete_args[parameter.name])
            else:
                proxy_args.append(self.create_proxy(
                    "placeholder", parameter.name, (), {}))

        call_args = ([self.root] if is_module else []) + proxy_args
        try:
            self._install_hooks()
        except BaseException:
            # Never leave a half-installed class patch behind.
            self._remove_hooks()
            raise
        self._tracing = True
        try:
            output = fn(*call_args)
        finally:
            self._tracing = False
            self._remove_hooks()
        self.create_node("output", "output", (output,), {})
        return self.graph
