"""Stable compiler/transform facades and explicit installation bindings."""

import sys as _sys

import jittor as jt

from ...permissive import install_permissive_package
from ...transaction import active_transaction
from ..fidelity import Fidelity, register_fidelity
from ..library import install_torch_library
from ..context import get_install_context
from ..context import TransformGetItemToIndex as _TransformGetItemToIndex
from .factories import _install_empty_like


_COMPILE_DEFAULT_BACKENDS = (None, "", "inductor", "eager", "aot_eager")


def compile(model=None, *args, **kwargs):
    """Expose the compiler-family callable as a stable module-level object."""
    from ...stub_policy import unimplemented
    if kwargs.get("fullgraph"):
        unimplemented(
            "torch.compile(fullgraph=True)",
            "accept an unchecked single-graph assertion",
            "Drop fullgraph=True because Jittor runs each op through its own JIT.",
        )
    backend = kwargs.get("backend")
    if backend not in _COMPILE_DEFAULT_BACKENDS:
        name = getattr(backend, "__name__", None) or repr(backend)
        unimplemented(
            "torch.compile(backend=%s)" % name,
            "silently discard a custom compiler backend",
            "Jittor has no pluggable torch.compile backend.",
        )
    return model if model is not None else (lambda value: value)


def script(obj=None, **kwargs):
    """TorchScript scripting remains an explicit eager pass-through."""
    return obj if obj is not None else (lambda value: value)


def trace(func=None, example_inputs=None, *args, **kwargs):
    """Expose eager tracing behavior as a stable module-level object."""
    from ...stub_policy import degraded, unimplemented
    if kwargs.get("check_trace"):
        unimplemented(
            "torch.jit.trace(check_trace=True)",
            "report an unchecked traced graph comparison",
            "Pass check_trace=False, or verify outputs yourself.",
        )
    if func is not None:
        degraded(
            "torch.jit.trace",
            "return an eager callable instead of a TorchScript artifact",
            "Values are identical; only the traced artifact is missing.",
        )
    return func if func is not None else (lambda value: value)


import abc as _abc
import typing


def _compiler_context():
    return get_install_context(jt)


def _bind_missing(target, name, implementation):
    if not hasattr(target, name):
        setattr(target, name, implementation)


def _identity(value):
    return value


class Graph:
    """Annotation placeholder; no FX graph representation."""


class GraphModule:
    """Annotation placeholder; no FX graph execution."""


class Proxy:
    """Annotation placeholder; no FX tracing."""


class Node:
    """Annotation placeholder; no FX node representation."""


class OptimizedModule(jt.nn.Module):
    """Native module template; no Dynamo compilation is provided."""


class OperatorExportTypes:
    ONNX = 0
    ONNX_ATEN = 1
    ONNX_ATEN_FALLBACK = 2

class CustomGraphPass(metaclass=_abc.ABCMeta):
    @_abc.abstractmethod
    def __call__(self, graph):
        raise NotImplementedError

    @_abc.abstractmethod
    def uuid(self):
        raise NotImplementedError


class OpaqueBase(object):
    pass


class TransformGetItemToIndex(_TransformGetItemToIndex):
    def __init__(self):
        super().__init__(_compiler_context().target_namespace)


def _vmap_tree_flatten(x, *args, **kwargs):
    return _compiler_context().target_namespace.utils._pytree.tree_flatten(x)


def _vmap_tree_unflatten(leaves, spec):
    return _compiler_context().target_namespace.utils._pytree.tree_unflatten(leaves, spec)


def _vmap_broadcast_to_and_flatten(in_dims, spec):
    leaves = _compiler_context().target_namespace.utils._pytree.tree_leaves(spec)
    n = len(leaves) if leaves else 1
    if isinstance(in_dims, (list, tuple)):
        flat, _ = _compiler_context().target_namespace.utils._pytree.tree_flatten(in_dims)
        return flat if len(flat) == n else None
    return [in_dims] * n


def _vmap_validate_and_get_batch_size(flat_in_dims, flat_args):
    for in_dim, arg in zip(flat_in_dims, flat_args):
        if in_dim is not None and hasattr(arg, "shape"):
            return int(arg.shape[in_dim])
    return 0


def _func_resolve(module, name):
    # navigate module.<a>.<b>.<2>... -> (owner, leaf_attr); supports int (Sequential)
    owner = module
    parts = name.split(".")
    for p in parts[:-1]:
        if p.isdigit() and hasattr(owner, "__getitem__"):
            owner = owner[int(p)]
        else:
            owner = getattr(owner, p)
    return owner, parts[-1]


def _functional_call(module, parameters_and_buffers, args=None, kwargs=None,
                     *, tie_weights=True, strict=False, **_):
    # torch.func.functional_call: run module.forward with the given params/buffers
    # swapped in (then restored), without mutating the module. Accepts a dict or a
    # sequence of dicts (merged), matching torch.
    if args is None:
        args = ()
    elif isinstance(args, jt.Var) or not isinstance(args, (tuple, list)):
        args = (args,)
    else:
        args = tuple(args)
    if kwargs is None:
        kwargs = {}
    if isinstance(parameters_and_buffers, (list, tuple)):
        merged = {}
        for d in parameters_and_buffers:
            merged.update(d)
        parameters_and_buffers = merged
    saved = []
    try:
        for name, val in parameters_and_buffers.items():
            owner, attr = _func_resolve(module, name)
            saved.append((owner, attr, getattr(owner, attr, None)))
            setattr(owner, attr, val)
        return module(*args, **kwargs)
    finally:
        for owner, attr, orig in reversed(saved):
            setattr(owner, attr, orig)


def _func_grad_core(f, argnums, has_aux, want_value):
    def wrapped(*args, **kwargs):
        single = isinstance(argnums, int)
        nums = (argnums,) if single else tuple(argnums)
        inputs = [args[i] for i in nums]
        out = f(*args, **kwargs)
        aux = None
        if has_aux:
            out, aux = out
        grads = jt.grad(out, inputs)            # list, aligned with inputs
        g0 = grads[0] if single else tuple(grads)
        if want_value:
            val = (out, aux) if has_aux else out
            return (g0, val)
        return (g0, aux) if has_aux else g0
    return wrapped


def _func_grad(f, argnums=0, has_aux=False):
    return _func_grad_core(f, argnums, has_aux, want_value=False)


def _func_grad_and_value(f, argnums=0, has_aux=False):
    return _func_grad_core(f, argnums, has_aux, want_value=True)


def _jacrev(f, argnums=0):
    # reverse-mode Jacobian: one backward pass per scalar output component.
    def wrapped(*args, **kwargs):
        x = args[argnums]
        out = f(*args, **kwargs)
        flat = out.reshape(-1)
        rows = [jt.grad(flat[i], [x])[0].reshape(-1) for i in range(int(flat.shape[0]))]
        J = jt.stack(rows, dim=0)
        return J.reshape(list(out.shape) + list(x.shape))
    return wrapped


def _stack_module_state(models):
    from collections import OrderedDict
    models = list(models)
    ps = [dict(m.named_parameters()) for m in models]
    bs = [dict(m.named_buffers()) for m in models]
    params = OrderedDict((k, jt.stack([d[k] for d in ps], dim=0)) for k in ps[0])
    buffers = OrderedDict((k, jt.stack([d[k] for d in bs], dim=0))
                          for k in (bs[0] if bs and bs[0] else {}))
    return params, buffers


class _BroadcastingList:
    def __getitem__(self, unused):
        return typing.List


def onnx_export(*args, **kwargs):
    raise NotImplementedError("ONNX export is not supported on the jittor torch shim")


def _api_rsqrt(x):
    return 1.0 / jt.sqrt(x)


def _api_gt(a, b):
    return a > b


def _api_lt(a, b):
    return a < b


def _api_ge(a, b):
    return a >= b


def _api_le(a, b):
    return a <= b


def _api_eq(a, b):
    return a == b


def _api_cid(f=None, *a, **k):
    return f if f is not None and callable(f) else _identity


def _api_compiler_is_compiling():
    return False


def _api_compiler_is_dynamo_compiling():
    return False


def _api_compiler_is_exporting():
    return False


def _api_compiler_reset(*a, **k):
    return None


def _api_compiler_cudagraph_mark_step_begin(*a, **k):
    return None


def _api_jit_script_if_tracing(f):
    return f


def _api_jit_ignore(f=None, **k):
    return f if callable(f) else _identity


def _api_jit_unused(f):
    return f


def _api_jit_export(f):
    return f


def _api_jit_is_scripting():
    return False


def _api_jit_is_tracing():
    return False


def _api_jit_interface(c):
    return c


def _api_fx_wrap(f=None, *a, **k):
    return f if f is not None and callable(f) else _identity


def _api_dynamo_disable(f=None, **k):
    return f if f is not None else _identity


def _api_dynamo_allow_in_graph(f=None, **k):
    return f if f is not None else _identity


def _api_dynamo_disallow_in_graph(f=None, **k):
    return f if f is not None else _identity


def _api_dynamo_assume_constant_result(f=None, **k):
    return f if f is not None else _identity


def _api_dynamo_is_compiling():
    return False


def _api_dynamo_is_dynamo_compiling():
    return False


def _api_dynamo_mark_static_address(*a, **k):
    return None


def _api_dynamo_mark_dynamic(*a, **k):
    return None


def _api_dynamo_graph_break(*a, **k):
    return None


def _api_dynamo_reset(*a, **k):
    return None


def _api_eval_frame_is_dynamo_supported():
    return False


def _api_functorch_vmap__maybe_remove_batch_dim(x, *a, **k):
    return x


def _api_functorch_vmap__add_batch_dim(x, *a, **k):
    return x


def _api_functorch_vmap__remove_batch_dim(x, *a, **k):
    return x


def _api_functorch_vmap__get_name(func):
    return getattr(func, '__name__', str(func))


def _api_func_ns_vmap(*a, **k):
    return _compiler_context().target_namespace.vmap(*a, **k)


def _api_func_ns_functionalize(fn, **k):
    return fn


def _api_onnx_is_in_onnx_export():
    return False


_COMPILER_PLACEHOLDERS = frozenset((Graph, GraphModule, Proxy, Node, OptimizedModule,
                                   onnx_export))


def _register_compiler_fidelity(ctx):
    namespaces = [("torch", ctx.target_namespace)]
    prefixes = ("torch.compiler", "torch.jit", "torch.fx", "torch._dynamo",
                "torch._inductor", "torch._functorch", "torch.func", "torch.onnx",
                "torch._opaque_base")
    namespaces.extend((name, module) for name, module in ctx.registry.module_map.items()
                      if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes))
    for namespace, module in namespaces:
        for name, implementation in tuple(vars(module).items()):
            if not callable(implementation) or getattr(implementation, "__module__", None) != __name__:
                continue
            # Keep the existing compile/script/trace limitations more specific.
            if implementation in (compile, script, trace):
                continue
            placeholder = implementation in _COMPILER_PLACEHOLDERS
            detail = ("Annotation or unsupported export placeholder; no compiler execution is provided."
                      if placeholder else
                      "Eager compatibility behavior; no Dynamo/FX/TorchScript compiler artifact or full transform equivalence.")
            register_fidelity(namespace + "." + name, implementation,
                              Fidelity.UNIMPLEMENTED if placeholder else Fidelity.APPROXIMATE, detail)


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    transaction = active_transaction(ctx)
    # ---- elementwise / reduction helpers that may be missing ----
    _bind_missing(g, "rsqrt", _api_rsqrt)
    _install_empty_like(g)
    # module-level comparison ops (torch.gt(a,b) etc.); .gt methods already exist.
    _bind_missing(g, "gt", _api_gt)
    _bind_missing(g, "lt", _api_lt)
    _bind_missing(g, "ge", _api_ge)
    _bind_missing(g, "le", _api_le)
    _bind_missing(g, "eq", _api_eq)
    # torch.compile: jittor already JIT-compiles every op, so the call itself is
    # a pass-through. Handles torch.compile(model), @torch.compile, and
    # torch.compile(mode=...)(model).
    #
    # What is NOT a pass-through are the arguments that assert something about
    # the compilation. `fullgraph=True` is an assertion that the callable
    # compiles into one graph with no breaks -- accepting it silently means
    # that assertion is never checked and never can be. A `backend=` other than
    # the default names a compiler that will not run: a custom backend callable
    # is simply dropped. Those are refused; `mode`/`options`/`dynamic` are
    # performance hints with no correctness claim and stay accepted.
    _COMPILE_DEFAULT_BACKENDS = (None, "", "inductor", "eager", "aot_eager")

    _bind_missing(g, "compile", compile)
    # torch.jit: jittor has no TorchScript; the script/trace decorators are pass-throughs
    # (the eager fn already runs), and is_scripting/is_tracing report False.
    import types as _types2
    import typing as _typing
    _compiler = getattr(g, "compiler", None) or _types2.ModuleType("torch.compiler")
    _cid = _api_cid
    _compiler.is_compiling = _api_compiler_is_compiling
    _compiler.is_dynamo_compiling = _api_compiler_is_dynamo_compiling
    _compiler.is_exporting = _api_compiler_is_exporting
    _compiler.disable = _cid
    _compiler.allow_in_graph = _cid
    _compiler.assume_constant_result = _cid
    _compiler.wrap_numpy = _cid
    _compiler.reset = _api_compiler_reset
    _compiler.cudagraph_mark_step_begin = _api_compiler_cudagraph_mark_step_begin
    _modules["torch.compiler"] = _compiler
    if not hasattr(g, "compiler"):
        g.compiler = _compiler
    _inductor = _types2.ModuleType("torch._inductor")
    _inductor.__path__ = []
    _custom_graph_pass = _types2.ModuleType(
        "torch._inductor.custom_graph_pass")
    _custom_graph_pass.CustomGraphPass = CustomGraphPass
    _custom_graph_pass.CustomGraphPassType = _typing.Union[
        CustomGraphPass, _typing.Callable, type(None)]
    _inductor.custom_graph_pass = _custom_graph_pass
    # torch._inductor.config: read and written by serving stacks that decide
    # graph partitioning from it (vLLM sets custom_should_partition_ops and
    # reads triton.cudagraphs). There is no inductor behind the shim, so these
    # are settings nothing acts on -- but the module has to exist and hold them,
    # or `import torch._inductor.config` fails outright.
    _inductor_config = _types2.ModuleType("torch._inductor.config")
    _inductor_config.custom_should_partition_ops = []
    _inductor_config._config = {}
    _inductor_config.triton = _types2.SimpleNamespace(cudagraphs=False)
    _inductor.config = _inductor_config
    _modules["torch._inductor"] = _inductor
    _modules["torch._inductor.custom_graph_pass"] = _custom_graph_pass
    _modules["torch._inductor.config"] = _inductor_config
    g._inductor = _inductor
    # The rest of inductor -- codecache, compile_fx, pattern_matcher and a
    # dozen more -- gets imported at module scope by serving stacks whose use
    # of it is gated on a compilation mode this backend never enters. Nothing
    # below runs, so answer those imports rather than failing them; the two
    # modules above keep their real definitions.
    # Only the modules downstream code is KNOWN to reference at import time get
    # fabricated; everything else under the prefix raises a normal ImportError.
    # Fabricating a whole namespace answered `from torch.fx.passes.shape_prop
    # import ShapeProp` too: the class constructed, the pass ran, and every
    # call returned None -- an analysis that silently did nothing.
    #
    # To extend a list, do not guess: run the workload with
    # JITTOR_TORCH_PERMISSIVE_AUDIT=1 (which fabricates everything and records
    # it) and read jittor.compat.permissive.fabricated_modules(). Refused names
    # are recorded in refused_modules() the same way.
    install_permissive_package(
        "torch._inductor", _sys.meta_path,
        transaction=transaction,
        allow=(
            # named by serving stacks at module scope, gated on a compilation
            # mode this backend never enters
            "torch._inductor.codecache",
            "torch._inductor.compile_fx",
            "torch._inductor.pattern_matcher",
        ))
    # The rest of torch's compile-and-dispatch internals, for the same reason:
    # imported at module scope by code whose use of them is gated on a compiled
    # path this backend never enters. The public counterparts that do matter --
    # torch.library, torch.compiler, torch.distributed -- are implemented, and
    # stay implemented: only these underscored namespaces are answered blind.
    # torch.fx joins them for its submodules only: the names above stay real,
    # and it is the graph-pickling and pass machinery underneath -- reached
    # only from the compiled path -- that gets answered blind.
    # torch._dispatch carries the python dispatcher the compiled path enters
    # around a trace. vLLM's compilation backend imports it at module scope
    # while the tracing it belongs to never runs here.
    # These are torch's private compile-and-dispatch plumbing: there is no
    # analysis or rewrite underneath them that could silently no-op, and
    # absence is the expected state, so their subtrees stay permissive.
    for _internal in ("torch._library", "torch._higher_order_ops",
                      "torch._guards", "torch._logging", "torch._dynamo",
                      "torch._dispatch"):
        install_permissive_package(_internal, _sys.meta_path,
                                   transaction=transaction,
                                   allow=(_internal + ".*",))
    # torch.fx is NOT one of them. Underneath it live real graph analyses and
    # rewrite passes (shape propagation, partitioning, the pass manager); a
    # fabricated one constructs, runs and returns None, which reads as "the
    # pass found nothing" rather than "the pass does not exist". Only the
    # module a definition site needs is answered.
    install_permissive_package(
        "torch.fx", _sys.meta_path,
        transaction=transaction,
        allow=("torch.fx.immutable_collections", "torch.fx.proxy",
               "torch.fx.node", "torch.fx.graph", "torch.fx.graph_module"))
    # torch 2.11 introduced opaque value types: an object an operator takes as
    # an argument and the graph carries along without inspecting it. Declaring
    # the base is the whole of what a definition site needs -- there is no
    # graph here to hoist a value into, so nothing else about it can matter.
    _opaque_base = _types2.ModuleType("torch._opaque_base")


    _opaque_base.OpaqueBase = OpaqueBase
    _modules["torch._opaque_base"] = _opaque_base
    g._opaque_base = _opaque_base
    _jit = _types2.SimpleNamespace()


    _jit.script = script
    _jit.trace = trace
    _jit.trace_module = trace
    register_fidelity(
        "torch.compile", compile, Fidelity.APPROXIMATE,
        "Jittor executes eagerly and does not produce a single compiled graph.")
    register_fidelity(
        "torch.jit.script", script, Fidelity.APPROXIMATE,
        "Jittor does not emit a TorchScript artifact.")
    register_fidelity(
        "torch.jit.trace", trace, Fidelity.APPROXIMATE,
        "Jittor returns the eager callable instead of a traced artifact.")
    _jit.script_if_tracing = _api_jit_script_if_tracing
    _jit.ignore = _api_jit_ignore
    _jit.unused = _api_jit_unused
    _jit.export = _api_jit_export
    _jit.is_scripting = _api_jit_is_scripting
    _jit.is_tracing = _api_jit_is_tracing
    _jit.ScriptModule = g.nn.Module
    _jit.interface = _api_jit_interface
    try:
        from typing import Final as _Final
    except ImportError:  # Python 3.7
        from typing_extensions import Final as _Final
    _jit.Final = _Final
    _bind_missing(g, "jit", _jit)
    _bind_missing(g, "ScriptModule", _jit.ScriptModule)
    _modules.setdefault("torch.jit", _jit)
    _fx = _types2.ModuleType("torch.fx")
    # A package, so its submodules can be imported and answered below.
    _fx.__path__ = []
    _fx.Graph = Graph
    _fx.GraphModule = GraphModule
    _fx.Proxy = Proxy
    _fx.Node = Node
    _fx.wrap = _api_fx_wrap
    _modules["torch.fx"] = _fx
    g.fx = _fx
    # torch._dynamo: minimal importable stubs for libraries that probe or
    # decorate with Dynamo APIs. Jittor runs eagerly/JIT through its own stack.
    _dynamo = _types2.ModuleType("torch._dynamo")
    # A package, so the submodules below and the permissive ones further down
    # can be imported rather than only read as attributes.
    _dynamo.__path__ = []
    _dynamo.disable = _api_dynamo_disable
    _dynamo.allow_in_graph = _api_dynamo_allow_in_graph
    _dynamo.disallow_in_graph = _api_dynamo_disallow_in_graph
    _dynamo.assume_constant_result = _api_dynamo_assume_constant_result
    _dynamo.is_compiling = _api_dynamo_is_compiling
    _dynamo.is_dynamo_compiling = _api_dynamo_is_dynamo_compiling
    _dynamo.config = _types2.SimpleNamespace()
    _dynamo.mark_static_address = _api_dynamo_mark_static_address
    _dynamo.mark_dynamic = _api_dynamo_mark_dynamic
    _dynamo.graph_break = _api_dynamo_graph_break
    _dynamo.reset = _api_dynamo_reset
    _modules["torch._dynamo"] = _dynamo
    setattr(g, "_dynamo", _dynamo)
    _eval_frame = _types2.ModuleType("torch._dynamo.eval_frame")
    _eval_frame.OptimizedModule = ctx.state.get("nn_class_adapter", _identity)(OptimizedModule)
    _eval_frame.is_dynamo_supported = _api_eval_frame_is_dynamo_supported
    _dynamo.OptimizedModule = _eval_frame.OptimizedModule
    _dynamo.eval_frame = _eval_frame
    _modules["torch._dynamo.eval_frame"] = _eval_frame
    _twh = _types2.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")

    _twh.TransformGetItemToIndex = TransformGetItemToIndex
    _modules["torch._dynamo._trace_wrapped_higher_order_op"] = _twh
    _functorch_pkg = _types2.ModuleType("torch._functorch")
    _functorch_vmap = _types2.ModuleType("torch._functorch.vmap")
    _functorch_vmap._maybe_remove_batch_dim = _api_functorch_vmap__maybe_remove_batch_dim
    _functorch_vmap._add_batch_dim = _api_functorch_vmap__add_batch_dim
    _functorch_vmap._remove_batch_dim = _api_functorch_vmap__remove_batch_dim
    _functorch_vmap._broadcast_to_and_flatten = _vmap_broadcast_to_and_flatten
    _functorch_vmap._get_name = _api_functorch_vmap__get_name
    _functorch_vmap._validate_and_get_batch_size = _vmap_validate_and_get_batch_size
    _functorch_vmap.Tensor = getattr(g, "Tensor", jt.Var)
    _functorch_vmap.tree_flatten = _vmap_tree_flatten
    _functorch_vmap.tree_unflatten = _vmap_tree_unflatten
    _functorch_pkg.vmap = _functorch_vmap
    _modules["torch._functorch"] = _functorch_pkg
    _modules["torch._functorch.vmap"] = _functorch_vmap
    setattr(g, "_functorch", _functorch_pkg)
    install_torch_library(g, _modules)

    # torch.func (functorch): functional transforms used by LoRA / meta-learning /
    # model ensembling (functorch). Jittor's autograd is graph-based, so these are
    # thin wrappers over jt.grad + temporary parameter rebinding.


    _func_ns = _types2.ModuleType("torch.func")
    _func_ns.functional_call = _functional_call
    _func_ns.grad = _func_grad
    _func_ns.grad_and_value = _func_grad_and_value
    _func_ns.vmap = _api_func_ns_vmap   # _vmap is defined later in this fn
    _func_ns.jacrev = _jacrev
    _func_ns.jacfwd = _jacrev          # same numerics; forward-mode falls back to reverse
    _func_ns.stack_module_state = _stack_module_state
    _func_ns.functionalize = _api_func_ns_functionalize
    _modules["torch.func"] = _func_ns
    g.func = _func_ns
    # torch.nn.utils also exposes stateless.functional_call (older API path).
    if not hasattr(g, "functional_call"):
        g.functional_call = _functional_call
    _register_compiler_fidelity(ctx)


def install_parity(ctx):
    import typing
    g = ctx.jittor_module
    registry = ctx.registry
    module = registry.ensure
    annotations = module("torch.jit.annotations")
    for name in ("Any", "List", "Dict", "Tuple", "Optional", "Union", "Callable"):
        setattr(annotations, name, getattr(typing, name, object))


    for dimensions in (1, 2, 3):
        name = "BroadcastingList%d" % dimensions
        if not hasattr(annotations, name):
            setattr(annotations, name, _BroadcastingList())
    annotations.Future = typing.Any
    g.jit.annotations = annotations

    onnx = module("torch.onnx")
    onnx.is_in_onnx_export = _api_onnx_is_in_onnx_export


    onnx.export = onnx_export
    onnx.OperatorExportTypes = getattr(
        onnx,
        "OperatorExportTypes",
        OperatorExportTypes,
    )
    g.onnx = onnx
    _register_compiler_fidelity(ctx)
