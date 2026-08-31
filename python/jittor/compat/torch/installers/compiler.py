"""Family-owned Torch compatibility installer.

This module contains source moved from the former monolithic installer without
changing the compatibility semantics.
"""

import jittor as jt

from ..library import install_torch_library


def install(ctx):
    _modules = ctx.registry.module_map
    g = ctx.jittor_module
    Var = ctx.state["Var"]
    _DTYPE_OBJS = ctx.state["dtypes"]
    # ---- elementwise / reduction helpers that may be missing ----
    def _alias(name, fn):
        if not hasattr(g, name):
            setattr(g, name, fn)
    _alias("rsqrt", lambda x: 1.0 / jt.sqrt(x))
    _alias("empty_like", lambda x, **k: jt.empty(x.shape, x.dtype))
    # module-level comparison ops (torch.gt(a,b) etc.); .gt methods already exist.
    _alias("gt", lambda a, b: a > b)
    _alias("lt", lambda a, b: a < b)
    _alias("ge", lambda a, b: a >= b)
    _alias("le", lambda a, b: a <= b)
    _alias("eq", lambda a, b: a == b)
    # torch.compile: jittor already JIT-compiles every op, so this is a pass-through.
    # Handles torch.compile(model), @torch.compile, and torch.compile(mode=...)(model).
    def _compile(model=None, *a, **k):
        return model if model is not None else (lambda m: m)
    _alias("compile", _compile)
    # torch.jit: jittor has no TorchScript; the script/trace decorators are pass-throughs
    # (the eager fn already runs), and is_scripting/is_tracing report False.
    import abc as _abc
    import types as _types2
    import typing as _typing
    _compiler = getattr(g, "compiler", None) or _types2.ModuleType("torch.compiler")
    _cid = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda h: h))
    _compiler.is_compiling = lambda: False
    _compiler.is_dynamo_compiling = lambda: False
    _compiler.is_exporting = lambda: False
    _compiler.disable = _cid
    _compiler.allow_in_graph = _cid
    _compiler.assume_constant_result = _cid
    _compiler.wrap_numpy = _cid
    _compiler.reset = lambda *a, **k: None
    _compiler.cudagraph_mark_step_begin = lambda *a, **k: None
    _modules["torch.compiler"] = _compiler
    if not hasattr(g, "compiler"):
        g.compiler = _compiler
    _inductor = _types2.ModuleType("torch._inductor")
    _inductor.__path__ = []
    _custom_graph_pass = _types2.ModuleType(
        "torch._inductor.custom_graph_pass")
    class CustomGraphPass(metaclass=_abc.ABCMeta):
        @_abc.abstractmethod
        def __call__(self, graph):
            raise NotImplementedError

        @_abc.abstractmethod
        def uuid(self):
            raise NotImplementedError
    CustomGraphPass.__module__ = _custom_graph_pass.__name__
    _custom_graph_pass.CustomGraphPass = CustomGraphPass
    _custom_graph_pass.CustomGraphPassType = _typing.Union[
        CustomGraphPass, _typing.Callable, type(None)]
    _inductor.custom_graph_pass = _custom_graph_pass
    _modules["torch._inductor"] = _inductor
    _modules["torch._inductor.custom_graph_pass"] = _custom_graph_pass
    g._inductor = _inductor
    _jit = _types2.SimpleNamespace()
    _jit.script = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _jit.trace = lambda f=None, *a, **k: (f if f is not None else (lambda g: g))
    _jit.script_if_tracing = lambda f: f
    _jit.ignore = lambda f=None, **k: (f if callable(f) else (lambda g: g))
    _jit.unused = lambda f: f
    _jit.export = lambda f: f
    _jit.is_scripting = lambda: False
    _jit.is_tracing = lambda: False
    _jit.ScriptModule = jt.nn.Module
    _jit.interface = lambda c: c
    try:
        from typing import Final as _Final
    except ImportError:  # Python 3.7
        from typing_extensions import Final as _Final
    _jit.Final = _Final
    _alias("jit", _jit)
    _alias("ScriptModule", _jit.ScriptModule)
    _modules.setdefault("torch.jit", _jit)
    _fx = _types2.ModuleType("torch.fx")
    _fx.Graph = type("Graph", (), {})
    _fx.GraphModule = type("GraphModule", (), {})
    _fx.Proxy = type("Proxy", (), {})
    _fx.Node = type("Node", (), {})
    _fx.wrap = lambda f=None, *a, **k: (f if f is not None and callable(f) else (lambda h: h))
    _modules["torch.fx"] = _fx
    g.fx = _fx
    # torch._dynamo: minimal importable stubs for libraries that probe or
    # decorate with Dynamo APIs. Jittor runs eagerly/JIT through its own stack.
    _dynamo = _types2.ModuleType("torch._dynamo")
    _dynamo.disable = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.allow_in_graph = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.disallow_in_graph = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.assume_constant_result = lambda f=None, **k: (f if f is not None else (lambda g: g))
    _dynamo.is_compiling = lambda: False
    _dynamo.is_dynamo_compiling = lambda: False
    _dynamo.config = _types2.SimpleNamespace()
    _dynamo.mark_static_address = lambda *a, **k: None
    _dynamo.mark_dynamic = lambda *a, **k: None
    _dynamo.graph_break = lambda *a, **k: None
    _dynamo.reset = lambda *a, **k: None
    _modules["torch._dynamo"] = _dynamo
    setattr(g, "_dynamo", _dynamo)
    _eval_frame = _types2.ModuleType("torch._dynamo.eval_frame")
    _eval_frame.OptimizedModule = type("OptimizedModule", (jt.nn.Module,), {})
    _eval_frame.is_dynamo_supported = lambda: False
    _dynamo.OptimizedModule = _eval_frame.OptimizedModule
    _dynamo.eval_frame = _eval_frame
    _modules["torch._dynamo.eval_frame"] = _eval_frame
    _twh = _types2.ModuleType("torch._dynamo._trace_wrapped_higher_order_op")
    class TransformGetItemToIndex:
        def __enter__(self):
            self._previous_depth = getattr(
                g, "_transform_getitem_to_index_depth", 0)
            g._transform_getitem_to_index_depth = self._previous_depth + 1
            return self
        def __exit__(self, *exc):
            g._transform_getitem_to_index_depth = self._previous_depth
            return False
    _twh.TransformGetItemToIndex = TransformGetItemToIndex
    _modules["torch._dynamo._trace_wrapped_higher_order_op"] = _twh
    _functorch_pkg = _types2.ModuleType("torch._functorch")
    _functorch_vmap = _types2.ModuleType("torch._functorch.vmap")
    _functorch_vmap._maybe_remove_batch_dim = lambda x, *a, **k: x
    _functorch_vmap._add_batch_dim = lambda x, *a, **k: x
    _functorch_vmap._remove_batch_dim = lambda x, *a, **k: x
    def _vmap_tree_flatten(x, *args, **kwargs):
        return g.utils._pytree.tree_flatten(x)
    def _vmap_tree_unflatten(leaves, spec):
        return g.utils._pytree.tree_unflatten(leaves, spec)
    def _vmap_broadcast_to_and_flatten(in_dims, spec):
        leaves = g.utils._pytree.tree_leaves(spec)
        n = len(leaves) if leaves else 1
        if isinstance(in_dims, (list, tuple)):
            flat, _ = g.utils._pytree.tree_flatten(in_dims)
            return flat if len(flat) == n else None
        return [in_dims] * n
    def _vmap_validate_and_get_batch_size(flat_in_dims, flat_args):
        for in_dim, arg in zip(flat_in_dims, flat_args):
            if in_dim is not None and hasattr(arg, "shape"):
                return int(arg.shape[in_dim])
        return 0
    _functorch_vmap._broadcast_to_and_flatten = _vmap_broadcast_to_and_flatten
    _functorch_vmap._get_name = lambda func: getattr(func, "__name__", str(func))
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

    _func_ns = _types2.SimpleNamespace()
    _func_ns.functional_call = _functional_call
    _func_ns.grad = _func_grad
    _func_ns.grad_and_value = _func_grad_and_value
    _func_ns.vmap = lambda *a, **k: g.vmap(*a, **k)   # _vmap is defined later in this fn
    _func_ns.jacrev = _jacrev
    _func_ns.jacfwd = _jacrev          # same numerics; forward-mode falls back to reverse
    _func_ns.stack_module_state = _stack_module_state
    _func_ns.functionalize = lambda fn, **k: fn
    _alias("func", _func_ns)
    # torch.nn.utils also exposes stateless.functional_call (older API path).
    if not hasattr(g, "functional_call"):
        g.functional_call = _functional_call


def install_parity(ctx):
    import typing
    g = ctx.jittor_module
    registry = ctx.registry
    def module(name):
        return registry.ensure(name)
    annotations = module("torch.jit.annotations")
    for name in ("Any", "List", "Dict", "Tuple", "Optional", "Union", "Callable"):
        setattr(annotations, name, getattr(typing, name, object))

    class _BroadcastingList:
        def __getitem__(self, unused):
            return typing.List

    for dimensions in (1, 2, 3):
        name = "BroadcastingList%d" % dimensions
        if not hasattr(annotations, name):
            setattr(annotations, name, _BroadcastingList())
    annotations.Future = typing.Any
    g.jit.annotations = annotations

    onnx = module("torch.onnx")
    onnx.is_in_onnx_export = lambda: False

    def onnx_export(*args, **kwargs):
        raise NotImplementedError("ONNX export is not supported on the jittor torch shim")

    onnx.export = onnx_export
    onnx.OperatorExportTypes = getattr(
        onnx,
        "OperatorExportTypes",
        type(
            "OperatorExportTypes",
            (),
            {"ONNX": 0, "ONNX_ATEN": 1, "ONNX_ATEN_FALLBACK": 2},
        ),
    )
    g.onnx = onnx
