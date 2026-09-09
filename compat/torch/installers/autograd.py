"""Publish stable autograd owners; mutable delegates belong to the context."""
from .. import autograd as api
from ..context import get_install_context
from ..fidelity import Fidelity, register_api_bindings
from ...diagnostics import EXPECTED, swallowed


def _install_autograd(g, registry=None):
    ctx = get_install_context(g)
    registry = registry or ctx.registry
    autograd = registry.ensure("torch.autograd")
    autograd.Function = g.Function
    autograd.no_grad = g.no_grad
    autograd.enable_grad = g.enable_grad
    autograd.Variable = g.Tensor
    for name in ("grad", "backward", "set_detect_anomaly", "detect_anomaly"):
        setattr(autograd, name, getattr(api, name))
    g.autograd = autograd
    autograd.__path__ = getattr(autograd, "__path__", [])
    from jittor.autograd import functional as native_functional
    functional = registry.ensure("torch.autograd.functional")
    functional.__dict__.update({name: value for name, value in vars(native_functional).items()
                                if not name.startswith("__")})
    registry.module_map["torch.autograd.functional"] = functional
    autograd.functional = functional
    profiler = registry.ensure("torch.autograd.profiler")
    for name in ("EventList", "profile", "record_function", "emit_nvtx", "kineto_available"):
        setattr(profiler, name, getattr(api, name))
    autograd.profiler = profiler
    register_api_bindings(autograd, "torch.autograd", ("grad", "backward", "Function"),
                          Fidelity.APPROXIMATE,
                          "Native Var/Op graph; backward uses existing leaf accumulation and "
                          "grad uses native optional gradients. Batched gradients and advanced "
                          "engine options retain the existing compatibility limitations.")
    register_api_bindings(autograd, "torch.autograd", ("set_detect_anomaly", "detect_anomaly"),
                          Fidelity.UNIMPLEMENTED, "Annotation-only scope; no anomaly detection.")
    register_api_bindings(profiler, "torch.autograd.profiler",
                          ("EventList", "profile", "record_function", "emit_nvtx", "kineto_available"),
                          Fidelity.UNIMPLEMENTED, "Placeholder profiling scopes; no events or traces collected.")


def install(ctx):
    g = ctx.jittor_module
    ctx.state.setdefault("autograd_api", {
        "native_function_call": ctx.native_backend.Function.__call__,
    })
    g.Function = api.Function
    _install_autograd(g, ctx.registry)


def _install_tensordict_compat():
    try:
        from tensordict.base import TensorDictBase
        from tensordict._lazy import LazyStackedTensorDict
    except EXPECTED as exc:
        swallowed("torch/installers/autograd.py tensordict import", exc)
        return
    if getattr(TensorDictBase, "_jittor_index_compat", False):
        return
    state = api._state()
    state["tensordict_getitem"] = TensorDictBase.__getitem__
    state["lazy_tensordict_getitem"] = LazyStackedTensorDict.__getitem__
    TensorDictBase.__getitem__ = api._tensordict_getitem
    TensorDictBase.__getitems__ = api._tensordict_getitem
    LazyStackedTensorDict.__getitem__ = api._lazy_tensordict_getitem
    TensorDictBase._jittor_index_compat = True


def install_tensordict(ctx):
    _install_tensordict_compat()


def install_parity(ctx):
    autograd = ctx.jittor_module.autograd
    function = ctx.registry.ensure("torch.autograd.function")
    function.Function = autograd.Function
    function.FunctionCtx = api.FunctionCtx
    function.once_differentiable = api.once_differentiable
    autograd.function = function
    autograd.once_differentiable = api.once_differentiable
    graph = ctx.registry.ensure("torch.autograd.graph")
    for name in ("saved_tensors_hooks", "save_on_cpu", "Node"):
        setattr(graph, name, getattr(api, name))
    autograd.graph = graph
    variable = ctx.registry.ensure("torch.autograd.variable")
    variable.Variable = api.Variable
    autograd.variable = variable
    register_api_bindings(function, "torch.autograd.function", ("FunctionCtx",),
                          Fidelity.APPROXIMATE, "Saved tensor metadata shares native Function context semantics.")
    register_api_bindings(function, "torch.autograd.function", ("once_differentiable",),
                          Fidelity.UNIMPLEMENTED, "Annotation-only decorator; does not prohibit higher derivatives.")
    register_api_bindings(graph, "torch.autograd.graph", ("saved_tensors_hooks", "save_on_cpu", "Node"),
                          Fidelity.UNIMPLEMENTED, "Metadata-only scopes; no saved-tensor packing or offload.")
    register_api_bindings(variable, "torch.autograd.variable", ("Variable",),
                          Fidelity.UNIMPLEMENTED, "Legacy engine placeholder; queue_callback does not schedule work.")
