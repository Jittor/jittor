"""Stable autograd/library owners share native graphs and scoped registries."""
import ast
import pickle
from pathlib import Path

import numpy as np
import pytest


def test_public_objects_identity_fidelity_and_pickle():
    import torch
    from jittor.compat.torch import autograd, library
    from jittor.compat.torch.fidelity import fidelity_of

    families = (
        (torch.autograd, autograd, "torch.autograd", ("Function", "grad", "backward",
                                                     "set_detect_anomaly", "detect_anomaly")),
        (torch.library, library, "torch.library", ("Library", "custom_op", "infer_schema",
         "register_fake", "impl", "register_kernel", "register_autograd", "opcheck",
         "get_ctx", "register_vmap", "register_torch_dispatch")),
        (torch.autograd.profiler, autograd, "torch.autograd.profiler",
         ("EventList", "profile", "record_function", "emit_nvtx", "kineto_available")),
        (torch.autograd.graph, autograd, "torch.autograd.graph",
         ("saved_tensors_hooks", "save_on_cpu", "Node")),
    )
    for published, owner, prefix, names in families:
        for name in names:
            obj = getattr(published, name)
            assert obj is getattr(owner, name)
            assert obj.__module__ == owner.__name__
            assert "<locals>" not in obj.__qualname__
            assert pickle.loads(pickle.dumps(obj)) is obj
            assert fidelity_of(prefix + "." + name).implementation is obj
    assert torch.Function is autograd.Function
    assert torch.autograd.function.Function is autograd.Function
    assert fidelity_of("torch.autograd.detect_anomaly").level.value == "unimplemented"


def test_native_graph_gradient_and_reused_function_contexts():
    import torch
    import jittor as jt

    class Square(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x * x

        @staticmethod
        def backward(ctx, dy):
            x, = ctx.saved_tensors
            return 2 * x * dy

    function = Square()
    x = torch.tensor([2., 3.], requires_grad=True)
    y = torch.tensor([4., 5.], requires_grad=True)
    first, second = function(x), function(y)
    dx, = torch.autograd.grad(first.sum(), x)
    dy, = torch.autograd.grad(second.sum(), y)
    np.testing.assert_array_equal(dx.numpy(), [4, 6])
    np.testing.assert_array_equal(dy.numpy(), [8, 10])
    assert isinstance(first, jt.Var)
    assert "_fwd_input_shapes" not in vars(function)
    assert "_saved_tensors" not in vars(function)
    with pytest.raises(RuntimeError, match="scalar outputs"):
        torch.autograd.grad(first, x)


def test_library_real_cpu_dispatch_registered_backward_and_schema():
    import torch

    lib = torch.library.Library("owner_gradient", "DEF")
    lib.define("cube(Tensor x) -> Tensor")
    lib.impl("cube", lambda x: (x * x * x).detach(), "CPU")
    lib.impl("cube", lambda x: x + 1000, "Meta")

    def setup(ctx, inputs, output):
        ctx.save_for_backward(inputs[0])

    def backward(ctx, grad):
        x, = ctx.saved_tensors
        return grad * 3 * x * x

    torch.library.register_autograd("owner_gradient::cube", backward, setup_context=setup)
    x = torch.tensor([2., 3.], requires_grad=True)
    out = torch.ops.owner_gradient.cube(x)
    np.testing.assert_array_equal(out.numpy(), [8, 27])
    dx, = torch.autograd.grad(out.sum(), x)
    np.testing.assert_array_equal(dx.numpy(), [12, 27])

    def prototype(x: torch.Tensor, scale: float = 2.) -> torch.Tensor:
        return x * scale

    assert torch.library.infer_schema(prototype, mutates_args=[]) == \
        "(Tensor x, float scale=2.0) -> Tensor"
    with pytest.raises(RuntimeError, match="already has an implementation"):
        lib.impl("cube", lambda x: x, "CPU")


def test_registration_failure_restores_owned_slots_and_existing_ops():
    import torch
    from jittor.compat.transaction import runtime_hook

    lib = torch.library.Library("owner_rollback", "DEF")
    lib.define("existing(Tensor x) -> Tensor")
    original = lambda x: x + 1
    lib.impl("existing", original, "CPU")
    op = torch.ops.owner_rollback.existing
    with pytest.raises(RuntimeError, match="deliberate registration failure"):
        with runtime_hook("autograd-library-failure"):
            lib.impl("existing", lambda x: x + 3, "CPU", allow_override=True)
            torch.library.register_fake(op, lambda x: x)
            torch.library.register_autograd(op, lambda ctx, grad: grad)
            fresh = torch.library.Library("owner_rollback_new", "DEF")
            fresh.define("f(Tensor x) -> Tensor")
            fresh.impl("f", lambda x: x, "CPU")
            raise RuntimeError("deliberate registration failure")
    assert op._implementations == {"CPU": original}
    assert op._fake_impl is None
    assert op._backward is None
    assert "owner_rollback_new" not in vars(torch.ops)["_namespaces"]


def test_rebinding_keeps_registry_state_and_all_api_objects():
    import torch
    import jittor as jt
    from jittor.compat.torch.context import get_install_context
    from jittor.compat.torch import library
    from jittor.compat.torch.installers import autograd

    ctx = get_install_context(jt)
    dispatcher, function = torch.ops, torch.autograd.Function
    lib = torch.library.Library("owner_rebind", "DEF")
    lib.define("f(Tensor x) -> Tensor")
    op = torch.ops.owner_rebind.f
    autograd.install(ctx)
    autograd.install_parity(ctx)
    library.install_torch_library(ctx.jittor_module, ctx.registry.module_map)
    assert torch.ops is dispatcher
    assert torch.ops.owner_rebind.f is op
    assert torch.autograd.Function is function
    assert torch.library.Library is library.Library


def test_installers_create_no_function_class_or_lambda_implementations():
    from jittor.compat.torch import library
    from jittor.compat.torch.installers import autograd

    for module, names in ((autograd, {"install", "install_parity", "install_tensordict",
                                     "_install_autograd", "_install_autograd_function",
                                     "_install_tensordict_compat"}),
                          (library, {"install_torch_library", "make_infer_schema"})):
        tree = ast.parse(Path(module.__file__).read_text())
        for function in tree.body:
            if isinstance(function, ast.FunctionDef) and function.name in names:
                assert not any(isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Lambda))
                               for node in ast.walk(function) if node is not function)
