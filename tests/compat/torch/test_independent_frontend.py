"""Native and Torch types coexist in one interpreter and one graph runtime."""

import textwrap

from _helpers.child_process import run_python_child


def test_independent_tensor_installation_preserves_native_type():
    result = run_python_child(["-c", textwrap.dedent("""
        import os
        import numpy as np
        import jittor as jt
        if os.environ.get("use_cuda") == "1":
            assert jt.has_cuda and jt.flags.use_cuda
        before = dict(vars(jt.Var))
        module_before = dict(vars(jt.Module))
        linear_before = dict(vars(jt.nn.Linear))
        native_nn = jt.nn
        native_init = jt.nn.init
        policy_before = jt.autograd.get_policy()
        from jittor.compat.shim.runtime import activate
        from jittor.compat import torch as compatibility
        required = compatibility._REQUIRED_STEPS
        def fail_after_install(context):
            raise RuntimeError("injected frontend install failure")
        compatibility._REQUIRED_STEPS = required + (("frontend.failure", fail_after_install),)
        try:
            try:
                activate(
                    independent_namespace=True, auto_scan_extensions=False,
                    build_extensions=False, configure_cuda=False,
                    local_home=False, verbose=False,
                )
            except RuntimeError as error:
                assert "injected frontend install failure" in str(error)
            else:
                raise AssertionError("injected failure was swallowed")
        finally:
            compatibility._REQUIRED_STEPS = required
        torch = activate(
            independent_namespace=True, auto_scan_extensions=False,
            build_extensions=False, configure_cuda=False,
            local_home=False, verbose=False,
        )["torch"]
        assert torch.Tensor is not jt.Var
        assert issubclass(torch.Tensor, jt.Var)
        assert before.keys() == vars(jt.Var).keys()
        assert all(value is vars(jt.Var)[key] for key, value in before.items())
        assert module_before.keys() == vars(jt.Module).keys()
        assert all(value is vars(jt.Module)[key]
                   for key, value in module_before.items())
        assert linear_before.keys() == vars(jt.nn.Linear).keys()
        assert all(value is vars(jt.nn.Linear)[key]
                   for key, value in linear_before.items())
        assert torch.nn is not native_nn
        assert torch.nn.init is not native_init
        assert torch.nn.Module is not jt.Module
        assert issubclass(torch.nn.Linear, torch.nn.Module)
        assert jt.autograd.get_policy() is policy_before
        x = torch.tensor([1., 2.], requires_grad=True)
        assert type(x) is torch.Tensor
        assert x.dtype is torch.float32
        if jt.flags.use_cuda:
            x.sync()
            assert x.location() == "device"
        (x * x).sum().backward()
        np.testing.assert_allclose(x.grad.numpy(), [2., 4.])
        for value in (torch.ones(2), torch.Tensor(2), torch.eye(2),
                      torch.FloatTensor([1., 2.]),
                      torch.from_numpy(np.ones(2, dtype=np.float32))):
            assert type(value) is torch.Tensor
        data = x.data
        assert type(data) is torch.Tensor
        with torch.no_grad():
            data[0].fill_(3)
        np.testing.assert_allclose(x.numpy(), [3., 2.])
        assert x.requires_grad
        plain = torch.ones(2)
        assert not plain.requires_grad
        assert not (plain + 1).requires_grad
        assert not x.detach().requires_grad
        assert jt.autograd.get_policy() is policy_before
        assert type(jt.array([1.])) is jt.Var
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        output = model(torch.ones((2, 2)))
        assert type(output) is torch.Tensor
        assert all(type(p) is torch.Tensor for p in model.parameters())
        output.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        assert jt.autograd.get_policy() is policy_before
        print("INDEPENDENT_TENSOR_OK")
    """)], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "INDEPENDENT_TENSOR_OK" in result.stdout
