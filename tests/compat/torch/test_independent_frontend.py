"""Native and Torch types coexist in one interpreter and one graph runtime."""

import textwrap

from _helpers.child_process import run_python_child


def test_deployed_torch_entry_defaults_to_independent_types(tmp_path):
    from jittor.compat.shim.deploy import deploy
    site = tmp_path / "site-packages"
    deploy(str(site))
    result = run_python_child(["-c", textwrap.dedent("""
        import os
        import sys
        import subprocess
        sys.path.insert(0, sys.argv[1])
        os.environ["JITTOR_TORCH_PROJECT_ROOT"] = sys.argv[2]
        os.environ["JITTOR_TORCH_RUNTIME_ROOT"] = sys.argv[2] + "/runtime"
        os.environ["JITTOR_TORCH_KEEP_HOME"] = "1"
        import torch
        import jittor as jt
        assert torch is not jt and torch.Tensor is not jt.Var
        assert torch.nn.Module is not jt.Module
        assert issubclass(torch.nn.Parameter, torch.Tensor)
        model = torch.nn.Linear(2, 1)
        value = model(torch.ones((2, 2)))
        assert type(value) is torch.Tensor
        value.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        assert os.environ["JITTOR_TORCH_INDEPENDENT"] == "1"
        subprocess.run([sys.executable, "-c",
            "import jittor as jt; import torch; "
            "assert torch is not jt; assert torch.Tensor is not jt.Var"],
            check=True, timeout=60)
        print("DEPLOYED_INDEPENDENT_OK")
    """), str(site), str(tmp_path)], cwd=str(tmp_path),
        without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "DEPLOYED_INDEPENDENT_OK" in result.stdout


def test_independent_tensor_installation_preserves_native_type():
    result = run_python_child(["-c", textwrap.dedent("""
        import os
        import copy
        import pickle
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
        assert issubclass(torch.nn.Parameter, torch.Tensor)
        source = torch.tensor([2., 3.], requires_grad=True)
        parameter = torch.nn.Parameter(source)
        assert type(parameter) is torch.nn.Parameter
        assert parameter is not source
        assert not getattr(source, "_is_torch_parameter", False)
        assert parameter.requires_grad and parameter.is_leaf
        assert parameter.grad_fn is None and source.requires_grad
        assert type(parameter + 1) is torch.Tensor
        assert type(parameter.detach()) is torch.Tensor
        class TaggedParameter(torch.nn.Parameter):
            def __new__(cls, data, label):
                return super().__new__(cls, data)
            def __init__(self, data, label):
                super().__init__(data)
                self.label = label
        tagged = TaggedParameter(source, "weight")
        assert type(tagged) is TaggedParameter and tagged.label == "weight"
        assert type(tagged * 2) is torch.Tensor
        for restored in (pickle.loads(pickle.dumps(parameter)), copy.deepcopy(parameter)):
            assert type(restored) is torch.nn.Parameter
            assert restored.requires_grad and restored.is_leaf
            np.testing.assert_allclose(restored.numpy(), [2., 3.])
        copied_tagged = copy.deepcopy(tagged)
        assert type(copied_tagged) is TaggedParameter and copied_tagged.label == "weight"
        restored_tensor = pickle.loads(pickle.dumps(source))
        assert type(restored_tensor) is torch.Tensor and restored_tensor.requires_grad
        parameters = torch.nn.ParameterList([source, parameter])
        assert type(parameters[0]) is torch.nn.Parameter and parameters[0] is not source
        assert parameters[1] is parameter
        parameters.append(tagged)
        assert parameters[1:][1] is tagged
        assert parameters.get_parameter("0") is parameters[0]
        assert list(parameters.state_dict()) == ["0", "1", "2"]
        mapping = torch.nn.ParameterDict({"weight": source, "alias": parameter})
        assert type(mapping["weight"]) is torch.nn.Parameter
        assert mapping.get_parameter("weight") is mapping["weight"]
        assert mapping["alias"] is parameter
        assert not getattr(source, "_is_torch_parameter", False)
        assert torch.nn.modules.parameter.ParameterList is torch.nn.ParameterList
        assert torch.nn.ParameterDict is not torch.nn.ParameterList
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
        output = model(torch.ones((2, 2)))
        assert type(output) is torch.Tensor
        assert all(type(p) is torch.nn.Parameter for p in model.parameters())
        output.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
        assert jt.autograd.get_policy() is policy_before
        print("INDEPENDENT_TENSOR_OK")
    """)], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "INDEPENDENT_TENSOR_OK" in result.stdout
