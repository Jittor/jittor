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
        optimizer_types = (jt.optim.Optimizer, jt.optim.SGD, jt.optim.Adam, jt.optim.AdamW)
        optimizer_before = [(cls, dict(vars(cls))) for cls in optimizer_types]
        function_before = dict(vars(jt.Function))
        autograd_before = dict(vars(jt.autograd))
        namespace_before = [(module, dict(vars(module)))
                            for module in (jt.linalg, jt.sparse, jt.distributions)]
        distribution_types = [(cls, dict(vars(cls))) for cls in
                              (jt.distributions.Distribution, jt.distributions.Normal,
                               jt.distributions.Categorical, jt.distributions.Bernoulli)]
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
        for cls, original in optimizer_before:
            assert original.keys() == vars(cls).keys()
            assert all(value is vars(cls)[key] for key, value in original.items())
        assert torch.optim is not jt.optim
        assert issubclass(torch.optim.SGD, torch.optim.Optimizer)
        assert torch.autograd is not jt.autograd
        assert torch.autograd.Function is not jt.Function
        assert function_before.keys() == vars(jt.Function).keys()
        assert all(value is vars(jt.Function)[key] for key, value in function_before.items())
        assert autograd_before.keys() == vars(jt.autograd).keys()
        assert all(value is vars(jt.autograd)[key] for key, value in autograd_before.items())
        for module, original in namespace_before + distribution_types:
            assert original.keys() == vars(module).keys()
            assert all(value is vars(module)[key] for key, value in original.items())
        assert torch.linalg is not jt.linalg and torch.sparse is not jt.sparse
        assert torch.distributions is not jt.distributions
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
        weight = model[0].weight
        model[0].alias = weight
        model.register_buffer("floating", torch.ones(2))
        model.register_buffer("integer", torch.ones(2, dtype=torch.int64))
        model.to(dtype=torch.float64)
        assert model[0].weight is weight and model[0].alias is weight
        assert type(weight) is torch.nn.Parameter and weight.is_leaf
        assert weight.dtype is torch.float64 and weight.grad.dtype is torch.float64
        assert model.floating.dtype is torch.float64
        assert model.integer.dtype is torch.int64
        parameters.double()
        assert all(p.dtype is torch.float64 for p in parameters)
        for algorithm, options, expected in (
            (torch.optim.SGD, {}, [0.8, 1.6]),
            (torch.optim.Adam, {}, [0.9, 1.9]),
            (torch.optim.AdamW, {"weight_decay": 0.1}, [0.89, 1.88]),
        ):
            trainable = torch.nn.Parameter(torch.tensor([1., 2.]))
            optimizer = algorithm((p for p in [trainable]), lr=0.1, **options)
            (trainable * trainable).sum().backward()
            optimizer.step()
            np.testing.assert_allclose(trainable.numpy(), expected, atol=1e-6)
            assert type(trainable) is torch.nn.Parameter
            state = optimizer.state_dict()
            for values in state["state"].values():
                assert all(isinstance(value, torch.Tensor) for value in values.values()
                           if isinstance(value, jt.Var))
            from jittor.compat.optimizer_kinds import kind_of
            assert kind_of(optimizer, require_unmodified_step=True) is not None
            optimizer.load_state_dict(state)
            assert optimizer.param_groups[0]["params"][0] is trainable
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
            scheduler.step()
            assert abs(optimizer.param_groups[0]["lr"] - 0.05) < 1e-12
            optimizer.zero_grad(set_to_none=True)
            assert trainable.grad is None
        assert "_current_optimizer" not in vars(jt)
        class Square(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                ctx.save_for_backward(value)
                return value * value
            @staticmethod
            def backward(ctx, gradient):
                value, = ctx.saved_tensors
                return gradient * value * 2
        custom_input = torch.tensor([2., 3.], requires_grad=True)
        gradient, = torch.autograd.grad(Square.apply(custom_input).sum(), custom_input)
        assert type(gradient) is torch.Tensor
        np.testing.assert_allclose(gradient.numpy(), [4., 6.])
        collate = torch.utils.data.default_collate
        batch = collate([{"index": 2**45, "value": np.float32(1.25)},
                         {"index": 2**45 + 1, "value": np.float32(2.5)}])
        assert type(batch["index"]) is torch.Tensor
        assert batch["index"].dtype is torch.int64
        assert batch["value"].dtype is torch.float32
        np.testing.assert_array_equal(batch["index"].numpy(), [2**45, 2**45 + 1])
        assert collate([1.25, 2.5]).dtype is torch.float64
        differentiable_batch = collate([custom_input, custom_input])
        batch_gradient, = torch.autograd.grad(differentiable_batch.sum(), custom_input)
        np.testing.assert_allclose(batch_gradient.numpy(), [2., 2.])
        import io
        checkpoint = io.BytesIO()
        saved = {"wide": torch.tensor([2**45, 2**45+1], dtype=torch.int64),
                 "bf16": torch.tensor([1.25, 2.5], dtype=torch.bfloat16),
                 "parameter": torch.nn.Parameter(torch.tensor([3., 4.]))}
        torch.save(saved, checkpoint)
        checkpoint.seek(0)
        loaded = torch.load(checkpoint, map_location="cpu")
        assert type(loaded["wide"]) is torch.Tensor
        assert loaded["wide"].dtype is torch.int64
        np.testing.assert_array_equal(loaded["wide"].numpy(), [2**45, 2**45+1])
        assert loaded["bf16"].dtype is torch.bfloat16
        assert type(loaded["parameter"]) is torch.nn.Parameter
        assert loaded["parameter"].requires_grad and loaded["parameter"].is_leaf
        if jt.flags.use_cuda and torch.cuda.device_count() >= 2:
            on_second = torch.nn.Parameter(torch.tensor([5., 6.], device="cuda:1"))
            on_second.sync()
            assert on_second.device_id == 1
            archive = io.BytesIO()
            torch.save(on_second, archive)
            assert on_second.device_id == 1 and on_second.location() == "device"
            archive.seek(0)
            restored = torch.load(archive, map_location="cuda:1")
            restored.sync()
            assert type(restored) is torch.nn.Parameter and restored.device_id == 1
            np.testing.assert_array_equal(restored.numpy(), [5., 6.])
            archive.seek(0)
            restored_default = torch.load(archive)
            restored_default.sync()
            assert restored_default.device_id == 1
            archive.seek(0)
            remapped = torch.load(archive, map_location={"cuda:1": "cpu"})
            assert type(remapped) is torch.nn.Parameter and remapped.is_cpu
            converted_model = torch.nn.Linear(2, 1)
            original_weight = converted_model.weight
            converted_model.to(device="cpu", dtype=torch.float64)
            assert converted_model.weight is original_weight and original_weight.is_cpu
            assert original_weight.dtype is torch.float64 and original_weight.is_leaf
        assert jt.autograd.get_policy() is policy_before
        normal = torch.distributions.Normal(loc=0.0, scale=1.0)
        assert isinstance(normal, torch.distributions.Distribution)
        assert type(normal.loc) is torch.Tensor and not normal.loc.requires_grad
        assert type(normal.sample((3,))) is torch.Tensor
        assert not normal.rsample((3,)).requires_grad
        np.testing.assert_allclose(normal.log_prob(torch.tensor(0.)).numpy(),
                                   -0.5 * np.log(2 * np.pi), atol=1e-6)
        loc = torch.tensor(0., requires_grad=True)
        sampled = torch.distributions.Normal(loc=loc, scale=1.).rsample((3,))
        loc_gradient, = torch.autograd.grad(sampled.sum(), loc)
        np.testing.assert_allclose(loc_gradient.numpy(), 3.)
        categorical = torch.distributions.Categorical(probs=torch.tensor([0.25, 0.75]))
        assert type(categorical.sample((3,))) is torch.Tensor
        np.testing.assert_allclose(categorical.log_prob(torch.tensor(1)).numpy(), np.log(0.75), atol=1e-6)
        bernoulli = torch.distributions.Bernoulli(probs=0.25)
        assert type(bernoulli.sample((3,))) is torch.Tensor
        np.testing.assert_allclose(bernoulli.log_prob(torch.tensor(1.)).numpy(), np.log(0.25), atol=1e-6)
        matrix = torch.tensor([[2., 0.], [0., 4.]])
        inverse = torch.linalg.inv(matrix)
        assert type(inverse) is torch.Tensor
        np.testing.assert_allclose(inverse.numpy(), [[0.5, 0.], [0., 0.25]])
        print("INDEPENDENT_TENSOR_OK")
    """)], without_torch_mode=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
    assert "INDEPENDENT_TENSOR_OK" in result.stdout
