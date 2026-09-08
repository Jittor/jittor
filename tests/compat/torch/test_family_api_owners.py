"""Family implementations are importable objects rather than install closures."""
import ast
import importlib
import inspect
import pickle
import textwrap

import numpy as np

import jittor as jt
from jittor.compat.torch.tensor_state import compatibility_owner


def test_family_installers_only_bind_objects():
    entries = {
        "data": ("install", "_install_torchdata_stateful_dataloader"),
        "tensor": ("install", "install_methods"),
        "tensor.methods": ("_install_tensor_methods",),
        "nn.extras": ("_install_nn_extras",),
        "nn.functional": ("_install_functional",),
        "nn.attention": ("install_attention",),
        "nn_init": ("_install_init_aliases",),
        "cuda": ("_install_cuda", "_install_accelerator"),
        "utilities": ("install", "install_parity", "install_runtime_knobs"),
        "compiler": ("install", "install_parity"),
    }
    for family, names in entries.items():
        module = importlib.import_module("jittor.compat.torch.installers." + family)
        for name in names:
            node = ast.parse(textwrap.dedent(inspect.getsource(getattr(module, name)))).body[0]
            nested = [child for child in ast.walk(node) if child is not node and
                      isinstance(child, (ast.FunctionDef, ast.ClassDef, ast.Lambda))]
            assert not nested, (family, name, [getattr(child, "name", "lambda") for child in nested])


def test_public_family_objects_have_real_importable_owners():
    torch = compatibility_owner(jt)
    tensor = importlib.import_module("jittor.compat.torch.installers.tensor")
    methods = importlib.import_module("jittor.compat.torch.installers.tensor.method_api")
    autograd = importlib.import_module("jittor.compat.torch.installers.tensor.autograd_api")
    data = importlib.import_module("jittor.compat.torch.installers.data")
    functional = importlib.import_module("jittor.compat.torch.installers.nn.functional")
    attention = importlib.import_module("jittor.compat.torch.installers.nn.attention")
    pairs = [(getattr(torch, name), getattr(tensor, name))
             for name in ("tensor", "as_tensor", "from_numpy", "frombuffer", "cat",
                          "stack", "no_grad", "enable_grad", "inference_mode", "FloatTensor")]
    pairs += [(torch.Var.backward, autograd._backward),
              (torch.Var.__add__, methods._tensor_add),
              (torch.Var.copy_, methods._copy_),
              (torch.utils.data.DataLoader, data._DataLoader),
              (torch.utils.data.default_collate, data._default_collate),
              (torch.nn.functional.softmax, functional._softmax),
              (torch.nn.functional.scaled_dot_product_attention, attention.scaled_dot_product_attention)]
    cuda = importlib.import_module("jittor.compat.torch.installers.cuda.api")
    utilities = importlib.import_module("jittor.compat.torch.installers.utilities")
    compiler = importlib.import_module("jittor.compat.torch.installers.compiler")
    pairs += [(torch.cuda.Stream, cuda._Stream),
              (torch.cuda.current_stream, cuda._current_stream),
              (torch.utils._pytree.tree_map, utilities._tree_map),
              (torch.hub.load_state_dict_from_url, utilities._load_state_dict_from_url),
              (torch.func.functional_call, compiler._functional_call),
              (torch.func.grad, compiler._func_grad)]
    for actual, expected in pairs:
        assert actual is expected
        assert "<locals>" not in actual.__qualname__
        assert getattr(importlib.import_module(actual.__module__), actual.__name__) is actual
        assert pickle.loads(pickle.dumps(actual)) is actual


def test_imported_apis_preserve_training_shapes_and_data_values():
    torch = compatibility_owner(jt)
    tensor = importlib.import_module("jittor.compat.torch.installers.tensor")
    data = importlib.import_module("jittor.compat.torch.installers.data")
    init = importlib.import_module("jittor.compat.torch.installers.nn_init")
    value = tensor.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    loss = (value * value).sum(dim=(0, 1))
    loss.backward()
    np.testing.assert_allclose(value.grad.numpy(), [[2., 4.], [6., 8.]])
    np.testing.assert_array_equal(value.transpose(0, 1).reshape(4).numpy(), [1., 3., 2., 4.])
    assert value.new_zeros((1, 2)).dtype == value.dtype
    init.uniform_(value, low=1., high=1.)
    assert value.requires_grad
    np.testing.assert_array_equal(value.numpy(), np.ones((2, 2)))
    loaded = next(iter(data._DataLoader([2**40, 2**40 + 1], batch_size=2)))
    assert loaded.dtype is torch.int64
    np.testing.assert_array_equal(loaded.numpy(), [2**40, 2**40 + 1])
    model = torch.nn.Linear(2, 1)
    result = model(tensor.tensor([[1., 2.]]))
    result.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    value.requires_grad_(False)
    from jittor.compat.torch.nested import _torch_prune_leaf_registry
    _torch_prune_leaf_registry()
    assert not ({"_torch_leaf_params", "_torch_retained", "_torch_tensor_state",
                 "_current_optimizer", "_active_optimizers",
                 "_transform_getitem_to_index_depth", "_torch_shim_runtime_state"}
                & vars(jt).keys())


def test_fidelity_table_reports_the_actual_final_binding():
    from jittor.compat.torch.fidelity import fidelity_of, fidelity_table
    torch = compatibility_owner(jt)
    for name, implementation in (("torch.tensor", torch.tensor),
                                 ("torch.Tensor.backward", torch.Var.backward),
                                 ("torch.nn.init.uniform_", torch.nn.init.uniform_),
                                 ("torch.utils.data.DataLoader", torch.utils.data.DataLoader)):
        record = fidelity_of(name)
        assert record.implementation is implementation
        assert record.detail
    table = fidelity_table("torch.Tensor.")
    assert "| torch.Tensor.backward |" in table
    assert "autograd_api._backward" in table
    assert "| torch.tensor |" not in table


def test_module_templates_keep_frontend_parameters_and_functional_values(monkeypatch):
    monkeypatch.setenv("JT_BUILD_USE_MKL", "0")
    torch = compatibility_owner(jt)
    layer = torch.nn.TransformerEncoderLayer(4, 2, dim_feedforward=8, dropout=0.0)
    assert isinstance(layer, torch.nn.Module)
    value = torch.tensor(np.arange(24, dtype=np.float32).reshape(3, 2, 4) / 24)
    result = layer(value)
    assert tuple(result.shape) == (3, 2, 4)
    assert np.isfinite(result.numpy()).all()
    assert list(layer.parameters())
    criterion = torch.nn.HuberLoss()
    loss = criterion(torch.tensor([1., 2.]), torch.tensor([0., 0.]))
    np.testing.assert_allclose(loss.numpy(), 1.0)
    wrapped = torch.nn.DataParallel(torch.nn.Identity())
    np.testing.assert_array_equal(wrapped(value).numpy(), value.numpy())
    for parameter in layer.parameters():
        parameter.requires_grad_(False)
    from jittor.compat.torch.nested import _torch_prune_leaf_registry
    _torch_prune_leaf_registry()
