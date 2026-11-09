"""``torch.export`` is an eager, direct-call exported program.

These tests pin the supported contract -- export() validates the example
inputs, the ExportedProgram is callable with the module's calling convention
and exposes the module's weights -- and the explicit boundary: no graph
artifact exists (or, once torch.fx provides a tracer, the graph exposed is the
traced one and execution stays eager).
"""
import importlib

import numpy as np
import pytest

import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, x):
        return torch.relu(self.linear(x))


class TwoInputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = 2.0

    def forward(self, left, right):
        return left + right * self.scale


class TupleOutputModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x):
        y = self.linear(x)
        return y, y + 1.0


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def test_public_surface_is_published_under_torch():
    export_module = importlib.import_module("torch.export")
    assert torch.export is export_module
    for name in ("export", "ExportedProgram", "Constraint", "Dim", "dims"):
        assert hasattr(export_module, name), name

    exported_program = importlib.import_module("torch.export.exported_program")
    assert exported_program.ExportedProgram is export_module.ExportedProgram
    dynamic_shapes = importlib.import_module("torch.export.dynamic_shapes")
    assert dynamic_shapes.Dim is export_module.Dim
    assert dynamic_shapes.dims is export_module.dims


def test_export_returns_a_callable_with_the_module_semantics():
    model = TinyModel()
    inputs = torch.randn(5, 4)
    program = torch.export.export(model, (inputs,))
    assert isinstance(program, torch.export.ExportedProgram)

    expected = model(inputs)
    actual = program(inputs)
    np.testing.assert_allclose(_numpy(actual), _numpy(expected), rtol=1e-6)


def test_export_supports_keyword_arguments_and_tuple_outputs():
    model = TupleOutputModel()
    inputs = torch.randn(3, 4)
    program = torch.export.export(model, (), {"x": inputs})
    first, second = program(x=inputs)
    expected_first, expected_second = model(x=inputs)
    np.testing.assert_allclose(_numpy(first), _numpy(expected_first), rtol=1e-6)
    np.testing.assert_allclose(_numpy(second), _numpy(expected_second), rtol=1e-6)


def test_export_state_dict_parameters_and_module_delegate_to_the_module():
    model = TinyModel()
    inputs = torch.randn(2, 4)
    program = torch.export.export(model, (inputs,))

    module_state = model.state_dict()
    program_state = program.state_dict
    assert set(program_state) == set(module_state)
    for key in module_state:
        np.testing.assert_array_equal(
            _numpy(program_state[key]), _numpy(module_state[key]))

    assert len(list(program.parameters())) == len(list(model.parameters()))
    assert len(list(program.buffers())) == len(list(model.buffers()))
    assert program.module() is model
    example_args, example_kwargs = program.example_inputs
    assert len(example_args) == 1
    np.testing.assert_array_equal(_numpy(example_args[0]), _numpy(inputs))
    assert example_kwargs == {}


def test_export_rejects_unusable_examples_at_export_time():
    model = TwoInputModel()
    with pytest.raises(TypeError):
        torch.export.export(model, (torch.randn(2, 2),))
    with pytest.raises(TypeError):
        torch.export.export(model, torch.randn(2, 2))
    with pytest.raises(TypeError):
        torch.export.export("not a module", ())


def test_graph_artifacts_are_either_real_or_refused_by_name():
    model = TinyModel()
    program = torch.export.export(model, (torch.randn(1, 4),))

    for name in ("graph_module", "graph"):
        try:
            artifact = getattr(program, name)
        except NotImplementedError as error:
            assert "torch.export.ExportedProgram." + name in str(error)
            assert "graph" in str(error)
        else:
            assert artifact is not None
    with pytest.raises(NotImplementedError, match="graph_signature"):
        program.graph_signature


def test_dynamic_shapes_are_recorded_but_do_not_constrain():
    model = TinyModel()
    inputs = torch.randn(1, 4)
    batch = torch.export.Dim("batch", min=1, max=8)
    program = torch.export.export(
        model, (inputs,), dynamic_shapes={"x": {0: batch}})
    assert program.dynamic_shapes == {"x": {0: batch}}
    # A batch above the recorded maximum still runs: nothing validates shapes.
    other = torch.randn(32, 4)
    np.testing.assert_allclose(
        _numpy(program(other)), _numpy(model(other)), rtol=1e-6)


def test_dim_carries_name_and_bounds():
    batch = torch.export.Dim("batch", max=8)
    assert batch.__name__ == "batch"
    assert batch.min == 0
    assert batch.max == 8
    unbounded = torch.export.Dim("seq")
    assert unbounded.min == 0
    assert repr(unbounded.max) == "int_oo"
    assert torch.export.Dim.AUTO != torch.export.Dim.STATIC

    named = torch.export.dims("left", "right", min=1, max=4)
    assert tuple(dim.__name__ for dim in named) == ("left", "right")
    assert all(dim.min == 1 and dim.max == 4 for dim in named)

    with pytest.raises((AssertionError, ValueError)):
        torch.export.Dim("not an identifier")
    with pytest.raises(AssertionError):
        torch.export.Dim("bad", min=4, max=2)


def test_export_fidelity_is_approximate_and_explains_the_gap():
    fidelity = importlib.import_module("jittor.compat.torch.fidelity")
    record = fidelity.fidelity_of("torch.export.export")
    assert record.level is fidelity.Fidelity.APPROXIMATE
    assert record.implementation is torch.export.export
    assert "graph" in record.detail.lower()
