"""``torch.distributed.pipelining``: single-stage GPipe, forward-only.

The tests pin what ``accelerate.prepare_pippy`` drives in a one-process
launch: ``pipeline(model, mb_args=..., split_spec={})``, ``build_stage(0)``,
``ScheduleGPipe(stage, num_chunks)`` and ``schedule.step(batch)``.  They also
pin the refused boundaries -- model partitioning into several stages,
multi-rank execution, and training -- so those never silently degrade into a
whole-model call.
"""
import importlib

import numpy as np
import pytest

import torch


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(4, 8)
        self.second = torch.nn.Linear(8, 2)

    def forward(self, x):
        return self.second(torch.relu(self.first(x)))


class KwargModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x, scale=1.0):
        return self.linear(x) * scale


class PairModel(torch.nn.Module):
    def forward(self, x):
        return x, x * 2.0


def _numpy(value):
    return value.numpy() if hasattr(value, "numpy") else np.asarray(value)


def test_public_surface_is_published_under_torch():
    pipelining = importlib.import_module("torch.distributed.pipelining")
    assert torch.distributed.pipelining is pipelining
    for name in ("Pipe", "PipelineStage", "ScheduleGPipe", "SplitPoint",
                 "pipeline", "pipe_split", "TensorChunkSpec"):
        assert hasattr(pipelining, name), name
    assert pipelining.SplitPoint.BEGINNING.value == 1
    assert pipelining.SplitPoint.END.value == 2

    schedules = importlib.import_module("torch.distributed.pipelining.schedules")
    assert schedules.ScheduleGPipe is pipelining.ScheduleGPipe
    stage = importlib.import_module("torch.distributed.pipelining.stage")
    assert stage.PipelineStage is pipelining.PipelineStage
    microbatch = importlib.import_module(
        "torch.distributed.pipelining.microbatch")
    assert microbatch.TensorChunkSpec is pipelining.TensorChunkSpec
    ir = importlib.import_module("torch.distributed.pipelining._IR")
    assert ir.pipeline is pipelining.pipeline


def test_accelerate_single_process_workflow_end_to_end():
    # The exact calls accelerate.inference.build_pipeline makes when
    # state.num_processes == 1 (split_points == [] -> split_spec == {}).
    model = MLP()
    example = torch.randn(1, 4)
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(example,), mb_kwargs=None, split_spec={})
    assert pipe.num_stages == 1
    assert pipe.info().num_stages == 1
    assert pipe.info().has_loss_and_backward is False

    stage = pipe.build_stage(0, device=torch.device("cpu"))
    assert stage.stage_index == 0
    assert stage.is_first and stage.is_last
    assert stage.submod is model

    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 1)
    batch = torch.randn(2, 4)
    with torch.no_grad():
        output = schedule.step(batch)
    expected = model(batch)
    np.testing.assert_allclose(_numpy(output), _numpy(expected), rtol=1e-6)


def test_microbatches_run_in_order_and_merge_along_the_batch_dimension():
    model = MLP()
    example = torch.randn(1, 4)
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(example,), split_spec={})
    stage = pipe.build_stage(0, device=None)
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 4)

    batch = torch.randn(9, 4)  # 9 does not divide evenly into 4
    with torch.no_grad():
        output = schedule.step(batch)
    expected = model(batch)
    np.testing.assert_allclose(_numpy(output), _numpy(expected), rtol=1e-6)

    # Downstream (accelerate.pippy_forward) calls stage.step, where "stage" is
    # the schedule object; it must expose the same bound method.
    assert callable(schedule.step)


def test_keyword_arguments_are_split_with_the_batch():
    model = KwargModel()
    example = torch.randn(1, 4)
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(example,), mb_kwargs={"scale": 3.0}, split_spec={})
    stage = pipe.build_stage(0, device=torch.device("cpu"))
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 2)
    batch = torch.randn(4, 4)
    with torch.no_grad():
        output = schedule.step(batch, scale=3.0)
    np.testing.assert_allclose(
        _numpy(output), _numpy(model(batch, scale=3.0)), rtol=1e-6)


def test_container_and_tuple_outputs_merge_back_to_their_structure():
    model = PairModel()
    example = torch.randn(1, 4)
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(example,), split_spec={})
    stage = pipe.build_stage(0, device=None)
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 2)
    batch = torch.randn(4, 4)
    with torch.no_grad():
        first, second = schedule.step(batch)
    np.testing.assert_allclose(_numpy(first), _numpy(batch), rtol=1e-6)
    np.testing.assert_allclose(_numpy(second), _numpy(batch) * 2.0, rtol=1e-6)


def test_registered_output_container_survives_merge_with_one_leaf():
    # transformers registers ModelOutput subclasses as pytree nodes; a model
    # that returns only ``logits`` flattens to one leaf, and merging must
    # still rebuild the container rather than return the bare tensor.
    from torch.utils._pytree import register_pytree_node

    class Output(object):
        def __init__(self, logits):
            self.logits = logits

    register_pytree_node(
        Output,
        lambda value: ([value.logits], "logits"),
        lambda values, context: Output(values[0]),
        serialized_type_name="test_torch_distributed_pipelining.Output")

    class OutputModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 2)

        def forward(self, x):
            return Output(self.linear(x))

    model = OutputModel()
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(torch.randn(1, 4),), split_spec={})
    stage = pipe.build_stage(0, device=None)
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 2)
    batch = torch.randn(4, 4)
    with torch.no_grad():
        output = schedule.step(batch)
    assert isinstance(output, Output)
    np.testing.assert_allclose(
        _numpy(output.logits), _numpy(model.linear(batch)), rtol=1e-6)


def test_pipe_runs_the_whole_pipeline_locally():
    model = MLP()
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(torch.randn(1, 4),), split_spec={})
    batch = torch.randn(3, 4)
    np.testing.assert_allclose(
        _numpy(pipe(batch)), _numpy(model(batch)), rtol=1e-6)


def test_replacing_module_forward_after_building_the_pipe_does_not_recurse():
    # accelerate.prepare_pippy builds the stage, then replaces model.forward
    # with a wrapper that calls schedule.step.  The stage must keep the
    # forward it was built from, exactly as upstream executes a captured
    # GraphModule rather than re-entering model.forward.
    model = MLP()
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(torch.randn(1, 4),), split_spec={})
    stage = pipe.build_stage(0, device=None)
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 1)

    batch = torch.randn(2, 4)
    expected = model(batch)

    def pipeline_forward(*args, **kwargs):
        return schedule.step(*args, **kwargs)

    model.forward = pipeline_forward
    with torch.no_grad():
        output = model(batch)
    np.testing.assert_allclose(_numpy(output), _numpy(expected), rtol=1e-6)

    # Pipe's local execution uses the captured forward too.
    np.testing.assert_allclose(_numpy(pipe(batch)), _numpy(expected), rtol=1e-6)


def test_chunk_specs_split_on_the_named_dimension():
    pipelining = importlib.import_module("torch.distributed.pipelining")
    microbatch = importlib.import_module(
        "torch.distributed.pipelining.microbatch")
    left = torch.randn(4, 2)
    right = torch.randn(4, 3)
    specs = pipelining.TensorChunkSpec.from_tuple((0, 1))
    args_split, kwargs_split = microbatch.split_args_kwargs_into_chunks(
        (left, right,), {}, 2, args_chunk_spec=specs)
    assert kwargs_split == [{}, {}]
    assert args_split[0][0].shape == (2, 2)
    # Dimension 1 has three entries; splitting into two gives 2 and 1.
    assert args_split[0][1].shape[1] + args_split[1][1].shape[1] == 3
    assert args_split[0][1].shape[0] == 4


def test_split_points_are_validated_then_refused_as_multi_stage():
    model = MLP()
    with pytest.raises(AttributeError, match="nonexistent module"):
        torch.distributed.pipelining.pipeline(
            model, mb_args=(torch.randn(1, 4),),
            split_spec={"no_such_layer": torch.distributed.pipelining.SplitPoint.BEGINNING})
    with pytest.raises(NotImplementedError, match="single stage"):
        torch.distributed.pipelining.pipeline(
            model, mb_args=(torch.randn(1, 4),),
            split_spec={"first": torch.distributed.pipelining.SplitPoint.BEGINNING})
    with pytest.raises(NotImplementedError, match="split_policy"):
        torch.distributed.pipelining.pipeline(
            model, mb_args=(torch.randn(1, 4),),
            split_policy=lambda graph_module: graph_module)


def test_multi_rank_execution_is_refused(monkeypatch):
    class _TwoRankDistributed(object):
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def is_initialized():
            return True

        @staticmethod
        def get_world_size():
            return 2

    monkeypatch.setattr(torch, "distributed", _TwoRankDistributed())
    pipelining = importlib.import_module("torch.distributed.pipelining")
    with pytest.raises(NotImplementedError, match="2 ranks"):
        pipelining.pipeline(MLP(), mb_args=(torch.randn(1, 4),), split_spec={})


def test_step_can_suppress_merged_outputs():
    model = MLP()
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(torch.randn(1, 4),), split_spec={})
    stage = pipe.build_stage(0, device=None)
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 1)
    batch = torch.randn(2, 4)
    with torch.no_grad():
        assert schedule.step(batch, return_outputs=False) is None
        # The run still happened, and the next call returns the output.
        output = schedule.step(batch)
    np.testing.assert_allclose(_numpy(output), _numpy(model(batch)), rtol=1e-6)


def test_training_and_invalid_schedules_are_refused():
    model = MLP()
    pipe = torch.distributed.pipelining.pipeline(
        model, mb_args=(torch.randn(1, 4),), split_spec={})
    stage = pipe.build_stage(0, device=None)
    with pytest.raises(NotImplementedError, match="backward"):
        torch.distributed.pipelining.ScheduleGPipe(
            stage, 1, loss_fn=lambda output, target: output.sum())
    schedule = torch.distributed.pipelining.ScheduleGPipe(stage, 1)
    with pytest.raises(NotImplementedError, match="loss"):
        schedule.step(torch.randn(2, 4), target=torch.zeros(2, 2))
    with pytest.raises(ValueError, match="microbatches"):
        torch.distributed.pipelining.ScheduleGPipe(stage, 0)
    with pytest.raises(ValueError, match="Invalid stage index"):
        pipe.build_stage(1, device=None)
    with pytest.raises(NotImplementedError, match="pipe_split"):
        torch.distributed.pipelining.pipe_split()


def test_pipelining_fidelity_is_approximate_and_explains_the_gap():
    fidelity = importlib.import_module("jittor.compat.torch.fidelity")
    record = fidelity.fidelity_of("torch.distributed.pipelining.pipeline")
    assert record.level is fidelity.Fidelity.APPROXIMATE
    assert record.implementation is torch.distributed.pipelining.pipeline
    assert "single stage" in record.detail.lower()
