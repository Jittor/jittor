"""Publish ``torch.distributed.pipelining`` with a single-stage GPipe schedule.

The module is a package (as upstream) with ``schedules``, ``stage``,
``microbatch`` and ``_IR`` children, because library code imports those paths
directly.  The implementation lives in :mod:`jittor.compat.torch.pipelining`;
this installer only builds the module graph and records fidelity.
"""
from __future__ import absolute_import

from .. import pipelining as _pipelining
from ..fidelity import Fidelity, register_api_bindings


_PUBLIC = (
    "Pipe", "PipelineStage", "ScheduleGPipe", "SplitPoint", "pipeline",
    "pipe_split", "TensorChunkSpec",
)

_SUBMODULES = (
    ("torch.distributed.pipelining._IR",
     ("Pipe", "SplitPoint", "pipeline", "pipe_split")),
    ("torch.distributed.pipelining.schedules", ("ScheduleGPipe",)),
    ("torch.distributed.pipelining.stage", ("PipelineStage",)),
    ("torch.distributed.pipelining.microbatch",
     ("TensorChunkSpec", "split_args_kwargs_into_chunks", "merge_chunks")),
)

_DETAIL = (
    "Implements the forward-only GPipe schedule for a single stage: whole-batch "
    "inputs are split into microbatches along dimension 0, executed in order on "
    "one device, and merged by concatenation. Model partitioning into more than "
    "one stage, multi-rank activation transfer, and backward passes are refused "
    "with NotImplementedError; torch.export is not used to build the pipe.")


def install(ctx):
    module = ctx.registry.ensure("torch.distributed.pipelining", package=True)
    module.__doc__ = _pipelining.__doc__
    module.__all__ = tuple(_PUBLIC)
    for name in _PUBLIC:
        setattr(module, name, getattr(_pipelining, name))
    for module_name, names in _SUBMODULES:
        child = ctx.registry.ensure(module_name)
        for name in names:
            setattr(child, name, getattr(_pipelining, name))
    register_api_bindings(
        module, "torch.distributed.pipelining",
        ("pipeline", "Pipe", "PipelineStage", "ScheduleGPipe", "SplitPoint"),
        Fidelity.APPROXIMATE, _DETAIL)
