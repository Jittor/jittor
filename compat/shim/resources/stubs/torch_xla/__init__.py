"""Jittor emulation of the PyTorch/XLA client API (``torch_xla``).

This is **not** PyTorch/XLA and **not** a TPU runtime.  ``jittor-compat``
deploys this package so that code written for TPUs keeps its launch and
device-control structure while running on Jittor's CUDA or CPU backend.  The
fidelity statement is available as ``torch_xla.__jittor_emulation_fidelity__``
and in :mod:`torch_xla._emulation`.

Only the client surface that launch scripts, Transformers and Accelerate touch
is provided: ``distributed.xla_multiprocessing.spawn``, ``core.xla_model``
device/ordinal/rendezvous helpers, ``runtime`` queries, a pass-through
``distributed.parallel_loader``, and empty ``debug.metrics``/``distributed.spmd``
imports.  Real XLA compilation, SPMD sharding, TPU hardware and XLA FSDP are
not emulated.
"""

from __future__ import absolute_import

import torch as _torch

from . import _emulation

if not _emulation.is_jittor_frontend(_torch):
    raise RuntimeError(
        "torch_xla emulation requires the Jittor Torch frontend; the imported "
        "'torch' module is real PyTorch. Import the Jittor torch shim first "
        "(the deployed torch entry activates it) instead of installing this "
        "emulation next to a binary PyTorch."
    )

#: API level of the emulated surface.  Transformers branches on ``>= 2.2`` to
#: decide whether the ``distributed.spmd`` namespace must exist; the emulation
#: keeps that namespace importable and raises on use.
__version__ = "2.9.0"
__xla_emulation__ = True
__jittor_emulation__ = True
__jittor_emulation_fidelity__ = _emulation.FIDELITY

from . import runtime as runtime  # noqa: E402  (eager public surface)

del _torch


def device(n=None, devkind=None):
    """Return the Jittor device emulating ``xla:N`` (see ``core.xla_model``)."""
    from .core.xla_model import xla_device

    return xla_device(n, devkind)


def sync(wait=False):
    """No-op stand-in for ``torch_xla.sync``; Jittor executes eagerly.

    With ``wait=True`` this synchronizes the CUDA device, which is the closest
    observable equivalent of draining the lazy XLA graph.
    """
    if wait:
        from .core.xla_model import wait_device_ops

        wait_device_ops()


def launch(fn, args=(), nprocs=None, start_method="spawn"):
    """Alias of ``distributed.xla_multiprocessing.spawn`` (newer XLA spelling)."""
    from .distributed import xla_multiprocessing

    return xla_multiprocessing.spawn(fn, args=args, nprocs=nprocs, start_method=start_method)
