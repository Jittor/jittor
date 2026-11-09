"""Emulation of ``torch_xla.core.xla_model`` on Jittor devices.

The public names mirror the PyTorch/XLA 2.x client API.  Semantics are
APPROXIMATE and are stated per function:

* devices are Jittor CUDA devices (or CPU), not XLA/PJRT devices;
* execution is eager, so ``mark_step`` has nothing to fuse and is a no-op;
* ``rendezvous`` and the collectives are filesystem exchanges performed over
  the same client protocol the real API documents (payloads are serialized
  with ``torch.save``), which is correct but slow and single-host;
* there is no cross-replica gradient synchronization beyond what
  ``all_reduce``/``reduce_gradients`` implement through those exchanges;
* ``get_rng_state``/``set_rng_state`` expose the Jittor host seed as an
  integer, not an XLA device stream position.
"""

from __future__ import absolute_import

import io
import sys

import torch

from .. import _emulation

REDUCE_SUM = "sum"
REDUCE_MUL = "mul"
REDUCE_AND = "and"
REDUCE_OR = "or"
REDUCE_MIN = "min"
REDUCE_MAX = "max"

_ALL_REDUCE_TAG = "torch_xla.core.xla_model.all_reduce"
_ALL_GATHER_TAG = "torch_xla.core.xla_model.all_gather"


def get_xla_supported_devices(max_devices=None):
    """Device strings this process can address (``cuda:N`` or ``cpu``).

    Real XLA returns ``xla:N`` names.  The emulation returns the Jittor device
    names those ordinals map to, because callers pass the result to
    ``torch.device`` and to replication setters that the emulation implements
    on real Jittor devices.
    """
    count = max(_emulation.visible_device_count(), 1)
    if max_devices:
        count = min(count, int(max_devices))
    if _emulation.device_type() == "GPU":
        return ["cuda:%d" % index for index in range(count)]
    return ["cpu"]


def get_ordinal():
    """Global ordinal (rank) of this process inside the emulated launch."""
    return _emulation.ordinal()


def get_world_size():
    """Number of processes in the emulated launch."""
    return _emulation.world_size()


def xrt_world_size():
    """Legacy alias of :func:`get_world_size` kept for TPU scripts."""
    return _emulation.world_size()


def get_local_ordinal():
    """Local ordinal of this process (single host: equals the global ordinal)."""
    return _emulation.local_ordinal()


def is_master_ordinal(local=True):
    """Whether this process is the (local or global) master ordinal 0."""
    ordinal = get_local_ordinal() if local else _emulation.ordinal()
    return ordinal == 0


def master_print(*args, fd=sys.stdout, local=False, flush=False):
    """``print`` on the master ordinal only."""
    if is_master_ordinal(local=local):
        print(*args, file=fd, flush=flush)


def xla_device(n=None, devkind=None):
    """Return ``torch.device('cuda', n)`` (or ``cpu``) as the XLA device.

    Real XLA returns ``xla:N``.  The emulation returns the Jittor CUDA device
    backing that ordinal; ``devkind`` is accepted for signature compatibility
    and ignored, as it is deprecated upstream.
    """
    del devkind
    return _emulation.make_device(n)


def xla_device_hw(device):
    """Hardware kind of a device: ``"GPU"`` or ``"CPU"``."""
    text = str(device).lower()
    if text.startswith("cuda") or text.startswith("gpu"):
        return "GPU"
    if text.startswith("cpu"):
        return "CPU"
    if text.startswith("xla"):
        return _emulation.device_type()
    raise ValueError("torch_xla emulation: unknown device %r" % (device,))


def xla_device_kind(device=None):
    """Vendor device kind of the emulated device (``cuda`` or ``cpu``)."""
    if device is None:
        device = xla_device()
    return "cuda" if xla_device_hw(device) == "GPU" else "cpu"


def set_replication(device, devices):
    """No-op: the emulation gives every process its own device already."""
    del device, devices
    return None


def mark_step(wait=False, reset_scope=True):
    """No-op: Jittor executes eagerly, so there is no lazy XLA graph to run.

    ``wait=True`` synchronizes the emulated device, which is the closest
    observable equivalent of the real "execute and wait" behaviour.
    """
    del reset_scope
    if wait:
        wait_device_ops()
    return None


def wait_device_ops(devices=[]):
    """Wait for pending device work; synchronizes CUDA when present."""
    del devices
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:  # pragma: no cover - partial shims have no streams
        pass
    return None


def get_rng_state(device=None):
    """Current Jittor host seed as an integer (approximate XLA RNG state)."""
    del device
    return int(torch.initial_seed())


def set_rng_state(seed, device=None):
    """Reseed the Jittor host generator from an integer (approximate)."""
    del device
    torch.manual_seed(int(seed))


def save(data, file_or_path, master_only=True, global_master=False):
    """Save ``data`` on the master ordinal with ``torch.save``.

    Tensors are copied to CPU first, matching the documented XLA contract that
    a later ``torch.load`` sees CPU data.  Unlike the real API this performs no
    implicit rendezvous; callers that need the documented ordering call
    ``rendezvous`` themselves, exactly as the upstream docstring shows.
    """
    should_write = not master_only or is_master_ordinal(local=not global_master)
    cpu_data = _emulation.to_cpu(data, convert=should_write)
    if should_write:
        torch.save(cpu_data, file_or_path)


def _maybe_convert_to_cpu(data, convert=True):
    """Copy tensors inside ``data`` to CPU (upstream private helper).

    Transformers' XLA checkpoint integration calls this before
    ``save_pretrained``; under the emulation every tensor lives on a Jittor
    CUDA/CPU device, so ``to_cpu`` is the direct equivalent.
    """
    return _emulation.to_cpu(data, convert=convert)


def send_cpu_data_to_device(datas, device, input_sharding=None):
    """Move CPU tensors inside ``datas`` to the emulated device.

    A bare tensor is wrapped in a list first, mirroring the upstream API's
    return shape.  ``input_sharding`` requests SPMD sharding and is rejected.
    """
    if input_sharding is not None:
        raise NotImplementedError("torch_xla emulation: input sharding (SPMD) is not implemented")
    if isinstance(datas, torch.Tensor):
        datas = [datas]
    return _emulation.to_device(datas, device)


def rendezvous(tag, payload=b"", replicas=[]):
    """Exchange one payload per rank; returns payloads in rank order.

    Implemented as atomic files under the launch's rendezvous directory.  The
    call is a barrier: it blocks until every rank has published its payload for
    this round of ``tag``.  Explicit replica subsets are not supported.
    """
    return _emulation.rendezvous(tag, payload, replicas)


def mesh_reduce(tag, data, reduce_fn):
    """Reduce ``data`` across ranks with ``reduce_fn``, via ``rendezvous``.

    Follows the upstream client protocol: CPU copies are serialized with
    ``torch.save``, exchanged through ``rendezvous``, deserialized, and passed
    to ``reduce_fn`` as one list per rank.  ``data`` must therefore be
    ``torch.save``-able.
    """
    cpu_data = _emulation.to_cpu(data)
    buffer = io.BytesIO()
    torch.save(cpu_data, buffer)
    exchanged = rendezvous(tag, buffer.getvalue())
    loaded = []
    for payload in exchanged:
        loaded.append(torch.load(io.BytesIO(payload)))
    return reduce_fn(loaded) if loaded else cpu_data


def _tensor_list(inputs):
    if isinstance(inputs, torch.Tensor):
        return [inputs]
    if isinstance(inputs, (list, tuple)):
        tensors = list(inputs)
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                raise TypeError("torch_xla emulation: all_reduce inputs must be tensors")
        return tensors
    raise TypeError(
        "torch_xla emulation: all_reduce expects a tensor or a flat list of "
        "tensors, got %r" % (type(inputs).__name__,)
    )


def _reduce_tensors(reduce_type, tensors):
    result = tensors[0]
    for other in tensors[1:]:
        if reduce_type == REDUCE_SUM:
            result = result + other
        elif reduce_type == REDUCE_MUL:
            result = result * other
        elif reduce_type == REDUCE_MAX:
            result = torch.where(other > result, other, result)
        elif reduce_type == REDUCE_MIN:
            result = torch.where(other < result, other, result)
        else:
            raise NotImplementedError(
                "torch_xla emulation: all_reduce(%r) is not implemented; "
                "supported reduce types are sum, mul, min and max" % (reduce_type,)
            )
    return result


def all_reduce(reduce_type, inputs, scale=1.0, groups=None, pin_layout=True):
    """In-place all-reduce of one tensor or a flat list of tensors.

    APPROXIMATE: values are exchanged as serialized CPU tensors through the
    filesystem rendezvous, reduced client-side, scaled, and copied back into
    the input tensors.  This matches the observable numerical contract for the
    ``sum``/``mul``/``min``/``max`` reductions used by Transformers and
    Accelerate, but it is not an XLA or NCCL collective and is much slower.
    """
    del pin_layout
    if groups is not None:
        raise NotImplementedError(
            "torch_xla emulation: all_reduce over explicit replica groups is not implemented"
        )
    if reduce_type in (REDUCE_AND, REDUCE_OR):
        raise NotImplementedError(
            "torch_xla emulation: logical all_reduce (%r) is not implemented" % (reduce_type,)
        )
    tensors = _tensor_list(inputs)
    if not tensors:
        return None

    cpu_tensors = [tensor.detach().cpu() for tensor in tensors]
    buffer = io.BytesIO()
    torch.save(cpu_tensors, buffer)
    exchanged = rendezvous(_ALL_REDUCE_TAG, buffer.getvalue())
    per_rank = [torch.load(io.BytesIO(payload)) for payload in exchanged]
    if len(per_rank) != 1:
        width = len(cpu_tensors)
        for rank_tensors in per_rank:
            if len(rank_tensors) != width:
                raise RuntimeError(
                    "torch_xla emulation: all_reduce ranks disagree on the "
                    "number of tensors (%d != %d)" % (len(rank_tensors), width)
                )
        reduced = [
            _reduce_tensors(reduce_type, [rank[index] for rank in per_rank])
            for index in range(width)
        ]
    else:
        reduced = cpu_tensors

    if scale != 1.0:
        reduced = [tensor * float(scale) for tensor in reduced]
    for target, value in zip(tensors, reduced):
        target.copy_(value.to(target.device))
    return None


def all_gather(value, dim=0, groups=None, pin_layout=True):
    """Concatenate ``value`` across ranks along ``dim`` (filesystem exchange)."""
    del pin_layout
    if groups is not None:
        raise NotImplementedError(
            "torch_xla emulation: all_gather over explicit replica groups is not implemented"
        )
    if isinstance(value, (list, tuple)):
        return type(value)(
            _all_gather_one(item, dim, "%s.%d" % (_ALL_GATHER_TAG, index))
            for index, item in enumerate(value)
        )
    return _all_gather_one(value, dim, _ALL_GATHER_TAG)


def _all_gather_one(value, dim, tag):
    if not isinstance(value, torch.Tensor):
        raise TypeError("torch_xla emulation: all_gather expects tensors")
    cpu_value = value.detach().cpu()
    buffer = io.BytesIO()
    torch.save(cpu_value, buffer)
    exchanged = rendezvous(tag, buffer.getvalue())
    parts = [torch.load(io.BytesIO(payload)) for payload in exchanged]
    if len(parts) == 1:
        return parts[0].to(value.device)
    return torch.cat(parts, dim=dim).to(value.device)


def _fetch_gradients(optimizer):
    """Gradient tensors held by ``optimizer``, in parameter-group order."""
    gradients = []
    for param_group in optimizer.param_groups:
        for parameter in param_group["params"]:
            gradient = getattr(parameter, "grad", None)
            if gradient is not None:
                gradients.append(getattr(gradient, "data", gradient))
    return gradients


def reduce_gradients(optimizer, groups=None, pin_layout=True):
    """All-reduce optimizer gradients with mean scaling across ranks.

    With one rank this is a no-op, matching the upstream implementation.
    """
    count = _emulation.world_size()
    if count > 1:
        all_reduce(
            REDUCE_SUM,
            _fetch_gradients(optimizer),
            scale=1.0 / count,
            groups=groups,
            pin_layout=pin_layout,
        )
    return None


def optimizer_step(optimizer, barrier=False, optimizer_args={}, groups=None, pin_layout=True):
    """Reduce gradients, then call ``optimizer.step(**optimizer_args)``."""
    reduce_gradients(optimizer, groups=groups, pin_layout=pin_layout)
    loss = optimizer.step(**(optimizer_args or {}))
    if barrier:
        wait_device_ops()
    return loss
