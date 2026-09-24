"""Native data-parallel wrapper for Jittor modules."""

import hashlib
import itertools

import jittor as jt
from jittor import Module


_DDP_SEQUENCE = itertools.count()
_SIGNATURE_WORDS = 16
_SIGNATURE_VARIANCE_TOLERANCE = 5e-2


class _ReducerState:
    def __init__(self, wrapper, group, world_size, parameters):
        self.wrapper = wrapper
        self.group = group
        self.world_size = world_size
        self.parameters = tuple(parameters)
        self.initial_signature = _module_signature(wrapper.module)
        self.sync_enabled = True
        self.has_unsynced_grads = False
        self.sequence = next(_DDP_SEQUENCE)


def _distributed_api():
    from jittor import distributed
    required = (
        "is_initialized", "get_world_size", "get_default_group",
        "all_reduce", "broadcast",
    )
    missing = [name for name in required if not callable(getattr(distributed, name, None))]
    if missing:
        raise RuntimeError(
            "native DistributedDataParallel requires jittor.distributed APIs: "
            + ", ".join(missing)
        )
    return distributed


def _group_size(distributed, group):
    if group is None:
        return int(distributed.get_world_size())
    size = getattr(group, "size", None)
    if not callable(size):
        raise TypeError("process_group must provide size()")
    return int(size())


def _module_signature(module):
    parameters = []
    for name, parameter in module.named_parameters():
        parameters.append((
            name,
            tuple(int(dim) for dim in parameter.shape),
            str(parameter.dtype),
            str(getattr(parameter, "placement_backend", "unknown")),
            bool(parameter.requires_grad),
        ))
    buffers = []
    for name, buffer in module.named_buffers():
        buffers.append((
            name,
            tuple(int(dim) for dim in buffer.shape),
            str(buffer.dtype),
            str(getattr(buffer, "placement_backend", "unknown")),
        ))
    return tuple(parameters), tuple(buffers)


def _module_placements(parameters, buffers):
    placements = {
        (getattr(value, "placement_backend", "unknown"),
         int(getattr(value, "device_id", 0)))
        for _, value in list(parameters) + list(buffers)
    }
    if len(placements) > 1:
        raise ValueError(
            "native DistributedDataParallel requires all module parameters and "
            "buffers on one device per process"
        )
    return next(iter(placements), None)


def _signature_check(distributed, group, payload, description, reference):
    """Reject mismatched rank metadata before issuing per-parameter collectives."""
    digest = hashlib.sha256(repr(payload).encode("utf-8")).digest()
    words = []
    for byte in digest[:8]:
        words.extend(((byte & 15) * 16, (byte >> 4) * 16))
    values = jt.array(
        words + [word * word for word in words], dtype="float32"
    ).stop_grad()
    backend = getattr(reference, "placement_backend", None)
    device_id = int(getattr(reference, "device_id", 0))
    if backend == "cuda":
        values = values.cuda(device_id)
    elif backend == "npu":
        values = values.npu(device_id)
    reduced = distributed.all_reduce(values, op="mean", group=group)
    observed = reduced.numpy().reshape(-1)
    for index in range(_SIGNATURE_WORDS):
        mean = float(observed[index])
        variance = float(observed[index + _SIGNATURE_WORDS]) - mean * mean
        if variance > _SIGNATURE_VARIANCE_TOLERANCE:
            raise RuntimeError(
                "DistributedDataParallel ranks have different " + description
            )


def _collective_result(result, original):
    return original if result is None else result


def _metadata_reference(parameters, buffers):
    if parameters:
        return parameters[0][1]
    if buffers:
        return buffers[0][1]
    raise ValueError(
        "multi-rank DistributedDataParallel requires a parameter or buffer "
        "to select its collective device"
    )


def _chain_dependency(value, dependency):
    producer = value._input(0)
    producer._add_dependency(dependency)
    return [producer]


class DistributedDataParallel(Module):
    """Synchronize a native Jittor module across data-parallel ranks.

    The first native implementation uses Jittor's ``optimizer.backward(loss)``
    training contract. Parameters and buffers are broadcast when the wrapper is
    constructed. Gradient buffers are averaged before ``optimizer.step`` reads
    them. ``no_sync`` leaves local accumulation untouched until a later synced
    backward.

    This wrapper uses one device per process. ``device_ids`` may be omitted or
    contain the single local device id; it does not move the wrapped module.
    ``output_device`` is accepted as a Torch-style alias for that same device.
    """

    def __init__(self, module, device_ids=None, output_device=None,
                 process_group=None, broadcast_buffers=True):
        if not isinstance(module, Module):
            raise TypeError("module must be a native jittor.nn.Module")
        if device_ids is not None:
            if isinstance(device_ids, int):
                device_ids = [device_ids]
            else:
                device_ids = list(device_ids)
            if len(device_ids) != 1:
                raise ValueError(
                    "native DistributedDataParallel supports one device per process"
                )
        if output_device is not None:
            if device_ids is None or int(output_device) != int(device_ids[0]):
                raise ValueError(
                    "output_device must match the single device in device_ids"
                )

        super().__init__()
        self.module = module
        self.device_ids = device_ids
        self.output_device = output_device
        self.process_group = process_group

        distributed = _distributed_api()
        if process_group is None:
            process_group = distributed.get_default_group()
        world_size = _group_size(distributed, process_group)
        if world_size < 1:
            raise ValueError("process_group must contain at least one rank")
        if world_size > 1 and not distributed.is_initialized():
            raise RuntimeError(
                "initialize jittor.distributed before constructing a multi-rank DDP"
            )
        self.process_group = process_group

        parameters = list(module.named_parameters())
        buffers = list(module.named_buffers())
        placement = _module_placements(parameters, buffers)
        metadata_reference = (
            _metadata_reference(parameters, buffers) if world_size > 1 else None
        )
        if device_ids is not None:
            if placement is None or placement[0] not in ("cuda", "npu"):
                raise ValueError(
                    "device_ids requires the wrapped module to be on an accelerator"
                )
            if int(device_ids[0]) != placement[1]:
                raise ValueError(
                    "device_ids must match the wrapped module's device id"
                )
        state = _ReducerState(self, process_group, world_size, parameters)
        for _, parameter in parameters:
            existing = getattr(parameter, "_jittor_ddp_state", None)
            if existing is not None:
                raise ValueError(
                    "a parameter cannot be registered with multiple native DDP wrappers"
                )

        if world_size > 1:
            _signature_check(
                distributed, process_group, state.initial_signature,
                "module parameter/buffer metadata",
                reference=metadata_reference,
            )
            ranks = getattr(process_group, "ranks", None)
            src = int(ranks[0]) if ranks else 0
            dependency = []
            for _, parameter in parameters:
                result = distributed.broadcast(parameter, src=src, group=process_group)
                result = _collective_result(result, parameter)
                if result is not parameter:
                    parameter.assign(result)
                    dependency = _chain_dependency(parameter, dependency)
            if broadcast_buffers:
                for _, buffer in buffers:
                    result = distributed.broadcast(buffer, src=src, group=process_group)
                    result = _collective_result(result, buffer)
                    if result is not buffer:
                        buffer.assign(result)
                        dependency = _chain_dependency(buffer, dependency)
            jt.sync_all()

        for _, parameter in parameters:
            object.__setattr__(parameter, "_jittor_ddp_state", state)
        object.__setattr__(self, "_jittor_ddp_state", state)

    def execute(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def buffers(self, recurse=True):
        return [buffer for _, buffer in self.named_buffers(recurse=recurse)]

    def no_sync(self):
        return _NoSync(self)


class _NoSync:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.previous = None

    def __enter__(self):
        state = self.wrapper._jittor_ddp_state
        self.previous = state.sync_enabled
        state.sync_enabled = False
        return self.wrapper

    def __exit__(self, exc_type, exc_value, traceback):
        self.wrapper._jittor_ddp_state.sync_enabled = self.previous
        return False


def _sync_optimizer_gradients(entries):
    """Reduce DDP-managed optimizer buffers once, in module parameter order.

    ``entries`` contains ``(parameter, accumulated_gradient)`` pairs after the
    optimizer has added the current local gradient. Unmanaged parameters are
    returned to the optimizer so its legacy MPI path can continue handling them.
    """
    grouped = {}
    owned_parameter_ids = set()
    for parameter, gradient in entries:
        state = getattr(parameter, "_jittor_ddp_state", None)
        if state is None:
            continue
        owned_parameter_ids.add(id(parameter))
        group = grouped.setdefault(state, {})
        group.setdefault(id(parameter), []).append(gradient)

    if not grouped:
        return owned_parameter_ids, []

    distributed = _distributed_api()
    dependencies = []
    for state in sorted(grouped, key=lambda item: item.sequence):
        by_parameter = grouped[state]
        if state.world_size <= 1:
            current = _module_signature(state.wrapper.module)
            if current != state.initial_signature:
                raise RuntimeError(
                    "DistributedDataParallel module parameters or buffers changed "
                    "after construction"
                )
            continue
        if not state.sync_enabled:
            state.has_unsynced_grads = True
            continue

        current_signature = _module_signature(state.wrapper.module)
        coverage = tuple(
            len(by_parameter.get(id(parameter), ()))
            if parameter.requires_grad else 0
            for _, parameter in state.parameters
        )
        _signature_check(
            distributed, state.group,
            (current_signature, coverage),
            "DDP module structure or optimizer parameter coverage",
            reference=_metadata_reference(
                list(state.wrapper.module.named_parameters()),
                list(state.wrapper.module.named_buffers()),
            ),
        )
        if current_signature != state.initial_signature:
            raise RuntimeError(
                "DistributedDataParallel module parameters or buffers changed "
                "after construction"
            )
        for (name, parameter), count in zip(state.parameters, coverage):
            if parameter.requires_grad and count == 0:
                raise RuntimeError(
                    "trainable DDP parameter {!r} is missing from the optimizer".format(name)
                )
            if count > 1:
                raise RuntimeError(
                    "DDP parameter {!r} appears more than once in the optimizer".format(name)
                )

        for _, parameter in state.parameters:
            if not parameter.requires_grad:
                continue
            gradient = by_parameter[id(parameter)][0]
            reduced = distributed.all_reduce(
                gradient, op="mean", group=state.group
            )
            if reduced is not gradient:
                gradient.assign(reduced)
                dependencies = _chain_dependency(gradient, dependencies)
        state.has_unsynced_grads = False

    return owned_parameter_ids, dependencies


def _assert_ddp_step_ready(optimizer):
    """Refuse an optimizer update if DDP gradients were only accumulated locally."""
    checked = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = getattr(parameter, "_jittor_ddp_state", None)
            if state is None or id(state) in checked:
                continue
            checked.add(id(state))
            if state.world_size > 1 and state.has_unsynced_grads:
                raise RuntimeError(
                    "DistributedDataParallel gradients are still local; run a synced "
                    "optimizer.backward(loss) before optimizer.step()"
                )


__all__ = ["DistributedDataParallel"]
