"""Native data-parallel wrapper for Jittor modules."""

import hashlib
import itertools

import jittor as jt
from jittor import Module


_DDP_SEQUENCE = itertools.count()
_SIGNATURE_WORDS = 16
_SIGNATURE_TOLERANCE = 5e-2
_BUCKET_BYTES = 25 * 1024 * 1024
_BUCKET_INPUTS = 64


class _ReducerState:
    def __init__(self, wrapper, group, world_size, parameters):
        self.wrapper = wrapper
        self.group = group
        self.world_size = world_size
        self.parameters = tuple(parameters)
        self.initial_signature = _module_signature(wrapper.module)
        self.parameter_ids = tuple(id(parameter) for _, parameter in parameters)
        self.local_placements = _local_placements(
            parameters, wrapper.module.named_buffers())
        self.buckets = _gradient_buckets(parameters)
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


def _module_signature(module, parameters=None, buffers=None):
    if parameters is None:
        parameters = module.named_parameters()
    if buffers is None:
        buffers = module.named_buffers()
    parameter_metadata = []
    for name, parameter in parameters:
        parameter_metadata.append((
            name,
            tuple(int(dim) for dim in parameter.shape),
            str(parameter.dtype),
            str(getattr(parameter, "placement_backend", "unknown")),
            bool(parameter.requires_grad),
        ))
    buffer_metadata = []
    for name, buffer in buffers:
        buffer_metadata.append((
            name,
            tuple(int(dim) for dim in buffer.shape),
            str(buffer.dtype),
            str(getattr(buffer, "placement_backend", "unknown")),
        ))
    return tuple(parameter_metadata), tuple(buffer_metadata)


def _value_placement(value):
    # Native Vars expose an integer backend id; -1 follows the runtime.
    # Residency may temporarily be CPU after numpy(); dispatch resolves the
    # graph's device without mistaking that host read for a model migration.
    from jittor._runtime.dispatch import dispatch_context
    backend = dispatch_context(value).backend
    device_id = int(getattr(value, "device_id", 0))
    if backend != "cpu" and device_id < 0:
        device_id = int(jt.current_device())
    return backend, device_id


def _local_placements(parameters, buffers):
    return tuple((getattr(value, "placement_backend", None),
                  int(getattr(value, "device_id", 0)))
                 for _, value in list(parameters) + list(buffers))


def _module_placements(parameters, buffers):
    placements = {_value_placement(value)
                  for _, value in list(parameters) + list(buffers)}
    if len(placements) > 1:
        raise ValueError(
            "native DistributedDataParallel requires all module parameters and "
            "buffers on one device per process"
        )
    return next(iter(placements), None)


def _validate_storage_layout(parameters, buffers):
    """Reject storage views that NCCL broadcast cannot safely flatten.

    Gradient buckets are packed before reduction, but construction broadcasts
    each parameter and buffer directly. NCCL treats those addresses as dense;
    expanded/strided storage can therefore read beyond its allocation or
    overwrite aliased elements. Until the constructor path packs these values,
    fail before issuing any collective.
    """
    for name, value in list(parameters) + list(buffers):
        is_contiguous = getattr(value, "_storage_is_contiguous", None)
        if callable(is_contiguous) and not is_contiguous():
            raise ValueError(
                "native DistributedDataParallel requires contiguous parameter "
                "and buffer storage; {!r} is a strided or overlapping view"
                .format(name)
            )


def _gradient_buckets(parameters):
    """Bounded homogeneous buckets in the agreed module parameter order."""
    buckets = []
    current = []
    current_key = None
    current_bytes = 0
    for index, (_, parameter) in enumerate(parameters):
        if not parameter.requires_grad:
            continue
        key = (str(parameter.dtype),
               str(getattr(parameter, "placement_backend", "unknown")),
               int(getattr(parameter, "device_id", 0)))
        size = int(parameter.numel()) * int(parameter.dtype.dsize())
        if current and (key != current_key or current_bytes + size > _BUCKET_BYTES
                        or len(current) >= _BUCKET_INPUTS):
            buckets.append(tuple(current))
            current = []
            current_bytes = 0
        current.append(index)
        current_key = key
        current_bytes += size
    if current:
        buckets.append(tuple(current))
    return tuple(buckets)


class _GradientBucketReadiness:
    """Drain ready gradients in a rank-stable reducer bucket order."""

    def __init__(self, buckets, target_to_parameter, on_bucket_ready):
        self.buckets = tuple(tuple(bucket) for bucket in buckets)
        self.target_to_parameter = tuple(target_to_parameter)
        self.on_bucket_ready = on_bucket_ready
        self.seen_targets = set()
        self.ready = {}
        self.next_bucket = 0

        bucket_parameters = [
            parameter_index
            for bucket in self.buckets
            for parameter_index in bucket
        ]
        if len(bucket_parameters) != len(set(bucket_parameters)):
            raise ValueError("gradient buckets contain duplicate parameters")
        if set(bucket_parameters) != set(self.target_to_parameter):
            raise ValueError(
                "gradient bucket parameters must match backward targets"
            )
        if len(set(self.target_to_parameter)) != len(self.target_to_parameter):
            raise ValueError("backward targets contain duplicate parameters")

    def __call__(self, target_index, gradient):
        target_index = int(target_index)
        if target_index < 0 or target_index >= len(self.target_to_parameter):
            raise IndexError("gradient callback target index is out of range")
        if target_index in self.seen_targets:
            raise RuntimeError(
                "gradient callback target {} was reported more than once"
                .format(target_index)
            )
        self.seen_targets.add(target_index)
        self.ready[self.target_to_parameter[target_index]] = gradient

        while self.next_bucket < len(self.buckets):
            bucket = self.buckets[self.next_bucket]
            if not all(index in self.ready for index in bucket):
                break
            gradients = tuple(self.ready.pop(index) for index in bucket)
            self.on_bucket_ready(bucket, gradients)
            self.next_bucket += 1

    def finish(self):
        missing = sorted(
            set(range(len(self.target_to_parameter))) - self.seen_targets
        )
        if missing:
            raise RuntimeError(
                "gradient callbacks missing target indices {}".format(missing)
            )
        if self.next_bucket != len(self.buckets):
            raise RuntimeError("gradient callbacks did not complete every bucket")


def _signature_check(distributed, group, payload, description, reference):
    """Reject mismatched rank metadata before issuing per-parameter collectives."""
    digest = hashlib.sha256(repr(payload).encode("utf-8")).digest()
    words = []
    for byte in digest[:8]:
        words.extend((byte & 15, byte >> 4))
    values = jt.array(words, dtype="float32").stop_grad()
    backend, device_id = _value_placement(reference)
    if backend == "cuda":
        values = values.cuda(device_id)
    elif backend == "npu":
        values = values.npu(device_id)
    reduced = distributed.all_reduce(values, op="mean", group=group)
    observed = reduced.numpy().reshape(-1)
    for index in range(_SIGNATURE_WORDS):
        expected = float(words[index])
        if abs(float(observed[index]) - expected) > _SIGNATURE_TOLERANCE:
            raise RuntimeError(
                "DistributedDataParallel could not verify the reduced "
                + description + " signature"
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


def _validate_reducer_state(state):
    current_parameters = list(state.wrapper.module.named_parameters())
    current_buffers = list(state.wrapper.module.named_buffers())
    current_signature = _module_signature(
        state.wrapper.module, current_parameters, current_buffers)
    if (current_signature != state.initial_signature
            or tuple(id(parameter) for _, parameter in current_parameters)
            != state.parameter_ids
            or _local_placements(current_parameters, current_buffers)
            != state.local_placements):
        raise RuntimeError(
            "DistributedDataParallel module parameters or buffers changed "
            "after construction"
        )


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
        _validate_storage_layout(parameters, buffers)
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
            # Materializing the metadata marker on the communication stream
            # must follow any asynchronous model initialization work.  A
            # pending CUDA update can otherwise leave NCCL's first P2P
            # collective ordered against an unfinished default-stream
            # operation; SHM fallback happens to hide that ordering hazard.
            jt.sync_all(True)
            _signature_check(
                distributed, process_group, state.initial_signature,
                "module parameter/buffer metadata",
                reference=metadata_reference,
            )
            ranks = getattr(process_group, "ranks", None)
            src = int(ranks[0]) if ranks else 0
            for _, parameter in parameters:
                result = distributed.broadcast(parameter, src=src, group=process_group)
                result = _collective_result(result, parameter)
                if result is not parameter:
                    parameter.assign(result)
            if broadcast_buffers:
                for _, buffer in buffers:
                    result = distributed.broadcast(buffer, src=src, group=process_group)
                    result = _collective_result(result, buffer)
                    if result is not buffer:
                        buffer.assign(result)
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


def _sync_optimizer_gradients(entries, optimizer=None):
    """Reduce DDP-managed optimizer buffers once, in module parameter order.

    ``entries`` contains ``(parameter, accumulated_gradient)`` pairs after the
    optimizer has added the current local gradient. Unmanaged parameters are
    returned to the optimizer so its legacy MPI path can continue handling them.
    """
    known_states = getattr(optimizer, "_jittor_ddp_states", {})
    grouped = {state: {} for state in known_states}
    owned_parameter_ids = set()
    for parameter, gradient in entries:
        state = getattr(parameter, "_jittor_ddp_state", None)
        if state is None:
            continue
        owned_parameter_ids.add(id(parameter))
        if optimizer is not None:
            known_states[state] = None
        group = grouped.setdefault(state, {})
        group.setdefault(id(parameter), []).append(gradient)

    if not grouped:
        return owned_parameter_ids, []

    distributed = _distributed_api()
    dependencies = []
    for state in sorted(grouped, key=lambda item: item.sequence):
        by_parameter = grouped[state]
        # Construction checks the complete schema across ranks. Every backward
        # then checks that local schema and parameter holders still match that
        # agreed schema; a valid step cannot change the collective sequence.
        # This avoids a device-to-host metadata round trip on every update.
        # As with a rank's forward exception, the launcher must stop peers when
        # a local contract violation raises before their gradient collective.
        _validate_reducer_state(state)
        coverage = tuple(
            len(by_parameter.get(id(parameter), ()))
            if parameter.requires_grad else 0
            for _, parameter in state.parameters
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
        if not state.sync_enabled:
            state.has_unsynced_grads = True
            continue
        if state.world_size <= 1:
            continue

        gradients = [
            by_parameter[id(parameter)][0]
            for _, parameter in state.parameters
            if parameter.requires_grad
        ]
        # Only raw sum outputs may be executed while the NCCL group is open.
        # Scaling and unpacking consume the output after group end and join.
        # WORLD uses the runtime backend and has no private subgroup kind.
        backend_query = getattr(state.group, "_get_backend_name", None)
        backend = (backend_query() if callable(backend_query)
                   else getattr(state.group, "_backend_kind", None))
        use_bucket = backend == "nccl"
        if use_bucket and gradients:
            from jittor.distributed.bucket import bucket_scope, comm_wait
            from jittor.backends.cuda.kernels.misc.ddp_bucket import (
                pack_gradients, unpack_gradients,
            )
            gradient_map = {
                index: by_parameter[id(parameter)][0]
                for index, (_, parameter) in enumerate(state.parameters)
                if parameter.requires_grad
            }
            reduced_pairs = []
            # The optimizer supplies complete backward results here. Reverse
            # bucket order and leave every communication-stream join deferred
            # until the final wait. This preserves the agreed collective
            # sequence while allowing the communication stream to progress
            # across bucket boundaries. Device profiling still has to
            # establish backward/communication overlap separately.
            try:
                for indices in reversed(state.buckets):
                    members = [gradient_map[index] for index in indices]
                    value = (pack_gradients(members)
                             if len(members) > 1 else members[0])
                    jt.sync(members, weak_sync=False)
                    with bucket_scope(defer_join=True):
                        reduced = distributed.all_reduce(
                            value, op="sum", group=state.group
                        )
                        jt.sync([reduced], weak_sync=False)
                    # nccl_hold keeps the input allocation reserved until the
                    # final join, so retaining the packed Var here is not
                    # needed. The averaged result is consumed after comm_wait.
                    reduced_pairs.append((members, reduced))
            finally:
                comm_wait()
            for members, reduced in reduced_pairs:
                if len(members) == 1:
                    members[0].assign(reduced / state.world_size)
                    continue
                for gradient, averaged in zip(
                        members, unpack_gradients(reduced, members, state.world_size)):
                    gradient.assign(averaged)
        else:
            for gradient in gradients:
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
    states = dict(getattr(optimizer, "_jittor_ddp_states", {}))
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = getattr(parameter, "_jittor_ddp_state", None)
            if state is not None:
                states[state] = None
    for state in states:
        if state.world_size > 1 and state.has_unsynced_grads:
            raise RuntimeError(
                "DistributedDataParallel gradients are still local; run a synced "
                "optimizer.backward(loss) before optimizer.step()"
            )


__all__ = ["DistributedDataParallel"]
