"""Native tensor-sharded checkpoint I/O and model/optimizer/RNG continuation.

All ranks call ``save`` in the same order with a shared checkpoint directory.
For multiple ranks, initialize a native process group with a Store first. This
module does not shard training parameters, gradients, or optimizer arithmetic.
"""

import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import uuid

import numpy as np
import jittor as jt

from jittor.serialization.sharded import (
    CheckpointError, FORMAT, VERSION, RankLocal, TensorShard,
    json_bytes, read_manifest, restore_tree, validate_payloads,
    validate_records, write_rank_state,
)


__all__ = ["CheckpointError", "RankLocal", "TensorShard", "save", "load",
           "save_training_state", "load_training_state"]

_SEQUENCES = {}


def _context(group=None, store=None):
    from jittor import distributed as dist

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if rank < 0:
        raise ValueError("checkpoint caller is not a process-group member")
    ranks = [dist.get_global_rank(group, i) if group is not None else i
             for i in range(world)]
    selected = store if store is not None else dist.get_default_store()
    if world > 1 and selected is None:
        raise ValueError("multi-rank checkpoint requires a native process-group Store")
    return rank, world, ranks, selected


def _exchange(store, key, rank, world, value, timeout):
    if world == 1:
        return [value]
    store.set(key + "/" + str(rank), json_bytes(value))
    keys = [key + "/" + str(i) for i in range(world)]
    store.wait(keys, timeout=timeout)
    return [json.loads(store.get(name).decode("utf8")) for name in keys]


def _get_json(store, key, timeout):
    # TCPStore/PrefixStore expose get(key), while wait has the per-operation
    # timeout contract shared by all native Store implementations.
    store.wait([key], timeout=timeout)
    return json.loads(store.get(key).decode("utf8"))


def save(state, checkpoint_id, *, process_group=None, store=None,
         overwrite=False, timeout=60):
    """Write rank-local payload files and atomically publish a complete manifest.

    Plain tensors are replicated input and are checked for equality before
    their files are partitioned. ``TensorShard`` supplies genuinely local
    input. ``RankLocal`` is retained separately for each global rank.
    A failed overwrite leaves the previous complete generation intact.
    """
    if type(overwrite) is not bool or timeout <= 0:
        raise ValueError("overwrite must be bool and timeout must be positive")
    rank, world, ranks, selected = _context(process_group, store)
    directory = Path(checkpoint_id).resolve()
    identity = (str(directory), tuple(ranks))
    sequence = next(_SEQUENCES.setdefault(identity, itertools.count()))
    prefix = "checkpoint/{}/{}/".format(
        hashlib.sha256(json_bytes(identity)).hexdigest(), sequence)
    announcement = None
    preparation_error = None
    if rank == 0:
        try:
            if (directory / "manifest.json").exists() and not overwrite:
                raise FileExistsError("checkpoint already exists: " + str(directory))
            directory.mkdir(parents=True, exist_ok=True)
            generation = uuid.uuid4().hex
            (directory / generation).mkdir()
            announcement = {"generation": generation, "overwrite": overwrite}
        except (OSError, ValueError) as error:
            preparation_error = error
            announcement = {"error": "{}: {}".format(type(error).__name__, error)}
        if world > 1:
            selected.set(prefix + "generation", json_bytes(announcement))
    elif world > 1:
        announcement = _get_json(selected, prefix + "generation", timeout)
    if "error" in announcement:
        raise CheckpointError("checkpoint preparation failed: " + announcement["error"]) from preparation_error
    generation = directory / announcement["generation"]
    error = None
    try:
        if announcement["overwrite"] != overwrite:
            raise ValueError("checkpoint overwrite options differ across ranks")
        jt.sync_all(True)
        record = write_rank_state(state, generation / "rank-{:05d}".format(rank),
                                  rank, world, ranks[rank])
        status = {"record": record}
    except Exception as failure:
        # An I/O protocol must notify peers even when one rank fails. Preserve
        # the exception as the cause; never claim completion or fall back.
        error = failure
        status = {"error": "{}: {}".format(type(failure).__name__, failure)}
    statuses = _exchange(selected, prefix + "written", rank, world, status, timeout)
    failures = {i: value["error"] for i, value in enumerate(statuses) if "error" in value}
    if failures:
        raise CheckpointError("checkpoint rank writes failed: " + repr(failures)) from error
    result = None
    commit_error = None
    if rank == 0:
        try:
            records = [value["record"] for value in statuses]
            validate_records(records, ranks)
            validate_payloads(generation, records)
            manifest = {"format": FORMAT, "version": VERSION,
                        "generation": announcement["generation"],
                        "world_size": world, "ranks": ranks, "records": records}
            envelope = {"manifest": manifest,
                        "sha256": hashlib.sha256(json_bytes(manifest)).hexdigest()}
            temporary = directory / ("manifest-" + announcement["generation"] + ".tmp")
            with temporary.open("wb") as output:
                output.write(json_bytes(envelope))
                output.flush()
                os.fsync(output.fileno())
            os.replace(str(temporary), str(directory / "manifest.json"))
            result = {"complete": True}
        except Exception as failure:
            commit_error = failure
            result = {"error": "{}: {}".format(type(failure).__name__, failure)}
        if world > 1:
            selected.set(prefix + "committed", json_bytes(result))
    elif world > 1:
        result = _get_json(selected, prefix + "committed", timeout)
    if "error" in result:
        raise CheckpointError("checkpoint commit failed: " + result["error"]) from commit_error
    return str(directory / "manifest.json")


def load(checkpoint_id, *, process_group=None, tensor_shards=None):
    """Validate every payload, then return a reconstructed native state tree.

    ``tensor_shards`` maps JSON-pointer tensor paths to ``TensorShard`` target
    intervals. The returned leaves are descriptors holding those local values.
    Without targets, global tensor leaves are fully reconstructed. Rank-local
    values follow the caller's global rank; arbitrary world-size RNG migration
    is intentionally unsupported.
    """
    rank, _, ranks, _ = _context(process_group)
    manifest = read_manifest(checkpoint_id)
    return restore_tree(checkpoint_id, manifest, ranks[rank], tensor_shards)


def _parameter_names(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        if any(id(parameter) not in names for parameter in group["params"]):
            raise ValueError("optimizer has parameters not owned by the checkpoint model")
        groups.append([names[id(parameter)] for parameter in group["params"]])
    return groups


def _optimizer_state(model, optimizer):
    source = optimizer.state_dict()
    if not isinstance(source, dict) or set(source) != {"defaults"}:
        raise NotImplementedError("training checkpoint requires the native Optimizer state_dict protocol")
    defaults = dict(source["defaults"])
    defaults["param_groups"] = [dict((key, value) for key, value in group.items() if key != "params")
                               for group in defaults["param_groups"]]
    return {"class": type(optimizer).__module__ + "." + type(optimizer).__qualname__,
            "param_names": _parameter_names(model, optimizer), "state": {"defaults": defaults}}


def _rng_snapshot():
    jt.sync_all(True)
    result = {"python": random.getstate(), "numpy": np.random.get_state(),
              "cpu": jt.core.rng_state("cpu"), "cuda": []}
    if jt.flags.use_cuda:
        backend = jt.compile_extern.get_library("curand", load=True)
        seed = int(backend.curand_generator_seed())
        result["cuda"] = [(seed, int(backend.curand_generator_offset(device)))
                          for device in range(int(jt.get_device_count()))]
    return result


def _restore_rng(state):
    # Native setters validate their opaque payloads and preserve independent
    # CPU/device streams. The public helper snapshots current state to roll
    # back RNG on a rejected field before model/optimizer mutation.
    if bool(state["cuda"]) != bool(jt.flags.use_cuda):
        raise ValueError("checkpoint RNG execution backend mismatch")
    jt.core.set_rng_state("cpu", state["cpu"])
    if state["cuda"]:
        if not jt.flags.use_cuda or len(state["cuda"]) != int(jt.get_device_count()):
            raise ValueError("checkpoint CUDA RNG device count/backend mismatch")
        backend = jt.compile_extern.get_library("curand")
        for device, (seed, offset) in enumerate(state["cuda"]):
            backend.curand_restore_state(device, seed, offset)
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    if isinstance(numpy_state[1], jt.Var):
        numpy_state = (numpy_state[0], numpy_state[1].numpy()) + tuple(numpy_state[2:])
    np.random.set_state(numpy_state)


def _tensor_parallel_layout(model):
    layout = {}
    for name, parameter in model.named_parameters():
        if getattr(parameter, "_jittor_tp_spec", None) is not None:
            from .tensor_parallel import _validate_optimizer_parameter
            _validate_optimizer_parameter(parameter)
            layout[name] = dict(parameter._jittor_tp_spec)
    return layout


def save_training_state(model, optimizer, checkpoint_id, *, extra=None,
                        process_group=None, store=None, overwrite=False, timeout=60):
    """Save native owners and each rank's RNG at a completed-step boundary.

    Pending gradient accumulation is refused because the native optimizer's
    private backward latch is not represented by its public state_dict.
    """
    if getattr(optimizer, "_Optimizer__zero_grad", None) is not True:
        raise NotImplementedError("native training checkpoint requires a completed optimizer step or zero_grad boundary")
    model_state = model.state_dict()
    state = {"training_version": 1, "model": model_state,
             "model_stop_grad": {name: bool(value.is_stop_grad())
                                 for name, value in model_state.items()},
             "optimizer": _optimizer_state(model, optimizer),
             "rng": RankLocal(_rng_snapshot()), "extra": extra}
    layout = _tensor_parallel_layout(model)
    if layout:
        # Only genuinely sharded leaves are rank-local. Replicated row bias,
        # ordinary model parameters and optimizer metadata must still agree.
        state["tensor_parallel"] = {
            name: dict(spec, rank=RankLocal(spec["rank"]))
            for name, spec in layout.items()}
        sharded = {name for name, spec in layout.items() if spec["axis"] is not None}
        for name in sharded:
            state["model"][name] = RankLocal(state["model"][name])
            state["model_stop_grad"][name] = RankLocal(state["model_stop_grad"][name])
        saved_optimizer = state["optimizer"]
        groups = saved_optimizer["state"]["defaults"]["param_groups"]
        for names, group in zip(saved_optimizer["param_names"], groups):
            for field, values in list(group.items()):
                tensor_list = isinstance(values, (list, tuple)) and any(
                    isinstance(value, jt.Var) for value in values)
                if field == "grads" or tensor_list or field == "momentum_initialized":
                    if not isinstance(values, (list, tuple)) or len(values) != len(names):
                        raise ValueError("native checkpoint optimizer tensor-list length mismatch: " + field)
                    group[field] = type(values)(RankLocal(value) if name in sharded else value
                                                for name, value in zip(names, values))
    from .fsdp.checkpoint import _encode
    _encode(state, model)
    return save(state, checkpoint_id, process_group=process_group, store=store,
                overwrite=overwrite, timeout=timeout)


def load_training_state(model, optimizer, checkpoint_id, *, process_group=None):
    """Restore into existing native owners with FQN/shape/dtype validation.

    Optimizer type, parameter group order and parameter order must match. The
    native optimizer owns how moments/defaults are installed; no optimizer
    update mathematics or compatible frontend implementation is copied here.
    """
    manifest = read_manifest(checkpoint_id)
    rank, world, ranks, _ = _context(process_group)
    if world != manifest["world_size"] or set(ranks) != set(manifest["ranks"]):
        raise ValueError("training checkpoint rank/world mismatch; RNG resharding is unsupported")
    from .fsdp.checkpoint import _requests, _decode, _restore_trainability
    requests = _requests(model, optimizer, manifest, rank, _parameter_names(model, optimizer))
    state = restore_tree(checkpoint_id, manifest, ranks[rank], tensor_shards=requests)
    if state["training_version"] != 1:
        raise CheckpointError("unsupported native training checkpoint version")
    if state.get("tensor_parallel", {}) != _tensor_parallel_layout(model):
        raise ValueError("native checkpoint tensor-parallel ownership/layout mismatch")
    saved_optimizer = state["optimizer"]
    expected_class = type(optimizer).__module__ + "." + type(optimizer).__qualname__
    if saved_optimizer["class"] != expected_class:
        raise ValueError("native checkpoint optimizer type mismatch")
    if saved_optimizer["param_names"] != _parameter_names(model, optimizer):
        raise ValueError("native checkpoint optimizer FQN/group order mismatch")
    if set(model.state_dict()) != set(state["model"]):
        raise ValueError("native checkpoint model FQN mismatch")
    _decode(state, model)
    saved_groups = saved_optimizer["state"]["defaults"]["param_groups"]
    if len(saved_groups) != len(optimizer.param_groups):
        raise ValueError("native checkpoint optimizer group count mismatch")
    for saved_group, target_group in zip(saved_groups, optimizer.param_groups):
        parameters = target_group["params"]
        for field, values in saved_group.items():
            template = target_group.get(field)
            tensor_list = isinstance(template, (list, tuple)) and any(isinstance(v, jt.Var) for v in template)
            if field == "grads" or tensor_list:
                if not isinstance(values, (list, tuple)) or len(values) != len(parameters):
                    raise ValueError("native checkpoint optimizer tensor-list length mismatch: " + field)
                for index, (value, parameter) in enumerate(zip(values, parameters)):
                    reference = template[index] if tensor_list else parameter
                    if (not isinstance(value, jt.Var) or tuple(value.shape) != tuple(reference.shape)
                            or field != "grads" and str(value.dtype) != str(reference.dtype)):
                        raise ValueError("native checkpoint optimizer state shape/dtype mismatch: " + field)
    current = model.state_dict()
    if set(current) != set(state["model"]):
        raise ValueError("native checkpoint model FQN mismatch")
    if set(current) != set(state["model_stop_grad"]) or any(
            type(flag) is not bool for flag in state["model_stop_grad"].values()):
        raise ValueError("native checkpoint model trainability metadata mismatch")
    for name, target in current.items():
        source = state["model"][name]
        if tuple(target.shape) != tuple(source.shape) or str(target.dtype) != str(source.dtype):
            raise ValueError("native checkpoint model shape/dtype mismatch: " + name)
        # Native Module.update adopts its source's gradient flag. Generic I/O
        # returns detached values, so reapply saved trainability before using
        # that existing owner, rather than silently freezing every parameter.
        if state["model_stop_grad"][name]:
            source.stop_grad()
        else:
            source.start_grad()
    validate_optimizer = getattr(optimizer, "validate_state_dict", None)
    if validate_optimizer is not None:
        validate_optimizer(saved_optimizer["state"])
    previous_rng = _rng_snapshot()
    try:
        _restore_rng(state["rng"])
    except Exception:
        _restore_rng(previous_rng)
        raise
    model.load_state_dict(state["model"])
    # Updating an already-frozen target does not restart its gradient flag.
    # Restore both directions explicitly on the existing parameter objects.
    for name, target in current.items():
        target.stop_grad() if state["model_stop_grad"][name] else target.start_grad()
    _restore_trainability(model)
    optimizer.load_state_dict(saved_optimizer["state"])
    optimizer.zero_grad()
    return state["extra"]
