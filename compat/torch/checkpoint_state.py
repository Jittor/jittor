"""FQN-based optimizer checkpoint transforms for the Torch frontend.

Tensor computation and optimizer loading stay with their native owners. FSDP
shard/full transforms are supplied through the registered FSDP provider seam.
"""

from collections.abc import Mapping
import warnings

import jittor as jt

from .context import get_install_context
from .. import fsdp_hooks


def _torch():
    return get_install_context(jt).target_namespace


def _options(options):
    if options is None:
        from .installers.distributed import StateDictOptions
        options = StateDictOptions()
    if options.broadcast_from_rank0 and not options.full_state_dict:
        raise ValueError("broadcast_from_rank0 requires full_state_dict=True")
    return options


def _optimizer_load_options(options):
    options = _options(options)
    if options.full_state_dict and options.flatten_optimizer_state_dict:
        raise NotImplementedError("loading full_state_dict with flatten_optimizer_state_dict "
                                  "is not supported; use either option separately")
    return options


def _optimizers(optimizers):
    if hasattr(optimizers, "param_groups"):
        return (optimizers,)
    return tuple(optimizers) if optimizers is not None else ()


def _map_tree(value, function):
    if isinstance(value, jt.Var):
        return function(value)
    if isinstance(value, Mapping):
        return {key: _map_tree(item, function) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_map_tree(item, function) for item in value)
    if isinstance(value, list):
        return [_map_tree(item, function) for item in value]
    return value


def _offload_to_cpu(value):
    result = value.cpu()
    # Every rank must execute lazy FULL gathers before a nonzero rank drops
    # its result. Otherwise rank-zero serialization and the next collective
    # can execute different NCCL operations in the same stream position.
    result.sync()
    return result


def _fsdp(model):
    provider = fsdp_hooks.provider()
    if provider is not None and any(getattr(child, "_is_fsdp_module", False)
                                    for child in model.modules()):
        return provider
    return None


def _unwrapped_model(model):
    # DDP is a wrapper; an ordinary child literally called `module` is not.
    if isinstance(model, _torch().nn.parallel.DistributedDataParallel):
        return model.module
    return model


def _parameter_names(model):
    model = _unwrapped_model(model)
    names = {}
    for name, parameter in model.named_parameters():
        names.setdefault(id(parameter), name)
    provider = _fsdp(model)
    if provider is not None:
        names.update(provider._optimizer_checkpoint_parameter_names(model))
    return names


def _parameter_aliases(model):
    """Resolve every public alias through the native parameter traversal."""
    names = _parameter_names(model)
    aliases = {name: names[id(parameter)] for name, parameter in
               _unwrapped_model(model).named_parameters(remove_duplicate=False)
               if id(parameter) in names}
    aliases.update((name, name) for name in names.values())
    return aliases


def _submodule_prefixes(model, submodules):
    if not submodules:
        return ()
    warnings.warn("Getting submodule-only checkpoint state is deprecated; "
                  "filter the returned state dictionary instead.", FutureWarning, stacklevel=3)
    selected = {id(module) for module in submodules}
    return tuple(name + "." for name, module in model.named_modules()
                 if id(module) in selected)


def _normalize_optimizer_aliases(state, model):
    aliases = _parameter_aliases(model)
    if "state" not in state:
        result = {}
        # Parameter names can contain dots; resolve the whole known prefix.
        prefixes = sorted(aliases, key=len, reverse=True)
        for key, value in state.items():
            normalized = key
            for section in ("state.", "param_groups."):
                if not key.startswith(section):
                    continue
                for name in prefixes:
                    prefix = section + name + "."
                    if key.startswith(prefix):
                        normalized = section + aliases[name] + "." + key[len(prefix):]
                        break
            if normalized in result:
                raise ValueError("duplicate optimizer checkpoint parameter alias: " + normalized)
            result[normalized] = value
        return result
    fields = {}
    for name, value in state["state"].items():
        canonical = aliases.get(name, name)
        if canonical in fields:
            raise ValueError("duplicate optimizer checkpoint parameter alias: " + canonical)
        fields[canonical] = value
    groups = []
    for group in state["param_groups"]:
        params = [aliases.get(name, name) for name in group["params"]]
        if len(set(params)) != len(params):
            raise ValueError("duplicate optimizer checkpoint parameter aliases in one group")
        groups.append(dict(group, params=params))
    return {"state": fields, "param_groups": groups}


def _initialize(optimizer):
    """Match DCP's initialization step without changing parameter values."""
    if optimizer.state:
        return
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    if any(p.grad is not None for p in parameters):
        return
    for parameter in parameters:
        if parameter.requires_grad:
            parameter.grad = _torch().zeros_like(parameter)
    rates = [group.get("lr") for group in optimizer.param_groups]
    try:
        for group in optimizer.param_groups:
            if "lr" in group:
                group["lr"] = 0.0
        optimizer.step()
    finally:
        for group, rate in zip(optimizer.param_groups, rates):
            if rate is not None:
                group["lr"] = rate
        optimizer.zero_grad(set_to_none=True)


def _flatten(state):
    result = {}
    def visit(value, prefix):
        if isinstance(value, Mapping):
            for key, item in value.items():
                visit(item, prefix + "." + str(key))
        else:
            result[prefix] = value
    for name, fields in state["state"].items():
        visit(fields, "state." + name)
    for group in state["param_groups"]:
        for name in group["params"]:
            for key, value in group.items():
                if key != "params":
                    result["param_groups." + name + "." + key] = value
    return result


def _unflatten(state, model, optimizers):
    names = _parameter_names(model)
    result = {"state": {}, "param_groups": []}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            fqns = [names[id(p)] for p in group["params"]]
            loaded = {"params": fqns}
            for name in fqns:
                fields = {}
                prefix = "state." + name + "."
                group_prefix = "param_groups." + name + "."
                for key, value in state.items():
                    if key.startswith(prefix):
                        path = key[len(prefix):].split(".")
                        target = fields
                        for part in path[:-1]:
                            target = target.setdefault(part, {})
                        target[path[-1]] = value
                    elif key.startswith(group_prefix):
                        field = key[len(group_prefix):]
                        if field in loaded and loaded[field] != value:
                            raise ValueError("inconsistent optimizer parameter-group option: " + field)
                        loaded[field] = value
                if fields:
                    result["state"][name] = fields
            result["param_groups"].append(loaded)
    return result


def get_model_state_dict(model, *, submodules=None, options=None):
    options = _options(options)
    model = _unwrapped_model(model)
    prefixes = _submodule_prefixes(model, submodules)
    provider = _fsdp(model)
    result = provider._get_full_state_dict(model) if provider is not None and options.full_state_dict else model.state_dict()
    if options.ignore_frozen_params:
        frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
        result = {name: value for name, value in result.items() if name not in frozen}
    if prefixes:
        selected = {}
        for name, value in result.items():
            for prefix in prefixes:
                if name.startswith(prefix):
                    selected[name if options.keep_submodule_prefixes else name[len(prefix):]] = value
        result = selected
    if options.cpu_offload:
        result = _map_tree(result, _offload_to_cpu)
        dist = _torch().distributed
        if options.full_state_dict and dist.is_initialized() and dist.get_rank() != 0:
            return {}
    return result


def _broadcast_state(state, options):
    dist = _torch().distributed
    if options.broadcast_from_rank0 and dist.is_initialized():
        payload = [state]
        dist.broadcast_object_list(payload, src=0)
        return payload[0]
    return state


def set_model_state_dict(model, state_dict, *, options=None):
    options = _options(options)
    model = _unwrapped_model(model)
    state = _broadcast_state(state_dict, options)
    if state and all(isinstance(key, _torch().nn.Module) for key in state):
        modules = {id(module): name for name, module in model.named_modules()}
        expanded = {}
        for module, values in state.items():
            if id(module) in modules:
                prefix = modules[id(module)]
                expanded.update(((prefix + "." if prefix else "") + name, value)
                                for name, value in values.items())
        state = expanded
    provider = _fsdp(model)
    if provider is not None and options.full_state_dict:
        provider._load_full_state_dict(model, state, strict=options.strict)
        return None
    loader = model.load_state_dict
    if getattr(loader, "__func__", None) is jt.Module.load_state_dict:
        # Native Module lacks Torch's strict/copying contract. Use the
        # installation's loader, retaining custom overrides on other modules.
        return _torch().nn.Module.load_state_dict(model, state, strict=options.strict)
    return loader(state, strict=options.strict)


def get_optimizer_state_dict(model, optimizers, *, submodules=None, options=None):
    options = _options(options)
    # Torch validates this legacy option but leaves optimizer FQNs and groups
    # complete, even when the accompanying model dictionary is filtered.
    _submodule_prefixes(_unwrapped_model(model), submodules)
    optimizers = _optimizers(optimizers)
    if not optimizers:
        raise RuntimeError("Optimizers are not passed in but optim_only is set to True.")
    result = {"state": {}, "param_groups": []}
    for optimizer in optimizers:
        _initialize(optimizer)
        state = optimizer.state_dict()
        # FSDP's initialization step may rotate its public shard views.
        names = _parameter_names(model)
        pid_to_name = {}
        for group, saved_group in zip(optimizer.param_groups, state["param_groups"]):
            for parameter, pid in zip(group["params"], saved_group["params"]):
                if id(parameter) not in names:
                    raise ValueError("optimizer parameter does not belong to the model")
                pid_to_name[pid] = names[id(parameter)]
            result["param_groups"].append(dict(saved_group,
                params=[pid_to_name[pid] for pid in saved_group["params"]]))
        result["state"].update({pid_to_name[pid]: fields for pid, fields in state["state"].items()})
    provider = _fsdp(model)
    if provider is not None and options.full_state_dict:
        metadata = {}
        for name, fields in result["state"].items():
            metadata[name] = {key: (("step", float(value.item())) if key == "step" and isinstance(value, jt.Var)
                                    else ("tensor", str(value.dtype)) if isinstance(value, jt.Var)
                                    else ("value", value)) for key, value in fields.items()}
        dist = _torch().distributed
        ranks = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if len(ranks) > 1:
            dist.all_gather_object(ranks, metadata)
        else:
            ranks[0] = metadata
        result = provider._gather_optimizer_state_dict(model, result, ranks)
    if options.flatten_optimizer_state_dict:
        result = _flatten(result)
    if options.cpu_offload:
        result = _map_tree(result, _offload_to_cpu)
        dist = _torch().distributed
        if options.full_state_dict and dist.is_initialized() and dist.get_rank() != 0:
            return {}
    return result


def set_optimizer_state_dict(model, optimizers, optim_state_dict, *, options=None):
    options = _optimizer_load_options(options)
    optimizers = _optimizers(optimizers)
    if not optimizers:
        raise RuntimeError("Optimizers are not passed in but optim_only is set to True.")
    state = _broadcast_state(optim_state_dict, options)
    if not state:
        raise RuntimeError("optimizer state_dict is empty but an optimizer was supplied")
    state = _normalize_optimizer_aliases(state, model)
    if "state" not in state:
        state = _unflatten(state, model, optimizers)
    provider = _fsdp(model)
    if provider is not None and options.full_state_dict:
        state = provider._shard_optimizer_state_dict(model, state)
    if provider is not None:
        # FULL model loading can replace flat-entry shard views. Rebind the
        # live optimizer through its existing owner before resolving FQNs.
        for optimizer in optimizers:
            provider.refresh_optimizer_fsdp_params(optimizer)
    names = _parameter_names(model)
    plans = []
    for optimizer in optimizers:
        current = optimizer.state_dict()
        groups, values = [], {}
        for group, current_group in zip(optimizer.param_groups, current["param_groups"]):
            fqns = [names[id(parameter)] for parameter in group["params"]]
            loaded_group = dict(current_group)
            for saved in state["param_groups"]:
                if (set(saved["params"]) & set(fqns)) or (not saved["params"] and not fqns):
                    loaded_group.update((key, value) for key, value in saved.items() if key != "params")
            groups.append(loaded_group)
            for parameter, pid, name in zip(group["params"], current_group["params"], fqns):
                if options.strict and state["state"] and parameter.requires_grad and name not in state["state"]:
                    raise RuntimeError("Missing optimizer state for parameter '" + name +
                                       "' in checkpoint. To load anyway, use StateDictOptions(strict=False).")
                if name in state["state"]:
                    # A host-offloaded checkpoint must resume with its moments
                    # on the parameter's actual device, never a CPU fallback.
                    values[pid] = _map_tree(state["state"][name],
                        lambda value, p=parameter: _torch().clone(value.to(device=p.device).detach()))
        plans.append((optimizer, {"state": values, "param_groups": groups}))
    for optimizer, state in plans:
        optimizer.load_state_dict(state)


def get_state_dict(model, optimizers=None, *, submodules=None, options=None):
    optimizers = _optimizers(optimizers)
    model_state = get_model_state_dict(model, submodules=submodules, options=options)
    optimizer_state = (get_optimizer_state_dict(model, optimizers, submodules=submodules, options=options)
                       if optimizers else {})
    return model_state, optimizer_state


def set_state_dict(model, optimizers=None, *, model_state_dict, optim_state_dict, options=None):
    optimizers = _optimizers(optimizers)
    options = _optimizer_load_options(options) if optimizers else _options(options)
    if optimizers and not optim_state_dict and not options.broadcast_from_rank0:
        raise RuntimeError("optimizer state_dict is empty but an optimizer was supplied")
    if model_state_dict or options.broadcast_from_rank0:
        incompatible = set_model_state_dict(model, model_state_dict, options=options)
    else:
        incompatible = _torch().nn.modules.module._IncompatibleKeys([], [])
    if optimizers:
        set_optimizer_state_dict(model, optimizers, optim_state_dict, options=options)
    return incompatible
