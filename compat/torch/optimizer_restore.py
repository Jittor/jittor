"""Optimizer state restore implementation for the Torch compatibility layer."""
from collections.abc import Mapping

import jittor as jt
import numpy as np

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from .context import get_install_context


def _restore_optimizer_state(self, state_dict):
    from .optimizer_api import (
        _MOMENTUM_INITIALIZED,
        _momentum_initialized_flags,
        _owned_state,
        _torch_optimizer_kind,
        _torch_param_steps,
    )
    from .optimizer_state import (
        _TORCH_TO_NATIVE_STATE,
        _adagrad_step_tensor,
        _native_optimizer_buffer,
    )

    restore_seen = bool(getattr(self, "_torch_optimizer_restore_seen", False))
    context = get_install_context(jt)
    target = context.target_namespace
    native_load = context.state["optimizer_native_api"]["_native_load_state_dict"]
    if not isinstance(state_dict, Mapping) or "param_groups" not in state_dict:
        return native_load(self, state_dict)
    saved_groups = state_dict.get("param_groups", [])
    saved_state = state_dict.get("state", {})
    kind = _torch_optimizer_kind(self)
    if not isinstance(saved_groups, (list, tuple)):
        raise TypeError("loaded optimizer param_groups must be a sequence")
    if not isinstance(saved_state, Mapping):
        raise TypeError("loaded optimizer state must be a mapping")
    if len(saved_groups) != len(self.param_groups):
        raise ValueError("loaded state dict has a different number of parameter groups")

    if kind == "adagrad":
        from jittor.optim.algorithms.adagrad import validate_adagrad_options

    load_plan = []
    max_step = 0
    state_slots = _TORCH_TO_NATIVE_STATE.get(kind, {})
    from .installers.tensor.method_api import _to

    for saved_group, current_group in zip(saved_groups, self.param_groups):
        if not isinstance(saved_group, Mapping):
            raise TypeError("loaded optimizer parameter group must be a mapping")
        saved_params = saved_group.get("params", [])
        if not isinstance(saved_params, (list, tuple)):
            raise TypeError("loaded optimizer group params must be a sequence")
        if len(saved_params) != len(current_group.get("params", [])):
            raise ValueError(
                "loaded state dict contains a parameter group that "
                "doesn't match the size of optimizer's group")
        if kind == "adagrad":
            validate_adagrad_options(dict(current_group, **saved_group))

        current_steps = _torch_param_steps(current_group)
        prepared_slots = []
        for index, (param_id, param) in enumerate(
                zip(saved_params, current_group["params"])):
            missing = object()
            try:
                state = saved_state.get(param_id, missing)
                if state is missing:
                    state = saved_state.get(str(param_id), {})
            except (TypeError, ValueError) as error:
                raise TypeError("loaded optimizer state key is invalid") from error
            if not isinstance(state, Mapping):
                raise TypeError("loaded optimizer parameter state must be a mapping")
            state = dict(state)

            if kind == "adagrad":
                total = state.get("sum")
                if not isinstance(total, jt.Var) or tuple(total.shape) != tuple(param.shape):
                    raise ValueError("loaded Adagrad sum must match its parameter shape")
                if _jittor_dtype_name(total.dtype) != "float32":
                    raise NotImplementedError("loaded Adagrad sum must be float32")
                if "step" not in state:
                    raise ValueError("loaded Adagrad state requires step")

            step = 1 if state else 0
            if "step" in state:
                step_value = state["step"]
                if isinstance(step_value, jt.Var):
                    step_value = step_value.item()
                try:
                    numeric = float(step_value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError("loaded optimizer step must be numeric") from error
                if (not np.isfinite(numeric) or numeric < 0
                        or numeric != int(numeric)):
                    raise ValueError(
                        "loaded optimizer step must be a non-negative integer")
                step = int(numeric)

            # Move and clone every recognized slot before commit. This preserves
            # source dtype, follows each parameter's real CPU/CUDA placement,
            # and prevents restored optimizers from sharing mutable checkpoint
            # storage with the source state or another optimizer.
            for torch_key, native_key in state_slots.items():
                if torch_key not in state:
                    continue
                buffers = current_group.get(native_key)
                if not isinstance(buffers, list) or index >= len(buffers):
                    continue
                value = state[torch_key]
                if torch_key == "momentum_buffer" and value is None:
                    continue
                if not isinstance(value, jt.Var):
                    raise TypeError("loaded optimizer %s must be a tensor" % torch_key)
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError(
                        "loaded optimizer %s shape does not match parameter" % torch_key)
                if kind == "adagrad" and torch_key == "sum"                         and _jittor_dtype_name(value.dtype) != "float32":
                    raise NotImplementedError("loaded Adagrad sum must be float32")

                backend, device_index = jt.core.dispatch_context([param])
                if backend == "cpu" and device_index == -1:
                    device = target.device("cpu")
                elif backend == "cuda" and device_index >= 0:
                    device = target.device("cuda", int(device_index))
                else:
                    # Keep established behavior on backends without a tested
                    # frontend device transfer path.
                    state[torch_key] = _owned_state(
                        _native_optimizer_buffer(value, buffers[index]))
                    continue

                moved = _to(value, device=device)
                if _jittor_dtype_name(moved.dtype) != _jittor_dtype_name(value.dtype):
                    raise RuntimeError("loaded optimizer %s dtype changed" % torch_key)
                moved.sync(True, False)
                native_value = _native_optimizer_buffer(moved, buffers[index])
                owned = _owned_state(native_value)
                owned.sync(True, False)
                if tuple(jt.core.dispatch_context([owned])) != (backend, device_index):
                    raise RuntimeError("loaded optimizer %s dispatch changed" % torch_key)
                state[torch_key] = owned

            max_step = max(max_step, step)
            adagrad_step = None
            adam_zero_step = None
            if kind == "adagrad":
                adagrad_step = _adagrad_step_tensor(step)
                adagrad_step.sync(True, False)
            elif (kind in ("adam", "adamw") and step == 0
                  and "exp_avg" in state and "exp_avg_sq" in state):
                fused = current_group.get("fused") or current_group.get("capturable")
                device = current_group["params"][index].device if fused else "cpu"
                dtype = (target.float64 if not fused
                         and target.get_default_dtype() == target.float64
                         else target.float32)
                adam_zero_step = target.tensor(0., dtype=dtype, device=device)
                adam_zero_step.sync(True, False)
            prepared_slots.append((state, step, adagrad_step, adam_zero_step))

        # Prepare reset buffers before commit, so allocation/transfer failures
        # cannot leave some groups partially restored.
        resets = {}
        for native_key in set(state_slots.values()):
            buffers = current_group.get(native_key)
            if not isinstance(buffers, list):
                continue
            torch_keys = [key for key, native in state_slots.items()
                          if native == native_key]
            reset = {}
            for index, buffer in enumerate(buffers):
                replaced = (index < len(prepared_slots)
                            and any(key in prepared_slots[index][0]
                                    for key in torch_keys))
                if replaced or not isinstance(buffer, jt.Var):
                    continue
                current_step = current_steps[index] if index < len(current_steps) else None
                if isinstance(current_step, jt.Var):
                    current_step = current_step.item()
                if (not restore_seen and current_step == 0
                        and index < len(prepared_slots)
                        and prepared_slots[index][1] == 0):
                    continue
                zero = jt.zeros_like(buffer).stop_grad()
                if index < len(current_group["params"]):
                    reset_backend, _ = jt.core.dispatch_context(
                        [current_group["params"][index]])
                    if reset_backend in ("cpu", "cuda"):
                        zero.sync(True, False)
                reset[index] = zero
            resets[native_key] = reset
        load_plan.append((dict(saved_group), prepared_slots, resets))

    # Commit only after every group's metadata, transfers, clones and zero
    # buffers have been prepared successfully.
    for group_index, (saved_group, slots, resets) in enumerate(load_plan):
        group = self.param_groups[group_index]
        steps = _torch_param_steps(group)
        if kind != "adagrad":
            for index in range(len(steps)):
                list.__setitem__(steps, index, 0)
        if kind == "sgd":
            flags = _momentum_initialized_flags(group)
            for index in range(len(flags)):
                flags[index] = False
        for native_key, reset in resets.items():
            for index, zero in reset.items():
                group[native_key][index] = zero
        for key, value in saved_group.items():
            if key in ("params", _MOMENTUM_INITIALIZED):
                continue
            group[key] = value

        for index, (state, step, adagrad_step, adam_zero_step) in enumerate(slots):
            for torch_key, native_key in state_slots.items():
                if torch_key not in state:
                    continue
                if torch_key == "momentum_buffer" and state[torch_key] is None:
                    continue
                if index >= len(group.get(native_key, ())):
                    continue
                group[native_key][index] = state[torch_key]
                if kind == "sgd" and torch_key == "momentum_buffer":
                    _momentum_initialized_flags(group)[index] = True

            if kind == "adagrad":
                # Keep the native per-parameter step Tensor object and update it
                # in place. Adagrad's step slot is not interchangeable with a
                # numeric Adam/SGD counter.
                steps[index].update(adagrad_step)
            elif kind in ("adam", "adamw") and isinstance(state.get("step"), jt.Var):
                # _AdamSteps wraps the source Tensor, preserving Torch's live
                # scalar-step behavior for state/state_dict views.
                steps[index] = state["step"]
            elif adam_zero_step is not None:
                steps[index] = adam_zero_step
            else:
                steps[index] = step

    self.n_step = max_step
    self._torch_optimizer_restore_seen = True
    return None
