"""Bounded optimizer checkpoint restore; dependencies injected by owner."""
from collections.abc import Mapping
import jittor as jt
import numpy as np
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from .context import get_install_context

def load_state_dict_torch(self, state_dict, _torch_optimizer_kind, _torch_param_steps):
    restore_seen = bool(getattr(self, "_torch_optimizer_restore_seen", False))
    _context = get_install_context(jt)
    _native_load_state_dict = _context.state["optimizer_native_api"]['_native_load_state_dict']
    if not isinstance(state_dict, Mapping) or "param_groups" not in state_dict:
        return _native_load_state_dict(self, state_dict)
    saved_groups = state_dict.get("param_groups", [])
    saved_state = state_dict.get("state", {})
    kind = _torch_optimizer_kind(self)
    if not isinstance(saved_groups, (list, tuple)):
        raise TypeError("loaded optimizer param_groups must be a sequence")
    if not isinstance(saved_state, Mapping):
        raise TypeError("loaded optimizer state must be a mapping")
    if len(saved_groups) != len(self.param_groups):
        raise ValueError(
            "loaded state dict has a different number of parameter groups")
    load_plan = []
    max_step = 0
    # These are the slot names the adapter actually consumes. Keep each
    # loaded tensor's dtype (e.g. FP32 moments for BF16 parameters).
    state_slots = {
        "adam": {"exp_avg": "m", "exp_avg_sq": "values"},
        "adamw": {"exp_avg": "m", "exp_avg_sq": "values"},
        "sgd": {"momentum_buffer": "values"},
        "rmsprop": {"square_avg": "values"},
        "adan": {"exp_avg": "m", "exp_avg_sq": "v",
                 "exp_avg_diff": "d", "pre_grad": "pre_grad"},
    }.get(kind, {})
    from .installers.tensor.method_api import _to
    for saved_pg, current_pg in zip(saved_groups, self.param_groups):
        if not isinstance(saved_pg, Mapping):
            raise TypeError("loaded optimizer parameter group must be a mapping")
        saved_params = saved_pg.get("params", [])
        if not isinstance(saved_params, (list, tuple)):
            raise TypeError("loaded optimizer group params must be a sequence")
        if len(saved_params) != len(current_pg.get("params", [])):
            raise ValueError(
                "loaded state dict contains a parameter group that "
                "doesn't match the size of optimizer's group")
        current_steps = _torch_param_steps(current_pg)
        slots = []
        for i, (pid, param) in enumerate(zip(saved_params, current_pg["params"])):
            missing = object()
            try:
                st = saved_state.get(pid, missing)
                if st is missing:
                    st = saved_state.get(str(pid), {})
            except (TypeError, ValueError) as error:
                raise TypeError("loaded optimizer state key is invalid") from error
            if not isinstance(st, Mapping):
                raise TypeError("loaded optimizer parameter state must be a mapping")
            st = dict(st)
            step = 1 if st else 0
            if "step" in st:
                value = st["step"]
                if isinstance(value, jt.Var):
                    value = value.item()
                try:
                    numeric = float(value)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError("loaded optimizer step must be numeric") from error
                if not np.isfinite(numeric) or numeric < 0 or numeric != int(numeric):
                    raise ValueError(
                        "loaded optimizer step must be a non-negative integer")
                step = int(numeric)
            # Prepare only known tensor fields that will actually be used by
            # this optimizer's existing slot. Never rewrite the source state.
            for field, slot in state_slots.items():
                buffers = current_pg.get(slot)
                if field not in st or not isinstance(buffers, list) or i >= len(buffers):
                    continue
                # A public device for FollowRuntime can be masked by a
                # frontend default-device context. Match native dispatch and
                # index independently for every CPU/CUDA parameter. Other
                # backends retain their existing load behavior until tested.
                backend, index = jt.core.dispatch_context([param])
                if backend == "cpu" and index == -1:
                    device = _context.target_namespace.device("cpu")
                elif backend == "cuda" and index >= 0:
                    device = _context.target_namespace.device("cuda", int(index))
                else:
                    continue
                value = st[field]
                if not isinstance(value, jt.Var):
                    raise TypeError("loaded optimizer %s must be a tensor" % field)
                if tuple(value.shape) != tuple(param.shape):
                    raise ValueError("loaded optimizer %s shape does not match parameter" % field)
                moved = _to(value, device=device)
                if _jittor_dtype_name(moved.dtype) != _jittor_dtype_name(value.dtype):
                    raise RuntimeError("loaded optimizer %s dtype changed" % field)
                # Materialize before committing any group changes, so a device
                # transfer failure cannot leave earlier slots partly restored.
                moved.sync(True, False)
                if tuple(jt.core.dispatch_context([moved])) != (backend, index):
                    raise RuntimeError("loaded optimizer %s dispatch changed" % field)
                st[field] = moved
            max_step = max(max_step, step)
            slots.append((st, step))
        # Prepare zero resets too: a failed allocation must precede commit.
        resets = {}
        for key in ("m", "values", "v", "d", "pre_grad"):
            buffers = current_pg.get(key)
            if not isinstance(buffers, list):
                continue
            fields = [field for field, slot in state_slots.items() if slot == key]
            reset = {}
            for i, buffer in enumerate(buffers):
                replaced = i < len(slots) and any(field in slots[i][0] for field in fields)
                if not replaced and isinstance(buffer, jt.Var):
                    # A fresh, unstepped slot already contains its zero
                    # initializer. Reuse it for an empty step-zero restore:
                    # Accelerate wraps optimizers by loading their own initial
                    # state_dict, and eagerly syncing replacement zeros here
                    # needlessly materializes every unused moment on every rank.
                    if (not restore_seen and i < len(current_steps)
                            and i < len(slots) and current_steps[i] == 0
                            and slots[i][1] == 0):
                        continue
                    zero = jt.zeros_like(buffer).stop_grad()
                    # Leave other backends' legacy reset realization alone.
                    if i < len(current_pg["params"]):
                        reset_backend, _ = jt.core.dispatch_context([current_pg["params"][i]])
                        if reset_backend in ("cpu", "cuda"):
                            zero.sync(True, False)
                    reset[i] = zero
            resets[key] = reset
        load_plan.append((dict(saved_pg), slots, resets))
    # All validation, transfers and reset allocations precede this commit.
    for gi, (saved_pg, slots, resets) in enumerate(load_plan):
        pg = self.param_groups[gi]
        steps = _torch_param_steps(pg)
        for i in range(len(steps)):
            steps[i] = 0
        for key, reset in resets.items():
            for i, zero in reset.items():
                pg[key][i] = zero
        for k, v in saved_pg.items():
            if k == "params":
                continue
            pg[k] = v
        for i, (st, step) in enumerate(slots):
            if kind in ("adam", "adamw"):
                if "m" in pg and i < len(pg["m"]) and "exp_avg" in st:
                    pg["m"][i] = st["exp_avg"]
                if "values" in pg and i < len(pg["values"]) \
                        and "exp_avg_sq" in st:
                    pg["values"][i] = st["exp_avg_sq"]
            elif kind == "sgd" and "momentum_buffer" in st:
                pg["values"][i] = st["momentum_buffer"]
            elif kind == "rmsprop" and "square_avg" in st:
                pg["values"][i] = st["square_avg"]
            elif kind == "adan":
                for source, target in (
                        ("m", "exp_avg"), ("v", "exp_avg_sq"),
                        ("d", "exp_avg_diff"),
                        ("pre_grad", "pre_grad")):
                    if target in st and source in pg and i < len(pg[source]):
                        pg[source][i] = st[target]
            steps[i] = step
    self.n_step = max_step
    self._torch_optimizer_restore_seen = True
    return None
