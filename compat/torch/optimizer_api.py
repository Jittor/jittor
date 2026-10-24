"""Stable optimizer state and update adapters, sharing native mathematics."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import weakref as _weakref
import jittor as jt
from .context import get_install_context
from ..diagnostics import EXPECTED, swallowed
from typing import Any, List
from .. import fsdp_hooks as _fsdp_hooks
from .. import optimizer_kinds as _optimizer_kinds
from .tensor_state import get_tensor_state
from .installers.tensor.autograd_api import _optimizer_maybe_has_fsdp_params
from jittor.optim.algorithms.adam import adam_update
from .optim_frontend import (
    _initialize_default, _validate_sgd_groups, _validate_adamw_capturable_groups,
)

def _init(self, *a, **k):
    _context = get_install_context(jt)
    _orig_init = _context.state["optimizer_native_api"]['_orig_init']
    _orig_init(self, *a, **k)
    # Maintain a registry of ALL live optimizers (not just the last). torch
    # supports several optimizers active at once (3DGS has a Gaussian Adam +
    # an exposure Adam); loss.backward() must fill grads for every one. Hold
    # weakrefs, pruned of dead entries + this same object.
    reg = get_tensor_state(jt).active_optimizers
    reg[:] = [r for r in reg if r() is not None and r() is not self]
    reg.append(_weakref.ref(self))
    try:
        for pg in self.param_groups:
            pg.setdefault("lr", self.lr)
    except EXPECTED as exc:
        swallowed("torch/optimizers.py _init: for pg in self.param_groups:", exc)
    # A gradient belongs to its Tensor, including when it was produced before
    # this optimizer existed. Keep the published objects themselves: clipping,
    # clearing and another optimizer must observe the same gradient storage.
    published = [(pg, [getattr(parameter, "_torch_grad", None)
                       for parameter in pg["params"]])
                 for pg in self.param_groups]
    if any(isinstance(gradient, jt.Var)
           for _, gradients in published for gradient in gradients):
        for pg, gradients in published:
            pg["grads"] = gradients
        object.__setattr__(self, "_Optimizer__zero_grad", False)


def _torch_param_steps(pg):
    _sync_adam_group_params(pg)
    params = list(pg.get("params", []))
    steps = pg.get("_torch_steps")
    if not isinstance(steps, list):
        steps = pg["_torch_steps"] = [0] * len(params)
    while len(steps) < len(params):
        steps.append(0)
    if len(steps) > len(params):
        del steps[len(params):]
    return steps


def _sync_adam_group_params(pg, *, replacements=()):
    """Realign state by identity, including explicit same-parameter view changes.

    FSDP supplies old/new views of the same logical parameter. Ordinary user
    additions and reordering still pass through the identity-based mapping;
    only those explicit substitutions inherit the preceding view's state.
    """
    previous = pg.get("_torch_adam_params")
    if previous is None:
        return
    current = tuple(pg.get("params", ()))
    if len(current) == len(previous) and all(a is b for a, b in zip(current, previous)):
        return
    from jittor.optim.base import _state_buffer
    rebound = {id(old): (old, new) for old, new in replacements}
    identities = []
    for parameter in previous:
        replacement = rebound.get(id(parameter))
        identities.append(replacement[1] if replacement is not None
                          and replacement[0] is parameter else parameter)
    positions = {id(parameter): index for index, parameter in enumerate(identities)}
    for key in ("m", "values", "_torch_steps"):
        old_values = pg.get(key, ())
        result = []
        for parameter in current:
            old_index = positions.get(id(parameter))
            if old_index is not None and identities[old_index] is parameter and old_index < len(old_values):
                result.append(old_values[old_index])
            else:
                result.append(0 if key == "_torch_steps" else _state_buffer(parameter))
        pg[key] = result
    pg["_torch_adam_params"] = current


def _torch_optimizer_kind(opt):
    """Which optimizer's state layout `opt` has.

    Identity through the MRO, not a substring of the class name -- `SGDW`
    and `MyAdamWrapper` used to match rules they do not implement. This
    answer only describes *state layout* (which keys `state` and
    `state_dict()` expose), so unlike the FSDP2 one it does not refuse a
    subclass that overrides step(): such a subclass still keeps the base
    class's state arrays. It falls back to the lowercased class name so an
    unrecognised optimizer keeps its previous, harmless behaviour here.

    See jittor/compat/optimizer_kinds.py.
    """
    return (_optimizer_kinds.kind_of(opt)
            or type(opt).__name__.lower())


from .optimizer_state import (
    _load_state_dict_torch, _OptState, _ParamState, _state_dict_torch,
)


def _zero_grad_compat(self, set_to_none=True):
    _context = get_install_context(jt)
    _orig_zero = _context.state["optimizer_native_api"]['_orig_zero']
    for _pg in getattr(self, "param_groups", []):
        _params = list(_pg.get("params", []))
        _new_grads: List[Any] = []
        if set_to_none:
            _pg.pop("grads", None)
        else:
            _old_grads = list(_pg.get("grads") or [])
            _new_grads = []
            for _i, _p in enumerate(_params):
                _old = _old_grads[_i] if _i < len(_old_grads) else None
                _published = (
                    getattr(_p, "_torch_grad", None)
                    if isinstance(_p, jt.Var) else None
                )
                if isinstance(_p, jt.Var) and (
                        isinstance(_old, jt.Var)
                        or isinstance(_published, jt.Var)):
                    _existing = (
                        _old if isinstance(_old, jt.Var)
                        else _published
                    )
                    _existing.update(jt.zeros_like(_existing))
                    _existing.stop_grad().stop_fuse()
                    _new_grads.append(_existing)
                else:
                    _new_grads.append(None)
            if any(isinstance(_g, jt.Var) for _g in _new_grads):
                _pg["grads"] = _new_grads
            else:
                _pg.pop("grads", None)
        for _i, _p in enumerate(_params):
            if not isinstance(_p, jt.Var):
                continue
            _value = None
            if not set_to_none and _i < len(_new_grads):
                _value = _new_grads[_i]
            try:
                _p.grad = _value
            except (AttributeError, TypeError) as exc:
                swallowed("torch/optimizers.py _zero_grad_compat: _p.grad = _value", exc)
                object.__setattr__(_p, "_torch_grad", _value)
    try:
        object.__setattr__(self, "_grad_map", {})
    except (AttributeError, TypeError) as exc:
        swallowed("torch/optimizers.py _zero_grad_compat: object.__setattr__(self, '_grad_map', {})", exc)
    result = _orig_zero(self)
    _fsdp2_zero = _fsdp_hooks.provider()
    if _fsdp2_zero is not None and _fsdp2_zero.optimizer_has_fsdp_params(self):
        _fsdp2_zero.refresh_visible_full_grads(self)
    object.__setattr__(self, "_torch_backward_advanced_n_step", False)
    return result


def _backward_with_step_marker(self, *args, **kwargs):
    _context = get_install_context(jt)
    _orig_backward = _context.state["optimizer_native_api"]['_orig_backward']
    result = _orig_backward(self, *args, **kwargs)
    object.__setattr__(self, "_torch_backward_advanced_n_step", True)
    return result


def _optimizer_has_ready_grads(opt):
    for _pg in getattr(opt, "param_groups", []):
        _grads = _pg.get("grads")
        if not _grads:
            continue
        for _p, _g in zip(_pg.get("params", []), _grads):
            if isinstance(_p, jt.Var) and isinstance(_g, jt.Var) and list(_p.shape) == list(_g.shape):
                return True
    return False


def _advance_ready_param_steps(opt):
    for pg in getattr(opt, "param_groups", []):
        steps = _torch_param_steps(pg)
        grads = pg.get("grads") or []
        for i, (param, grad) in enumerate(zip(pg.get("params", []), grads)):
            if isinstance(param, jt.Var) and isinstance(grad, jt.Var) \
                    and list(param.shape) == list(grad.shape):
                steps[i] = int(steps[i]) + 1


def _advance_trainable_param_steps(opt):
    for pg in getattr(opt, "param_groups", []):
        steps = _torch_param_steps(pg)
        for i, param in enumerate(pg.get("params", [])):
            if isinstance(param, jt.Var) and param.requires_grad:
                steps[i] = int(steps[i]) + 1


def _load_fsdp2_for_optimizer(opt):
    # The seam, not an import: this file is below fsdp2 in the dependency
    # order (jittor/compat/fsdp_hooks.py). The guard is what makes the
    # `None` answer safe -- `_jittor_fsdp2_state` is set only by fsdp2, and
    # fsdp2 registers itself when imported, so an optimizer holding a
    # sharded parameter always finds a provider here.
    if not _optimizer_maybe_has_fsdp_params(opt):
        return None
    return _fsdp_hooks.provider()


class LBFGS(jt.optim.Optimizer):
    def __init__(self, params, lr=1.0, *args, **kwargs):
        super().__init__(params, lr)
    def step(self, closure=None):
        raise NotImplementedError("torch.optim.LBFGS is not implemented by the jittor torch shim")


def _lsd(self, state):
    _context = get_install_context(jt)
    _orig_lsd = _context.state["optimizer_native_api"]['_orig_lsd']
    trainable = []
    try:
        for pg in self.param_groups:
            for p in pg.get("params", []):
                if p.requires_grad:
                    trainable.append(p)
    except EXPECTED as exc:
        swallowed("torch/optimizers.py _lsd: for pg in self.param_groups:", exc)
    r = _orig_lsd(self, state)
    for p in trainable:
        try: p.start_grad()
        except EXPECTED as exc: swallowed("torch/optimizers.py _lsd: p.start_grad()", exc)
    return r


def _state_getter(self):
    view = self.__dict__.get("_torch_state_view")
    if view is None:
        view = get_install_context(jt).target_namespace.optim.Optimizer._OptState(self)
        object.__setattr__(self, "_torch_state_view", view)
    return view


def _lbfgs_type(base):
    if base is jt.optim.Optimizer:
        return LBFGS
    return type("LBFGS", (LBFGS, base), {"__module__": "torch.optim"})


def _step_with_closure(self, loss, retain_graph, closure, kwargs, native_kind):
    from jittor._runtime import step_capture   # only SGD hands its books to a capture
    if native_kind != "sgd": step_capture.refuse(f"a captured step cannot replay {native_kind}")
    _orig_step = get_install_context(jt).state["optimizer_native_api"]["steps"][native_kind]
    called_closure = False
    native_fsdp_loss = None
    if closure is None and callable(loss):
        closure = loss
        loss = None
    if closure is not None:
        loss = closure()
        called_closure = True
    _fsdp2_step = _load_fsdp2_for_optimizer(self)
    if _fsdp2_step is not None and _fsdp2_step.optimizer_has_fsdp_params(self):
        if loss is not None and not called_closure:
            native_fsdp_loss = loss
            loss.backward(retain_graph=retain_graph)
            loss = None
        if not _fsdp2_step.optimizer_step(
                self, None, retain_graph=retain_graph, native_kind=native_kind):
            raise NotImplementedError(
                f"FSDP2 optimizer step is not implemented for {type(self).__name__}")
        if not _fsdp2_step.optimizer_has_non_fsdp_params(self):
            if native_fsdp_loss is not None:
                try:
                    self.post_step()
                except EXPECTED as exc:
                    swallowed("torch/optimizers.py _step_torch_closure: self.post_step()", exc)
            else:
                jt.flags.node_order = 0
                object.__setattr__(
                    self, "_torch_backward_advanced_n_step", False)
            return loss if called_closure else None
        _fsdp2_step.clear_fsdp_optimizer_grads(self)
    torch_style = native_fsdp_loss is None and (loss is None or called_closure)
    if torch_style and not _optimizer_has_ready_grads(self):
        jt.flags.node_order = 0
        object.__setattr__(
            self, "_torch_backward_advanced_n_step", False)
        return loss if called_closure else None
    if not torch_style:
        if loss is None and not getattr(
                self, "_torch_backward_advanced_n_step", False):
            self.n_step = int(getattr(self, "n_step", 0)) + 1
        if loss is None:
            _advance_ready_param_steps(self)
        else:
            _advance_trainable_param_steps(self)
        out = _orig_step(self, loss, retain_graph=retain_graph)
        object.__setattr__(self, "_torch_backward_advanced_n_step", False)
        return out
    # Native SGD/RMSprop/Adan always post_step()->zero_grad(). Torch
    # keeps parameter.grad until the caller explicitly clears it.
    previous_post = self.__dict__.get("post_step", None)
    had_post = "post_step" in self.__dict__
    previous_step = int(getattr(self, "n_step", 0))
    if not getattr(self, "_torch_backward_advanced_n_step", False):
        self.n_step = previous_step + 1
    _advance_ready_param_steps(self)
    from .optimizers import replay_books_of_step
    replay_books_of_step(self)
    self.post_step = _torch_post_step
    try:
        # Torch optimizer updates do not record parameter writes. The closure
        # has already run with its own grad mode before this native step.
        with jt.no_grad():
            out = _orig_step(self, None, retain_graph=retain_graph)
    finally:
        object.__setattr__(self, "_torch_backward_advanced_n_step", False)
        if had_post:
            self.post_step = previous_post
        else:
            self.__dict__.pop("post_step", None)
    return loss if called_closure and out is None else out




def _adam_step(self, loss, retain_graph, closure, kwargs, decoupled_weight_decay):
    if decoupled_weight_decay:
        _validate_adamw_capturable_groups(self)
    native_fsdp_loss = None
    if closure is None and callable(loss):
        closure = loss
        loss = None
    if closure is not None:
        loss = closure()
    _fsdp2_step = _load_fsdp2_for_optimizer(self)
    if _fsdp2_step is not None and _fsdp2_step.optimizer_has_fsdp_params(self):
        if loss is not None and closure is None:
            native_fsdp_loss = loss
            loss.backward(retain_graph=retain_graph)
            loss = None
        if not _fsdp2_step.optimizer_step(
                self, None, retain_graph=retain_graph,
                native_kind="adamw" if decoupled_weight_decay else "adam"):
            raise NotImplementedError(
                f"FSDP2 optimizer step is not implemented for {type(self).__name__}")
        if not _fsdp2_step.optimizer_has_non_fsdp_params(self):
            if native_fsdp_loss is not None:
                try:
                    self.post_step()
                except EXPECTED as exc:
                    swallowed("torch/optimizers.py _adam_step_torch: self.post_step()", exc)
            else:
                jt.flags.node_order = 0
                object.__setattr__(
                    self, "_torch_backward_advanced_n_step", False)
            return native_fsdp_loss if native_fsdp_loss is not None else loss
        _fsdp2_step.clear_fsdp_optimizer_grads(self)
    self.pre_step(None if closure is not None and _optimizer_has_ready_grads(self) else loss, retain_graph)
    if not _optimizer_has_ready_grads(self):
        if native_fsdp_loss is not None:
            self.post_step()
        else:
            jt.flags.node_order = 0
            object.__setattr__(
                self, "_torch_backward_advanced_n_step", False)
        return native_fsdp_loss if native_fsdp_loss is not None else loss
    from jittor._runtime import step_capture
    if not getattr(self, "_torch_backward_advanced_n_step", False):
        self.n_step = int(getattr(self, "n_step", 0)) + 1
        step_capture.on_replay(lambda: setattr(self, "n_step", int(self.n_step) + 1))
    jt.flags.node_order = 1
    for pg in self.param_groups:
        lr = pg.get("lr", self.lr)
        eps = pg.get("eps", self.eps)
        weight_decay = pg.get("weight_decay", self.weight_decay)
        b0, b1 = pg.get("betas", self.betas)
        param_steps = _torch_param_steps(pg)
        # torch permits optimizers containing frozen or otherwise
        # unused parameters. loss.backward() then leaves the group
        # without gradients and step() must be a no-op, not KeyError.
        grads = pg.get("grads") or [None] * len(pg["params"])
        # Native adaptive optimizers intentionally retain FP32 master state for
        # half parameters. Torch initializes moments with zeros_like(parameter).
        # Adapt only fresh public Torch slots, including a group added later;
        # leave initialized or user-restored state and native policy unchanged.
        with jt.flag_scope(amp_reg=0):
            for i, (parameter, gradient) in enumerate(zip(pg["params"], grads)):
                if (int(param_steps[i]) != 0 or not parameter.requires_grad
                        or not isinstance(gradient, jt.Var)
                        or list(gradient.shape) != list(parameter.shape)):
                    continue
                for key in ("m", "values"):
                    buffer = pg[key][i]
                    if _jittor_dtype_name(buffer.dtype) != _jittor_dtype_name(parameter.dtype):
                        pg[key][i] = buffer.cast(parameter.dtype).stop_grad()
        # `use_acl` is an alias of `use_cuda` (see FLAG_ALIASES), so it is true
        # on a CUDA build as well and cannot say which backend is actually in
        # use. Asking the dispatcher does: `optim.adamw_fused` is registered
        # for "acl" only, so it answers None everywhere else. Keying off the
        # alias sent CUDA into the Ascend-only path, where the native
        # `fused_adamw` operator aborts with "only available through a mapped
        # backend" -- reachable from plain `torch.optim.AdamW(..., fused=True)`.
        fused_impl = None
        # torch's AdamW is a multi-tensor kernel unless told otherwise: its
        # default (fused=None, foreach=None) takes the foreach path on CUDA.
        # So the fused list update is taken unless the caller turned both
        # off, or asked for a variant it does not compute.
        fused = pg.get("fused", getattr(self, "fused", None))
        want_fused = (decoupled_weight_decay and fused is not False
                      and (fused is True or pg.get("foreach") is not False)
                      and not pg.get("amsgrad") and not pg.get("maximize"))
        active = []
        if want_fused:
            for i, (p, g, v, m) in enumerate(zip(
                    pg["params"], grads, pg["values"], pg["m"])):
                if not p.requires_grad or not isinstance(g, jt.Var) \
                        or list(g.shape) != list(p.shape):
                    continue
                active.append((p, m, v, g, int(param_steps[i])))
            if active:
                from jittor._runtime.dispatch import select_kernel
                fused_impl = select_kernel("optim.adamw_fused", active)
        if fused_impl is not None:
            stepped = []
            for i, (p, g, v, m) in enumerate(zip(
                    pg["params"], grads, pg["values"], pg["m"])):
                if not p.requires_grad or not isinstance(g, jt.Var) \
                        or list(g.shape) != list(p.shape):
                    continue
                param_steps[i] = int(param_steps[i]) + 1
                stepped.append(i)
            lr_arg = lr
            if step_capture.active():   # a replay advances these without step()
                lr_arg = step_capture.live_fused_update(
                    fused_impl, lr, param_steps, stepped, lambda g=pg: g.get("lr", self.lr),
                    lambda g=pg: (tuple(g.get("betas", self.betas)),
                                  g.get("weight_decay", self.weight_decay), g.get("eps", self.eps)),
                    self.__dict__.get("_amp_found_inf"), self)
            updates = fused_impl(
                active, lr_arg, b0, b1, weight_decay, eps)
            for (p, m, v, _, _), (new_p, new_m, new_v) in zip(
                    active, updates):  # fused_adamw keeps each input's dtype
                p.update(new_p)
                m.update(new_m)
                v.update(new_v)
                if p.is_stop_grad():
                    p.start_grad()
            continue
        step_capture.refuse("the per-parameter Adam update bakes its step count in")
        for i, (p, g, v, m) in enumerate(zip(
                pg["params"], grads, pg["values"], pg["m"])):
            was_trainable = bool(p.requires_grad)
            if not was_trainable or not isinstance(g, jt.Var) or list(g.shape) != list(p.shape):
                continue
            param_steps[i] = int(param_steps[i]) + 1
            _update_in_target_dtype(p, adam_update(
                p, g, v, m, lr=lr, eps=eps, weight_decay=weight_decay,
                betas=(b0, b1), step=param_steps[i],
                decoupled_weight_decay=decoupled_weight_decay,
                torch_math=True,
            ))
            try:
                if was_trainable and p.is_stop_grad():
                    p.start_grad()
            except EXPECTED as exc:
                swallowed("torch/optimizers.py _adam_step_torch: if was_trainable and p.is_stop_grad():", exc)
    # The torch loss.backward(); optimizer.step() spelling keeps grads
    # until an explicit zero_grad(). Preserve Jittor's historical
    # step(loss) behavior for callers using that native shorthand.
    if (loss is not None and closure is None) or native_fsdp_loss is not None:
        self.post_step()
    else:
        jt.flags.node_order = 0
    object.__setattr__(self, "_torch_backward_advanced_n_step", False)
    return native_fsdp_loss if native_fsdp_loss is not None else loss


def _torch_post_step():
    jt.flags.node_order = 0


# The Adam/AdamW update lives in its own module (it is most of the file's
# implementation); bound here under the names the rest of the package uses.
from .optimizer_adam import _adam_step, _update_in_target_dtype  # noqa: E402


def adam_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _adam_step(self, loss, retain_graph, closure, kwargs, False)


def adamw_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _adam_step(self, loss, retain_graph, closure, kwargs, True)


def sgd_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    _validate_sgd_groups(self)
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'sgd')


def rmsprop_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'rmsprop')


def adan_step(self, loss=None, retain_graph=False, closure=None, **kwargs):
    return _step_with_closure(self, loss, retain_graph, closure, kwargs, 'adan')


_STEP_APIS = {"sgd": sgd_step, "rmsprop": rmsprop_step, "adan": adan_step}

def adam_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_adam(self, params, lr, args, kwargs, 'Adam')


def _initialize_adam(self, params, lr, args, kwargs, name):
    # Native Adam has no foreach argument. AdamW's frontend validates its
    # options itself, so keep that argument on its existing initialization path.
    foreach = kwargs.pop('foreach', None) if name == 'Adam' else kwargs.get('foreach')
    result = _initialize_default(self, params, lr, args, kwargs, name)
    for group in self.param_groups:
        group.setdefault('foreach', foreach)
        group['_torch_adam_params'] = tuple(group['params'])
    return result

def adamw_init(self, params, lr=1e-3, *args, **kwargs):
    if len(args) < 3 and "weight_decay" not in kwargs: kwargs["weight_decay"] = 0.01
    return _initialize_adam(self, params, lr, args, kwargs, 'AdamW')


def rmsprop_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_default(self, params, lr, args, kwargs, 'RMSprop')


def adan_init(self, params, lr=1e-3, *args, **kwargs):
    return _initialize_default(self, params, lr, args, kwargs, 'Adan')


_INITIALIZER_APIS = {
    'Adam': adam_init,
    'AdamW': adamw_init,
    'RMSprop': rmsprop_init,
    'Adan': adan_init,
}
