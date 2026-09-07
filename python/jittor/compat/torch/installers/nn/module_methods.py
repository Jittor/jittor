import collections as _collections
import functools as _functools
import weakref
import os
import jittor as jt
from jittor import nn
from jittor.nn.backends import hooks as _backend_hooks
from jittor.backends.cuda.kernels.nn.rms_norm_training_cuda import _rms_norm_training_cuda
from jittor.backends.cuda.kernels.nn.rms_norm_cuda import _rms_norm_cuda
from ...context import registry_for
from ...fidelity import Fidelity, register_fidelity
from ...nested import _torch_register_leaf
from ...tensor_state import get_tensor_state
from ...types import _device_is_cpu, _device_is_cuda, _make_cpu_resident, _make_cuda_resident, device, dtype, _cuda_index_of
from ....diagnostics import EXPECTED, swallowed
from .... import fsdp_hooks as _fsdp_hooks

def _pipelining_from_environment():
    """Initial execution-pipelining threshold, read from the environment.

    An env var is the only way a program that never imports this module directly
    -- a benchmark harness, a downstream package's entry point -- can opt in.
    Anything unparseable leaves pipelining off rather than failing an import over
    a tuning knob.
    """
    try:
        return max(0, int(os.environ.get("JITTOR_EXECUTION_PIPELINING", "0")))
    except (TypeError, ValueError):
        return 0


# Jittor's own ``nn.Module`` methods, captured at import time. ``install``
# rebinds the very attributes these delegate to, so a late lookup would recurse;
# nothing in the install order replaces them before this module is imported, and
# ``test_the_captured_module_methods_are_still_native`` pins that rather than
# leaving it assumed.
_ORIG_MODULE_EXECUTE = nn.Module.execute
_ORIG_MODULE_DISPATCH_CALL = nn.Module._dispatch_call
_ORIG_MODULE_NAMED_PARAMETERS = nn.Module.named_parameters
_ORIG_MODULE_NAMED_BUFFERS = nn.Module.named_buffers
_ORIG_MODULE_NAMED_MODULES = nn.Module.named_modules
_ORIG_MODULE_LOAD_STATE_DICT = nn.Module.load_state_dict
_ORIG_MODULE_PARAMETERS = nn.Module.parameters


# torch models define forward(); jittor calls execute(). Make the base
# execute() delegate to a subclass-defined forward() so torch models run.
def _execute(self, *args, **kwargs):
    """Route a call to a subclass ``forward()`` when it defines one."""
    fwd = getattr(type(self), "forward", None)
    if fwd is not None and fwd is not _forward_alias:
        return fwd(self, *args, **kwargs)
    return _ORIG_MODULE_EXECUTE(self, *args, **kwargs)


def _forward_alias(self, *args, **kwargs):
    """``forward()`` for a subclass that only defines ``execute()``."""
    return self.execute(*args, **kwargs)


#: Per-class answer from ``_prefer_forward``. Module level so the answer is
#: computed once per class per process rather than once per install.
_dispatch_cache = {}


# Central dispatch fix: an HF module may SUBCLASS a jittor builtin (e.g.
# transformers OPTLearnedPositionalEmbedding(nn.Embedding)) and override
# forward() with a different signature. The builtin (Embedding) defines its
# own execute(), which MRO-shadows the patched base Module.execute above, so
# `module(...)` -> __call__ -> self.execute(...) lands on the builtin's
# execute() and never sees the subclass forward() -> TypeError.
#
# Decide per class whether the OWN forward() override should take precedence
# over the inherited builtin execute(): it should iff a real (non-alias)
# forward() is defined at an MRO position at least as derived as the nearest
# execute(). Conservative: classes that only define execute() (every native
# jittor module + jittor-native subclasses of builtins) keep calling
# execute() exactly as before; only a genuine, more-derived forward()
# override flips dispatch.
def _prefer_forward(cls):
    """Whether ``cls``'s own ``forward()`` should win over an inherited execute."""
    cached = _dispatch_cache.get(cls)
    if cached is not None:
        return cached
    fwd_idx = exec_idx = None
    for i, c in enumerate(cls.__mro__):
        d = c.__dict__
        if fwd_idx is None and "forward" in d and d["forward"] is not _forward_alias:
            fwd_idx = i
        if exec_idx is None and "execute" in d and d["execute"] is not _execute:
            exec_idx = i
    # forward() wins only if it exists and is no less derived than execute()
    result = fwd_idx is not None and (exec_idx is None or fwd_idx <= exec_idx)
    _dispatch_cache[cls] = result
    return result


def _acl_bfloat16_rms_norm(value, weight, epsilon):
    """ACL's bfloat16 RMS norm, in PyTorch's operand order, or None."""
    if not (
        getattr(jt.compiler, "has_acl", 0)
        and getattr(jt.flags, "use_acl", 0)
        and jt.flags.use_cuda
        and str(value.dtype) == "bfloat16"
        and str(weight.dtype) == "bfloat16"
    ):
        return None
    unit_weight = weight.__dict__.get("_torch_acl_rms_norm_unit_weight")
    if unit_weight is None or tuple(unit_weight.shape) != tuple(weight.shape):
        unit_weight = jt.ones(weight.shape, dtype="bfloat16")
        unit_weight.stop_grad()
        weight.__dict__["_torch_acl_rms_norm_unit_weight"] = unit_weight
    grouped = _backend_hooks.acl_grouped_bfloat16_rms_norm
    if grouped is not None:
        result = grouped(value, unit_weight, weight, epsilon)
        if result is not None:
            return result
    backend = _backend_hooks.rms_norm_cuda or _rms_norm_cuda
    normalized = backend(value, unit_weight, epsilon)
    if normalized is None:
        return None
    return weight * normalized


def _standard_rms_norm(self, args, kwargs):
    """A fused RMS norm for the standard single-argument shape, or None."""
    cls_name = type(self).__name__
    if (
        not cls_name.endswith("RMSNorm")
        or cls_name.endswith("RMSNormGated")
        or len(args) != 1
        or kwargs
        or "variance_epsilon" not in self.__dict__
    ):
        return None
    value = args[0]
    weight = getattr(self, "weight", None)
    if not isinstance(value, jt.Var) or not isinstance(weight, jt.Var):
        return None
    epsilon = self.__dict__["variance_epsilon"]
    pytorch_order = _acl_bfloat16_rms_norm(value, weight, epsilon)
    if pytorch_order is not None:
        return pytorch_order
    training_backend = (
        _backend_hooks.rms_norm_training_cuda or _rms_norm_training_cuda)
    fast = training_backend(value, weight, epsilon)
    if fast is None:
        backend = _backend_hooks.rms_norm_cuda or _rms_norm_cuda
        fast = backend(value, weight, epsilon)
    return fast


#: Execution-pipelining knob, read from the environment at import and re-read by
#: each install so a fresh install still honours the env var.
_pipeline_state = {"threshold": _pipelining_from_environment(), "mark": 0}


def set_execution_pipelining(pending_ops):
    ''' Launch the pending graph at module boundaries once it holds this many
    ops, instead of waiting for the next sync. 0 (the default) disables it.

    Returns the previous setting. See ``_maybe_pipeline`` for the trade.
    '''
    previous = _pipeline_state["threshold"]
    _pipeline_state["threshold"] = max(0, int(pending_ops))
    _pipeline_state["mark"] = jt.core.number_of_lived_ops()
    return previous


def get_execution_pipelining():
    ''' The current pending-op threshold; 0 when pipelining is off. '''
    return _pipeline_state["threshold"]


def _maybe_pipeline(result):
    """Optionally launch the graph built so far at a module boundary."""
    # A lazy graph reaches the device only at the next sync, so the GPU sits
    # idle for the whole of the Python-side construction: measured on
    # ViT-base, one contiguous ~6ms stall per step, immediately before the
    # first kernel of the forward. Launching the graph built so far at a
    # module boundary -- ``jt.sync`` does not wait for the device -- lets the
    # GPU start while Python keeps building.
    #
    # The cost is fusion: ops either side of a flush cannot fuse, which also
    # regroups floating-point accumulation, so results move by a rounding
    # step. Off unless asked for, and the threshold counts pending ops rather
    # than module calls, so leaf modules do not each trigger one.
    threshold = _pipeline_state["threshold"]
    if threshold <= 0:
        return result
    # Count ops added since the last flush, not ops alive: the live count
    # includes everything the graph still holds, so once it crossed the
    # threshold every later module call would flush and no two ops would ever
    # fuse.
    lived = jt.core.number_of_lived_ops()
    if lived - _pipeline_state["mark"] < threshold:
        if lived < _pipeline_state["mark"]:
            _pipeline_state["mark"] = lived
        return result
    target = result[0] if isinstance(result, (tuple, list)) and result else result
    if isinstance(target, jt.Var):
        jt.sync([target])
        _pipeline_state["mark"] = jt.core.number_of_lived_ops()
    return result


#: Modules whose parameters are already in the backward registry. Kept
#: outside the instance so it cannot show up in ``__dict__`` -- a module's
#: field set is part of its published shape, and tests pin it exactly.
_leaves_published = weakref.WeakSet()


def _dispatch_module_call(self, *args, **kwargs):
    """The call itself, once ``_call`` has done its bookkeeping.

    Separate from ``_call`` because FSDP2 needs it as a callable it can invoke
    around its own all-gather; ``_call`` hands it over as a
    ``functools.partial`` bound to the module.
    """
    # torch lets a module override forward per-INSTANCE (`self.forward = fn`,
    # used by vLLM's samplers / CustomOp dispatch). Honor it before class-level
    # dispatch.
    inst_fwd = self.__dict__.get("forward", None)
    if inst_fwd is not None and callable(inst_fwd):
        return inst_fwd(*args, **kwargs)
    rms_norm = _standard_rms_norm(self, args, kwargs)
    if rms_norm is not None:
        return rms_norm
    if _prefer_forward(type(self)):
        return type(self).forward(self, *args, **kwargs)
    return _ORIG_MODULE_DISPATCH_CALL(self, *args, **kwargs)


# Replace the DISPATCH, not ``__call__``: jittor's ``Module.__call__``
# runs the instance's hooks and then calls ``_dispatch_call``, so patching here
# keeps this dispatch inside the hooks -- where it was when a hook was installed
# by swapping ``cls.__call__`` (see Module._hooks).
def _call(self, *args, **kwargs):
    """``Module._dispatch_call``: leaf publication, FSDP2, then the call."""
    # Publish this module's own parameters as backward leaves, once.
    # `loss.backward()` has no graph walk to find them, so the registry is
    # filled by whoever iterates named_parameters() -- and a training loop
    # that never calls .parameters() (no optimizer, just backward and read
    # .grad) used to get None for every weight, silently, where torch fills
    # them. A module cannot contribute to a loss without being called, so
    # this is the earliest point that is both reliable and paid once.
    if self not in _leaves_published:
        try:
            _leaves_published.add(self)
        except TypeError as exc:
            swallowed("torch/installers/nn.py _call: _leaves_published.add(self)", exc)
        try:
            registry = get_tensor_state(jt).leaf_params
            for _leaf in _ORIG_MODULE_NAMED_PARAMETERS(self, recurse=False):
                _leaf = _leaf[1] if isinstance(_leaf, tuple) else _leaf
                if isinstance(_leaf, jt.Var) and _leaf.requires_grad:
                    registry[id(_leaf)] = _leaf
        except EXPECTED as exc:
            swallowed("torch/installers/nn.py _call: registry = getattr(jt, '_torch_leaf_params', None)", exc)

    # `_fsdp_state` is set only by fsdp2, so a module carrying one is proof
    # that fsdp2 was imported and therefore registered -- see
    # jittor/compat/fsdp_hooks.py for why this file must not import it.
    state = getattr(self, "_fsdp_state", None)
    if state is not None and getattr(state, "true_fsdp_initialized", False):
        _fsdp = _fsdp_hooks.provider()
        if _fsdp is not None:
            return _maybe_pipeline(_fsdp._execute_with_true_fsdp(
                self, _functools.partial(_dispatch_module_call, self),
                *args, **kwargs))
    return _maybe_pipeline(_dispatch_module_call(self, *args, **kwargs))


# torch's named_parameters/named_buffers/named_modules accept extra kwargs
# (remove_duplicate, prefix, recurse) and return iterators; jittor's take
# only `recurse` and return lists, with named_buffers defaulting recurse=
# False (torch defaults True). Wrap to be torch-compatible.
def _named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    """Torch's ``named_parameters``: an iterator, with prefix/dedup."""
    reg = get_tensor_state(jt).leaf_params
    seen = set()
    for name, v in _ORIG_MODULE_NAMED_PARAMETERS(self, recurse=recurse):
        if remove_duplicate and id(v) in seen:
            continue
        seen.add(id(v))
        # register trainable params as autograd leaves so the no-optimizer
        # loss.backward() path can populate their .grad (see parameters()).
        try:
            if isinstance(v, jt.Var) and v.requires_grad:
                reg[id(v)] = v
        except EXPECTED as exc:
            swallowed("torch/installers/nn.py _named_parameters: if isinstance(v, jt.Var) and v.requires_grad:", exc)
        yield (prefix + ("." if prefix else "") + name, v)


def _named_buffers(self, prefix="", recurse=True, remove_duplicate=True):
    """Torch's ``named_buffers``, which defaults ``recurse=True``."""
    seen = set()
    for name, v in _ORIG_MODULE_NAMED_BUFFERS(self, recurse=recurse):
        if remove_duplicate and id(v) in seen:
            continue
        seen.add(id(v))
        yield (prefix + ("." if prefix else "") + name, v)


def _named_modules(self, memo=None, prefix="", remove_duplicate=True):
    """Torch's ``named_modules``, accepting memo/prefix/remove_duplicate."""
    for item in _ORIG_MODULE_NAMED_MODULES(self):
        # jittor yields (name, module) pairs
        if isinstance(item, tuple) and len(item) == 2:
            name, mod = item
        else:
            name, mod = "", item
        yield (prefix + ("." if prefix and name else "") + name, mod)


# torch's Module.load_state_dict(state, strict=True, assign=False) accepts a
# `strict` kwarg and returns a namedtuple(missing_keys, unexpected_keys);
# jittor's takes only `params` and returns None. Wrap for torch callers
# (peft's set_peft_model_state_dict passes strict=False).
_IncompatibleKeys = _collections.namedtuple(
    "IncompatibleKeys", ["missing_keys", "unexpected_keys"])


def _find_state_target(root, key):
    """Resolve a dotted state-dict key to the live attribute it names."""
    obj = root
    for part in str(key).split("."):
        if isinstance(obj, nn.Sequential):
            if part in obj.layers:
                obj = obj.layers[part]
            elif str(part).isdigit() and int(part) in obj.layers:
                obj = obj.layers[int(part)]
            else:
                return None
        elif hasattr(obj, part):
            obj = getattr(obj, part)
        else:
            return None
    return obj


def _state_source_to_var(value):
    """Coerce one state-dict value to a Jittor Var."""
    if isinstance(value, jt.Var):
        return value
    try:
        return jt.array(value.cpu().detach().numpy())
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _state_source_to_var: return jt.array(value.cpu().detach().numpy())", exc)
        return jt.array(value)


def _preserve_target_dtypes_for_load(root, state_dict):
    """Cast each source value to the dtype of the live destination."""
    # torch.load_state_dict(assign=False), the default used by TRELLIS.2,
    # copies checkpoint values into existing parameters/buffers and keeps
    # the destination dtype.  Jittor's native load replaces through update(),
    # so a bf16 target can be widened to fp32 when the loader had to widen a
    # BF16 safetensor through numpy. Cast the source to the live target dtype
    # before delegating to native load_state_dict.
    if not isinstance(state_dict, dict):
        return state_dict
    converted = None
    for key, value in state_dict.items():
        target = _find_state_target(root, key)
        if not isinstance(target, jt.Var):
            continue
        src = _state_source_to_var(value)
        if not isinstance(src, jt.Var):
            continue
        if src.shape != target.shape:
            continue
        target_dtype = str(target.dtype)
        if str(src.dtype) == target_dtype:
            continue
        if converted is None:
            converted = dict(state_dict)
        converted[key] = src.cast(target_dtype)
    return state_dict if converted is None else converted


def _state_dict_key_diff(root, state_dict):
    """(missing, unexpected, mismatched) between a module and a state dict.

    This used to be nothing at all: the wrapper returned
    ``_IncompatibleKeys([], [])`` unconditionally, so ``strict=True`` --
    the whole point of which is to reject a checkpoint that does not match
    the model -- accepted a checkpoint with every key wrong and left the
    model at its random initialisation without a word.
    """
    own = {}
    try:
        for name, value in root.state_dict().items():
            own[str(name)] = value
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _state_dict_key_diff: for name, value in root.state_dict().items():", exc)
        return [], [], []
    given = list(state_dict.keys()) if hasattr(state_dict, "keys") else []
    given_set = set(str(k) for k in given)
    missing = [k for k in own if k not in given_set]
    unexpected = [str(k) for k in given if str(k) not in own]
    mismatched = []
    for key in given:
        target = own.get(str(key))
        if not isinstance(target, jt.Var):
            continue
        src = state_dict[key]
        src_shape = getattr(src, "shape", None)
        if src_shape is None:
            continue
        if tuple(int(d) for d in src_shape) != tuple(int(d) for d in target.shape):
            mismatched.append((str(key), tuple(int(d) for d in src_shape),
                               tuple(int(d) for d in target.shape)))
    # torch never reports num_batches_tracked as missing for modules that
    # do not keep one; jittor's BatchNorm has no such buffer at all.
    missing = [k for k in missing if not k.endswith("num_batches_tracked")]
    return missing, unexpected, mismatched


def _load_state_dict(self, state_dict, strict=True, assign=False):
    """Torch's ``load_state_dict``: honours ``strict`` and returns the keys."""
    missing, unexpected, mismatched = _state_dict_key_diff(self, state_dict)
    # torch raises on a shape mismatch whatever `strict` says: the value
    # cannot be copied at all.  jittor's load_parameters only LOG.e'd it.
    if mismatched:
        raise RuntimeError(
            "Error(s) in loading state_dict for %s:\n\t%s"
            % (type(self).__name__,
               "\n\t".join(
                   "size mismatch for %s: copying a param with shape %s "
                   "from checkpoint, the shape in current model is %s."
                   % (k, list(a), list(b)) for k, a, b in mismatched)))
    if strict and (missing or unexpected):
        parts = []
        if missing:
            parts.append("Missing key(s) in state_dict: %s. "
                         % ", ".join('"%s"' % k for k in sorted(missing)))
        if unexpected:
            parts.append("Unexpected key(s) in state_dict: %s. "
                         % ", ".join('"%s"' % k for k in sorted(unexpected)))
        raise RuntimeError(
            "Error(s) in loading state_dict for %s:\n\t%s"
            % (type(self).__name__, "\n\t".join(parts)))
    # preserve trainable flags: jittor assign can flip stop_grad
    trainable = set()
    try:
        for n, p in self.named_parameters():
            if p.requires_grad:
                trainable.add(n)
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _load_state_dict: for n, p in self.named_parameters():", exc)
    load_state = state_dict if assign else _preserve_target_dtypes_for_load(self, state_dict)
    if unexpected:
        # jittor's load_parameters LOG.w's on every unknown key; drop them
        # here so a strict=False load stays quiet, exactly like torch.
        load_state = {k: v for k, v in load_state.items()
                      if str(k) not in set(unexpected)}
    _ORIG_MODULE_LOAD_STATE_DICT(self, load_state)
    try:
        for n, p in self.named_parameters():
            if n in trainable and p.is_stop_grad():
                p.start_grad()
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _load_state_dict: for n, p in self.named_parameters():", exc)
    return _IncompatibleKeys(missing, unexpected)


# torch's Module.parameters() returns an *iterator*; peft does
# `next(model.parameters())`. jittor returns a list (needed for len()/
# indexing by optimizers). Return a list subclass that is also an iterator
# so both `next(...)` and `len(...)`/indexing work.
class _ParamList(list):
    """A list that is also its own iterator, for ``next(model.parameters())``."""

    def __iter__(self):
        return list.__iter__(self)

    def __next__(self):
        it = getattr(self, "_it", None)
        if it is None:
            it = self._it = list.__iter__(self)
        return next(it)


# Register every trainable parameter as an autograd "leaf" the first time a
# module's params are enumerated. torch code reads param.grad only after
# enumerating params (optimizer construction, gradient clipping, gradcheck,
# manual inspection all call parameters()/named_parameters() first), so this
# is the reliable hook that lets the optimizer-free loss.backward() path
# populate param.grad. jittor params are trainable-by-default and
# almost never pass through the requires_grad setter, which is why the prior
# registry stayed empty (bert: 0/39 grads exposed). Enumeration is also the
# *leak-safe* hook: only declared parameters are captured -- never transient
# forward activations, which a Module.__setattr__ hook would wrongly retain
# and leak one Var per step. Idempotent (id-keyed); skips frozen params so
# their .grad stays None like torch.
def _register_leaf_params(params):
    """Record trainable parameters as backward leaves, idempotently."""
    try:
        reg = get_tensor_state(jt).leaf_params
        for p in params:
            if isinstance(p, jt.Var) and p.requires_grad:
                reg[id(p)] = p
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _register_leaf_params: register autograd leaves", exc,
                  "these parameters will not receive .grad from a "
                  "loss.backward() that runs without an optimizer")


def _parameters(self, recurse=True):
    """Torch's ``parameters()``: iterable *and* indexable."""
    pl = _ORIG_MODULE_PARAMETERS(self, recurse=recurse)
    _register_leaf_params(pl)
    return _ParamList(pl)


# torch's Module.train(mode=True)/eval() take a mode arg; jittor's train()
# takes none. Wrap to accept it and toggle jittor's real training flag.
#
# The flag that controls layers like Dropout/BatchNorm is `is_train` -- an
# instance attribute read by jittor.nn.Dropout.execute. `is_training` is a
# *method* and `training` a *property*, so they must NEVER be assigned a
# bool (the old code did `m.is_training = False`, which both shadowed the
# method and failed to flip the flag the layers actually read). We set
# `is_train` recursively on every submodule. We deliberately do NOT touch
# parameter stop-grad state (torch's .eval() leaves requires_grad alone),
# so this is purely a mode flip with no gradient side effects.
def _set_is_train(self, mode):
    """Flat sweep of ``is_train`` over this module and its submodules."""
    mode = bool(mode)
    try:
        mods = self.modules() if hasattr(self, "modules") else [self]
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _set_is_train: mods = self.modules() if hasattr(self, 'modules') else ...", exc)
        mods = [self]
    for m in mods:
        try:
            m.is_train = mode
        except (AttributeError, TypeError) as exc:
            swallowed("torch/installers/nn.py _set_is_train: m.is_train = mode", exc)


def _train(self, mode=True):
    """Torch's ``train(mode)``, recursing through each child's own ``train``."""
    # torch semantics: set this module's flag, then recurse into DIRECT
    # children calling each child's .train(mode) so overridden train()
    # methods run (e.g. e2cnn's R2Conv.train() rebuilds/discards its cached
    # filter; a flat is_train sweep silently bypasses it, leaving stale or
    # empty filters and zero output). For ordinary modules this is
    # behaviourally identical to the old flat sweep.
    mode = bool(mode)
    try:
        self.is_train = mode
    except (AttributeError, TypeError) as exc:
        swallowed("torch/installers/nn.py _train: self.is_train = mode", exc)
    kids = None
    try:
        kids = list(self.children())
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _train: kids = list(self.children())", exc)
        kids = None
    if kids is None:
        _set_is_train(self, mode)          # fallback: flat sweep
        return self
    for child in kids:
        tr = getattr(child, "train", None)
        if callable(tr):
            try:
                tr(mode)
                continue
            except EXPECTED as exc:
                swallowed("torch/installers/nn.py _train: tr(mode)", exc)
        _set_is_train(child, mode)
    return self


def _eval(self):
    """Torch's ``eval()``: ``train(False)``, with no gradient side effects."""
    return _train(self, False)


_MODULE_FLOAT_DTYPES = ("float16", "bfloat16", "float32", "float64")


def _module_cast_var_if_needed(v, ds, copy=False):
    """Cast ``v`` to ``ds``, or return it as-is when that is already its dtype."""
    if copy or str(v.dtype) != ds:
        return v.cast(ds)
    return v


def _module_cast_float_dtype(self, ds):
    """Cast every floating parameter of this module in place."""
    if getattr(type(self), "_frontend_tensor_type", None) is not None:
        return _module_replace_vars(
            self, _functools.partial(_module_to_conversion, ds, None, False))
    if ds is not None and ds in _MODULE_FLOAT_DTYPES:
        for p in self.parameters():
            if p.dtype.is_float() if hasattr(p.dtype, "is_float") else ("float" in str(p.dtype)):
                new_p = _module_cast_var_if_needed(p, ds)
                if new_p is not p:
                    p.assign(new_p)
    return self


def _module_replace_vars(self, convert):
    """Run ``convert`` over every parameter/buffer this module tree holds."""
    converted = {}
    try:
        modules = list(self.modules()) if hasattr(self, "modules") else [self]
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _module_replace_vars: modules = list(self.modules()) if hasattr(self, 'module...", exc)
        modules = [self]
    if not modules or modules[0] is not self:
        modules.insert(0, self)
    if getattr(type(self), "_frontend_tensor_type", None) is not None:
        for module in modules:
            # The role accessor also covers ParameterList/ParameterDict, whose
            # values do not live as ordinary public instance attributes.
            for name, value, role in tuple(module._var_roles()):
                if role not in ("parameter", "buffer", "non_persistent_buffer"):
                    continue
                identity = id(value)
                if identity not in converted:
                    replacement = convert(value)
                    if role == "parameter":
                        if replacement is not value:
                            # Module conversion changes parameter storage, not
                            # its identity or its position as a graph leaf.
                            on_cpu = replacement.location() == "cpu"
                            value.assign(replacement.detach())
                            if on_cpu:
                                _make_cpu_resident(value, inplace=True)
                        replacement = value
                        gradient = getattr(value, "_torch_grad", None)
                        if isinstance(gradient, jt.Var):
                            new_gradient = convert(gradient)
                            if new_gradient is not gradient:
                                gradient.assign(new_gradient)
                    elif replacement is not value:
                        for attribute in ("is_buffer", "persistent"):
                            if hasattr(value, attribute):
                                setattr(replacement, attribute, getattr(value, attribute))
                    converted[identity] = replacement
                replacement = converted[identity]
                if replacement is not value:
                    setattr(module, str(name), replacement)
        return self
    seen = set()
    for module in modules:
        mid = id(module)
        if mid in seen:
            continue
        seen.add(mid)
        attrs = []
        if hasattr(module, "params"):
            attrs.append(("params", getattr(module, "params")))
        attrs.append(("__dict__", getattr(module, "__dict__", {})))
        for _container_name, container in attrs:
            if not isinstance(container, dict):
                continue
            buffer_names = getattr(module, "_buffer_names", set())
            for name, value in list(container.items()):
                if isinstance(value, jt.Var):
                    if _container_name == "__dict__":
                        is_public_param = not (isinstance(name, str) and name.startswith("_"))
                        is_buffer = getattr(value, "is_buffer", False) or name in buffer_names
                        if not (is_public_param or is_buffer):
                            continue
                    vid = id(value)
                    if vid in converted:
                        new_value = converted[vid]
                    else:
                        new_value = convert(value)
                        converted[vid] = new_value
                        if new_value is not value:
                            try:
                                new_value.persistent = getattr(value, "persistent")
                            except (AttributeError, TypeError) as exc:
                                swallowed("torch/installers/nn.py _module_replace_vars: new_value.persistent = getattr(value, 'persistent')", exc)
                            try:
                                new_value.is_buffer = getattr(value, "is_buffer")
                            except (AttributeError, TypeError) as exc:
                                swallowed("torch/installers/nn.py _module_replace_vars: new_value.is_buffer = getattr(value, 'is_buffer')", exc)
                            try:
                                new_value._torch_grad = getattr(value, "_torch_grad")
                            except (AttributeError, TypeError) as exc:
                                swallowed("torch/installers/nn.py _module_replace_vars: new_value._torch_grad = getattr(value, '_torch_grad')", exc)
                            try:
                                if value.is_stop_grad() and not new_value.is_stop_grad():
                                    new_value.stop_grad()
                                elif (not value.is_stop_grad()) and new_value.is_stop_grad():
                                    new_value.start_grad()
                                    _torch_register_leaf(new_value)
                            except EXPECTED as exc:
                                swallowed("torch/installers/nn.py _module_replace_vars: if value.is_stop_grad() and not new_value.is_stop_grad():", exc)
                            try:
                                reg = get_tensor_state(jt).leaf_params
                                if vid in reg:
                                    reg.pop(vid, None)
                                    if not new_value.is_stop_grad():
                                        reg[id(new_value)] = new_value
                            except EXPECTED as exc:
                                swallowed("torch/installers/nn.py _module_replace_vars: reg = getattr(jt, '_torch_leaf_params', None)", exc)
                    if new_value is value:
                        continue
                    container[name] = new_value
    return self

def _module_to_conversion(ds, dev, copy, v):
    """Convert one Var for ``_module_to``: dtype cast, then residency.

    Split out of ``_module_to`` so this file holds no closures; ``_module_to``
    hands the decoded (dtype, device, copy) triple over with a partial.
    """
    out = v
    if ds is not None and ds in _MODULE_FLOAT_DTYPES:
        is_float = v.dtype.is_float() if hasattr(v.dtype, "is_float") else ("float" in str(v.dtype))
        if is_float:
            out = _module_cast_var_if_needed(out, ds, copy=copy)
    if _device_is_cpu(dev):
        out = _make_cpu_resident(out, inplace=(out is v))
    elif _device_is_cuda(dev):
        src_index = getattr(v, "device_id", -1)
        out = _make_cuda_resident(out, force=True, inplace=(out is v))
        # A bare .to("cuda") must not drag a parameter off the device
        # it is already on; see _move_to_cuda_index.
        idx = _cuda_index_of(dev)
        if idx is None and src_index is not None and src_index >= 0:
            idx = src_index
        if idx is not None and isinstance(out, jt.Var):
            cur = getattr(out, "device_id", -1)
            if cur >= 0 and cur != int(idx):
                moved = out.to_device(int(idx))
                if out is v:
                    # torch's Module.to is in place: the Parameter
                    # object keeps its identity -- an optimizer or a
                    # state_dict already holding it must keep working
                    # -- and only its storage moves.
                    v.assign(moved)
                    out = v
                else:
                    out = moved
    return out


def _module_to(self, *args, **kwargs):
    """Torch's ``Module.to``: cast floating tensors, migrate residency."""
    # torch Module.to(device/dtype/...) casts floating tensors and migrates
    # tensor residency when an explicit cpu/cuda device is requested.
    ds = None
    dev = kwargs.get("device")
    copy = bool(kwargs.get("copy", False))
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, dtype):
            ds = a.name
        elif isinstance(a, device):
            dev = a
        elif isinstance(a, jt.Var):
            ds = str(a.dtype)
            dev = a.device
        elif isinstance(a, str):
            bare = a.replace("torch.", "")
            if bare in dtype._registry:
                ds = bare
            elif bare.split(":")[0] in ("cpu", "cuda", "npu"):
                dev = bare
    if _device_is_cuda(dev):
        jt.flags.use_cuda = 1
    if dev is not None or ds is not None:
        return _module_replace_vars(
            self, _functools.partial(_module_to_conversion, ds, dev, copy))
    return self


def _module_to_empty(self, *, device, recurse=True):
    """Torch's ``to_empty``; Jittor has no meta storage, so values survive."""
    # Jittor does not expose meta storage. Models are already materialized,
    # so preserve their values while honoring the requested residency.
    return _module_to(self, device=device)


def _module_cuda(self, dev=None):
    """Torch's ``Module.cuda``, optionally pinned to one device index."""
    return _module_to(self, device("cuda", dev) if isinstance(dev, int) else "cuda")


def _module_npu(self, dev=None):
    """Torch's ``Module.npu``, optionally pinned to one device index."""
    return _module_to(self, device("npu", dev) if isinstance(dev, int) else "npu")


def _module_cpu(self):
    """Torch's ``Module.cpu``."""
    return _module_to(self, "cpu")


def _module_float(self):
    """Torch's ``Module.float``."""
    return _module_cast_float_dtype(self, "float32")


def _module_double(self):
    """Torch's ``Module.double``."""
    return _module_cast_float_dtype(self, "float64")


def _module_half(self):
    """Torch's ``Module.half``."""
    return _module_cast_float_dtype(self, "float16")


# torch's zero_grad() clears each param's .grad so the next backward starts
# fresh; the optimizer-free backward path accumulates with += (matching
# torch), so a real reset is required. The prior no-op left grads silently
# accumulating across steps. Clear the torch-exposed grad and, when an
# optimizer is bridged, delegate to its zero_grad as well.
def _zero_grad(self, set_to_none=True):
    """Torch's ``zero_grad``: a real reset, honoring ``set_to_none``.

    ``set_to_none=False`` must leave ``.grad`` as an all-zero tensor of the same
    shape and dtype, which is what torch 2.12.1 does. Dropping it to None
    instead is silently wrong rather than loud: gradient clipping and gradient
    accumulation are both written as ``if p.grad is not None``, so every
    parameter gets skipped and no error is ever raised.
    """
    # The bridged optimizer runs first: its zero_grad clears the torch-visible
    # .grad as a side effect, so doing it afterwards would undo the zero tensors
    # that set_to_none=False is required to leave behind.
    from ...tensor_state import compatibility_owner
    opt = getattr(compatibility_owner(jt), "_current_optimizer", None)
    if opt is not None:
        try:
            opt.zero_grad()
        except EXPECTED as exc:
            swallowed("torch/installers/nn.py _zero_grad: opt.zero_grad()", exc)
    try:
        for p in self.parameters():
            grad = getattr(p, "_torch_grad", None)
            if set_to_none:
                if grad is not None:
                    object.__setattr__(p, "_torch_grad", None)
            elif grad is not None:
                object.__setattr__(
                    p, "_torch_grad", jt.zeros(grad.shape, dtype=grad.dtype))
    except EXPECTED as exc:
        swallowed("torch/installers/nn.py _zero_grad: for p in self.parameters():", exc)
    return None


def _buffers(self, recurse=True):
    """Torch's ``buffers()``, derived from ``named_buffers``."""
    return [v for _, v in self.named_buffers()]


def _get_submodule(self, target):
    """Torch's ``get_submodule``, resolving a dotted path."""
    mod = self
    for part in target.split("."):
        if part:
            mod = getattr(mod, part)
    return mod


def _get_parameter(self, target):
    """Torch's ``get_parameter``: a parameter, never a buffer."""
    mod = self
    parts = target.split(".")
    for part in parts[:-1]:
        if part:
            mod = getattr(mod, part)
    leaf = parts[-1]
    if not hasattr(mod, leaf):
        raise AttributeError(f"`{target}` is not a parameter")
    v = getattr(mod, leaf)
    # a parameter is a trainable Var directly attached to the module
    if isinstance(v, jt.Var) and not v.is_stop_grad():
        return v
    if isinstance(v, jt.Var):
        # could still be a (frozen) parameter; distinguish from buffers
        names = {n for n, _ in self.named_parameters()}
        if target in names:
            return v
    raise AttributeError(f"`{target}` is not a parameter")


def _get_buffer(self, target):
    """Torch's ``get_buffer``: a buffer, never a parameter."""
    mod = self
    parts = target.split(".")
    for part in parts[:-1]:
        if part:
            mod = getattr(mod, part)
    leaf = parts[-1]
    if not hasattr(mod, leaf):
        raise AttributeError(f"`{target}` is not a buffer")
    v = getattr(mod, leaf)
    names = {n for n, _ in self.named_buffers()}
    if isinstance(v, jt.Var) and target in names:
        return v
    raise AttributeError(f"`{target}` is not a buffer")


def _register_parameter(self, name, param):
    """Torch's ``register_parameter``: an explicit "this is a parameter"."""
    # This is torch's explicit "this attribute is a parameter" call, so
    # it carries the same weight as wrapping the value in nn.Parameter --
    # a torch-authored class otherwise only registers by that wrapper, and
    # the value handed here need not have gone through it (vLLM builds its
    # typed ModelWeightParameter through its own metaclass __call__).
    if isinstance(param, jt.Var):
        param._is_torch_parameter = True
    setattr(self, name, param)


def _module_type(self, dst_type=None):
    """Torch's ``Module.type``, which Jittor has nothing to do for."""
    return self


# torch's nn.Module keeps `_non_persistent_buffers_set`, a set of the
# *immediate* (non-recursive) buffer attribute names that were registered
# with persistent=False. transformers' from_pretrained reads it via
# `named_non_persistent_buffers()` (parent._non_persistent_buffers_set).
# jittor instead tags each buffer Var with `.persistent`; derive the set
# from that. It's a property so it stays correct as buffers are (de)added.
def _nonpersist_set(self):
    """The immediate buffer names registered with ``persistent=False``."""
    out = set()
    for k, v in self.__dict__.items():
        if (isinstance(k, str) and not k.startswith("_")
                and isinstance(v, jt.Var)
                and getattr(v, "is_buffer", False)
                and not getattr(v, "persistent", True)):
            out.add(k)
    return out


# Fidelity for the promoted Module methods. Every entry is APPROXIMATE: these
# wrap jittor's own Module, whose parameter/buffer bookkeeping is attribute-based
# rather than torch's ordered `_parameters`/`_buffers` dicts, so ordering and
# edge-case reporting can differ even where values agree. Only claim EXACT with
# evidence, per the cohort-promotion skill.
register_fidelity(
    "torch.nn.Module.execute", _execute, Fidelity.APPROXIMATE,
    "Routes to a torch-style forward() when the subclass defines one, else to "
    "jittor's execute(). Hooks fire in registration order. Does not implement "
    "torch's full _call_impl (no global forward pre-hooks, no __torch_function__ "
    "dispatch).")
register_fidelity(
    "torch.nn.Module.named_parameters", _named_parameters, Fidelity.APPROXIMATE,
    "Yields (name, Var) for trainable Vars reachable by attribute walk. "
    "recurse= and prefix= honored; remove_duplicate= is accepted and always "
    "de-duplicates. Order follows attribute definition order, which matches "
    "torch for modules built in __init__ but is not guaranteed for modules "
    "assembled dynamically.")
register_fidelity(
    "torch.nn.Module.named_buffers", _named_buffers, Fidelity.APPROXIMATE,
    "Yields (name, Var) for buffer-tagged Vars, including non-persistent ones, "
    "matching torch. Buffer identity comes from jittor's .is_buffer tag rather "
    "than a separate _buffers dict.")
register_fidelity(
    "torch.nn.Module.named_modules", _named_modules, Fidelity.APPROXIMATE,
    "Yields ('', self) first then descendants, as torch does. memo= and "
    "remove_duplicate= are honored. Container children are enumerated through "
    "jittor's named_modules, so ordering within a ModuleList follows insertion.")
register_fidelity(
    "torch.nn.Module.load_state_dict", _load_state_dict, Fidelity.APPROXIMATE,
    "Returns a namedtuple with missing_keys/unexpected_keys like torch, and "
    "preserves each target parameter's existing dtype so loading a float32 "
    "checkpoint into a half module stays half. assign= is accepted but always "
    "copies into the existing Var so parameter identity survives.")
register_fidelity(
    "torch.nn.Module.parameters", _parameters, Fidelity.APPROXIMATE,
    "Returns a list-like that also registers its members as autograd leaves, so "
    "an optimizer built from it can see gradients. torch returns a lazy "
    "generator; this is materialized, so mutating the module while iterating "
    "does not raise as it would in torch.")
register_fidelity(
    "torch.nn.Module.train", _train, Fidelity.APPROXIMATE,
    "Sets is_train recursively and returns self. Does not distinguish torch's "
    "per-module training flag from jittor's is_train, which some jittor-native "
    "layers read directly.")
register_fidelity(
    "torch.nn.Module.eval", _eval, Fidelity.APPROXIMATE,
    "Equivalent to train(False); returns self. Same caveat as train().")
register_fidelity(
    "torch.nn.Module.to", _module_to, Fidelity.APPROXIMATE,
    "In place like torch: casts only floating-point Vars (integer buffers "
    "survive to(float64)) and migrates residency for cpu/cuda/npu, keeping "
    "parameter object identity so a live optimizer or state_dict stays valid. "
    "memory_format= and non_blocking= are accepted and ignored; there is no meta "
    "device.")
register_fidelity(
    "torch.nn.Module.to_empty", _module_to_empty, Fidelity.APPROXIMATE,
    "Honors the requested residency but preserves values instead of leaving "
    "them uninitialized, because jittor exposes no meta storage. Callers that "
    "rely on torch's uninitialized result see initialized data.")
register_fidelity(
    "torch.nn.Module.cuda", _module_cuda, Fidelity.APPROXIMATE,
    "Delegates to to(); a bare .cuda() keeps a parameter on the device it is "
    "already on rather than forcing device 0.")
register_fidelity(
    "torch.nn.Module.zero_grad", _zero_grad, Fidelity.APPROXIMATE,
    "Honors set_to_none: True (the torch default) clears .grad to None, False "
    "leaves an all-zero tensor of the same shape and dtype. Also forwards to a "
    "bridged optimizer's zero_grad when one is active. Returns None.")


def _install_module_methods(nn, registry=None):
    """Bind the torch-compatible ``nn.Module`` methods; all are module level.

    Every object bound here is defined at module scope above, so a single Module
    method is a single importable function that can be unit-tested on its own.
    The ``_ORIG_MODULE_*`` handles the wrappers delegate to are captured at
    import time for the same reason ``installers/tensor.py`` captures its
    natives: install replaces the attributes they read, so a late lookup would
    recurse.
    """
    registry_for(jt, registry)
    M = nn.Module

    # A fresh install re-reads the pipelining env var and forgets any threshold a
    # previous one had been asked for.
    _pipeline_state["threshold"] = _pipelining_from_environment()
    _pipeline_state["mark"] = 0

    M.execute = _execute
    if not hasattr(M, "forward"):
        M.forward = _forward_alias
    M._dispatch_call = _call
    M.set_execution_pipelining = staticmethod(set_execution_pipelining)
    M.get_execution_pipelining = staticmethod(get_execution_pipelining)
    M.named_parameters = _named_parameters
    M.named_buffers = _named_buffers
    M.named_modules = _named_modules
    M.load_state_dict = _load_state_dict
    M.parameters = _parameters
    M.train = _train
    M.eval = _eval
    M.to = _module_to
    M.to_empty = _module_to_empty
    M.cuda = _module_cuda
    M.npu = _module_npu
    M.cpu = _module_cpu
    M.zero_grad = _zero_grad
    # Guarded: jittor owns `half` natively and its version wins, exactly as
    # before this promotion.
    if not hasattr(M, "float"):
        M.float = _module_float
    if not hasattr(M, "double"):
        M.double = _module_double
    if not hasattr(M, "half"):
        M.half = _module_half
    if not hasattr(M, "buffers"):
        M.buffers = _buffers
    if not hasattr(M, "get_submodule"):
        M.get_submodule = _get_submodule
    if not hasattr(M, "get_parameter"):
        M.get_parameter = _get_parameter
    if not hasattr(M, "get_buffer"):
        M.get_buffer = _get_buffer
    if not hasattr(M, "register_parameter"):
        M.register_parameter = _register_parameter
    if not hasattr(M, "type"):
        M.type = _module_type
    if not isinstance(M.__dict__.get("_non_persistent_buffers_set"), property):
        M._non_persistent_buffers_set = property(_nonpersist_set)
