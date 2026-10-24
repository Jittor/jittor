"""Persistent Adam state; shared counters, not replacement Adam mathematics.

SOURCE_ONLY candidate for compat/torch/optimizer_state.py. No eager imports of
Jittor or NumPy: the installation provides its authenticated public namespace.
"""
from collections import defaultdict
from collections.abc import Mapping
import math


def _count(value):
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0 or numeric != int(numeric):
        raise ValueError("optimizer step must be a non-negative integer")
    return int(numeric)


class PublicStepPolicy:
    """Torch's real scalar policy; no fabricated device/type metadata."""
    def __init__(self, torch):
        self.torch = torch

    def read(self, value):
        if not isinstance(value, self.torch.Tensor):
            raise TypeError("Adam step must be a real public Tensor")
        if tuple(value.shape) != () or value.requires_grad:
            raise ValueError("Adam step must be a detached scalar Tensor")
        if value.dtype not in (self.torch.float32, self.torch.float64):
            raise ValueError("Adam step dtype must be float32 or float64")
        return _count(value.item())

    def validate_group(self, group):
        for option in ("capturable", "differentiable", "amsgrad", "maximize"):
            if group.get(option, False):
                raise NotImplementedError("Adam state policy does not admit " + option)
        fused = group.get("fused", None)
        if fused is not None and type(fused) is not bool:
            raise TypeError("Adam fused must be bool or None")
        foreach = group.get("foreach", None)
        if foreach is not None and type(foreach) is not bool:
            raise TypeError("Adam foreach must be bool or None")
        if fused is True and foreach is True:
            raise RuntimeError("fused and foreach cannot both be True")

    def create(self, parameter, group, value):
        self.validate_group(group)
        fused = group.get("fused", None)
        dtype = (self.torch.float32 if fused else
                 self.torch.float64 if self.torch.get_default_dtype() == self.torch.float64
                 else self.torch.float32)
        options = {"dtype": dtype}
        if fused:
            options["device"] = parameter.device
        # The ordinary branch intentionally follows the actual default-device
        # context, exactly like torch2.8. Never force this scalar onto CUDA.
        return self.torch.tensor(float(_count(value)), **options)

    def assign(self, tensor, value):
        self.read(tensor)
        with self.torch.no_grad():
            tensor.fill_(float(_count(value)))

    def loaded(self, parameter, group, value):
        self.validate_group(group)
        if not isinstance(value, self.torch.Tensor):
            # Older checkpoints use a number; current Adam.__setstate__ also
            # migrates that legacy spelling into a scalar Tensor.
            return self.create(parameter, group, _count(value))
        self.read(value)
        if group.get("capturable", False):
            raise NotImplementedError("capturable Adam state restore is not admitted")
        if group.get("fused", False):
            return value.to(dtype=self.torch.float32, device=parameter.device)
        return value  # Official nonfused/noncapturable step load policy.


class AdamStepSlots(list):
    """Existing integer-counter API backed by the canonical scalar objects.

    The list's zero placeholders are NEVER numerical authority. __iter__ and
    indexed reads derive counts from the persistent public Tensor, including a
    caller's genuine in-place edit. All old increment sites therefore converge
    without adding a second independently advancing integer counter.
    """
    def __init__(self, owner, group):
        self.owner, self.group = owner, group
        list.__init__(self, [0] * len(group["params"]))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self))) ]
        index = range(len(self))[index]
        return self.owner.count(self.group, index)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            raise TypeError("Adam counter slices cannot replace canonical state")
        index = range(len(self))[index]
        self.owner.set_count(self.group, index, value)

    def resize(self):
        list.__setitem__(self, slice(None), [0] * len(self.group["params"]))


class AdamExecutionGroup:
    """PRIVATE execution handle, never installed in optimizer.param_groups.

    Public config is the original actual dict, read live. Tensor buffers and
    bookkeeping occupy separate private storage, even if a caller's custom
    public config happens to use the same spelling (e.g. 'm' or 'grads').
    """
    INTERNAL = frozenset(('m', 'values', 'grads', '_torch_steps',
                          '_torch_adam_params', '_torch_adam_state_owner'))

    def __init__(self, owner, public_group):
        if not isinstance(public_group, dict):
            raise TypeError('Adam public parameter group must be an actual dict')
        self.owner = owner
        self.public_group = public_group
        self.storage = {'_torch_adam_params': tuple(public_group['params']),
                        '_torch_adam_state_owner': owner}

    def __getitem__(self, key):
        return (self.storage if key in self.INTERNAL else self.public_group)[key]

    def __setitem__(self, key, value):
        (self.storage if key in self.INTERNAL else self.public_group)[key] = value

    def __contains__(self, key):
        return key in (self.storage if key in self.INTERNAL else self.public_group)

    def get(self, key, default=None):
        return (self.storage if key in self.INTERNAL else self.public_group).get(key, default)

    def setdefault(self, key, default=None):
        return (self.storage if key in self.INTERNAL else self.public_group).setdefault(key, default)

    def pop(self, key, *default):
        return (self.storage if key in self.INTERNAL else self.public_group).pop(key, *default)

    def __iter__(self):
        return iter(dict.fromkeys((*self.public_group, *self.storage)))

    def items(self):
        # Native placement repair enumerates only private storage. A legitimate
        # user config list must never be mistaken for an optimizer state buffer.
        return self.storage.items()


class AdamState:
    """Private controller; public state is an ACTUAL defaultdict(dict).

    Keys are the actual parameters. Hash/equality must be proven on real CUDA;
    no id-key or wrapper substitutes for the public parameter-key dictionary.
    """
    def __init__(self, optimizer, policy, reset_buffer):
        self.optimizer, self.policy = optimizer, policy
        self.reset_buffer = reset_buffer
        self.public_state = defaultdict(dict)
        self._known_records = {}  # Actual parameter -> prior inner dictionary.
        self._groups = []  # Private handles; public list/dicts are NEVER replaced.

    def execution_groups(self, *, include=None):
        public = self.optimizer.param_groups if include is None else [include]
        result = []
        for group in public:
            match = next((entry for entry in self._groups if entry.public_group is group), None)
            if match is None:
                match = AdamExecutionGroup(self, group)
                self._groups.append(match)
            result.append(match)
        if include is None:
            self._groups = result[:]
        return result

    def _find(self, parameter):
        for group in self.execution_groups():
            for index, current in enumerate(group["params"]):
                if current is parameter:
                    return group, index
        raise KeyError(parameter)

    def _record(self, parameter):
        return self.public_state.get(parameter)

    def __getitem__(self, parameter):
        return self.public_state[parameter]

    def get(self, parameter, default=None):
        record = self._record(parameter)
        return default if record is None else record

    def __setitem__(self, parameter, value):
        group, index = self._find(parameter)
        if not isinstance(value, Mapping):
            raise TypeError("optimizer state must be a mapping")
        value = value if isinstance(value, dict) else dict(value)
        if value:
            self._validate_record(value)
        else:
            self._reset(group, index)
        self.public_state[parameter] = value
        self.sync_group(group, validate=False)

    def __delitem__(self, parameter):
        group, index = self._find(parameter)
        if self._record(parameter) is None:
            raise KeyError(parameter)
        self._reset(group, index)
        del self.public_state[parameter]
        self._known_records.pop(parameter, None)
        self.sync_n_step()

    def __iter__(self):
        return iter(self.public_state)

    def __len__(self):
        return len(self.public_state)

    def __contains__(self, parameter):
        return parameter in self.public_state

    def items(self):
        return self.public_state.items()

    def adopt_public_state(self, state):
        # This is replacement of the actual publication object, NOT wrapping it.
        # Keep an explicit narrow restore surface instead of pretending that an
        # arbitrary Mapping has defaultdict's public type/serialization behavior.
        if type(state) not in (dict, defaultdict):
            raise TypeError('Adam public state must be a real dict or defaultdict')
        if type(state) is defaultdict and state.default_factory is not dict:
            raise TypeError('Adam state defaultdict factory must be dict')
        self.public_state = state

    def _reset(self, group, index):
        for key in ("m", "values"):
            group[key][index] = self.reset_buffer(group["params"][index])

    def _validate_record(self, record):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key not in record:
                raise KeyError(key)
        self.policy.read(record["step"])

    def initialized(self, group, index):
        parameter = group["params"][index]
        record = self[parameter]
        if not record:
            if parameter in self._known_records:
                self._reset(group, index)
                self._known_records.pop(parameter, None)
            record.update(step=self.policy.create(parameter, group, 0),
                          exp_avg=group["m"][index],
                          exp_avg_sq=group["values"][index])
        self._validate_record(record)
        self._known_records[parameter] = record
        return record

    def count(self, group, index):
        record = self._record(group["params"][index])
        if not record:
            return 0
        self._validate_record(record)
        return self.policy.read(record["step"])

    def set_count(self, group, index, value):
        value = _count(value)
        record = self.initialized(group, index)
        self.policy.assign(record["step"], value)

    def sync_n_step(self):
        self.optimizer.n_step = max((self.count(group, index)
            for group in self.execution_groups()
            for index in range(len(group["params"]))), default=0)

    def validate_groups(self):
        for group in self.execution_groups():
            self.policy.validate_group(group)

    def sync_group(self, group, *, validate=True):
        if type(group) is not AdamExecutionGroup or group.owner is not self:
            raise TypeError('Adam buffers must bind to this owner private execution group')
        for index, parameter in enumerate(group["params"]):
            record = self._record(parameter)
            if not record and parameter in self._known_records:
                # Direct public del/clear/state[p]={} bypasses this controller.
                # Detect that real dict operation before another optimizer step;
                # do not silently reuse the preceding moments at fresh step 1.
                self._reset(group, index)
                self._known_records.pop(parameter, None)
            if record:
                if validate:
                    self._validate_record(record)
                if "exp_avg" in record:
                    group["m"][index] = record["exp_avg"]
                if "exp_avg_sq" in record:
                    group["values"][index] = record["exp_avg_sq"]
                if validate:
                    self._known_records[parameter] = record
        slots = group.get("_torch_steps")
        if not isinstance(slots, AdamStepSlots) or slots.owner is not self:
            slots = group["_torch_steps"] = AdamStepSlots(self, group)
        slots.resize()
        group["_torch_adam_state_owner"] = self
        return slots

    def rebind(self, replacements):
        for old, new in replacements:
            if old is new:
                continue
            if old not in self.public_state:
                continue
            if new in self.public_state:
                raise ValueError("FSDP cannot merge distinct Adam state owners")
            record = self.public_state.pop(old)
            self.public_state[new] = record
            if old in self._known_records:
                self._known_records.pop(old)
                self._known_records[new] = record

    def publish_buffers(self, group):
        # FSDP may legitimately replace invalid-sized moment buffers. Publish
        # their actual new objects into the SAME canonical dictionary.
        for index, parameter in enumerate(group["params"]):
            record = self._record(parameter)
            if record:
                record["exp_avg"] = group["m"][index]
                record["exp_avg_sq"] = group["values"][index]

    def load_plan(self, plans):
        result = []
        for group, saved_group, entries in plans:
            for parameter, source in zip(group["params"], entries):
                if source is None:
                    continue  # Missing state stays absent, not an empty entry.
                record = dict(source)
                if record:
                    for key in ("step", "exp_avg", "exp_avg_sq"):
                        if key not in record:
                            raise KeyError(key)
                    record["step"] = self.policy.loaded(parameter, saved_group, record["step"])
                    self._validate_record(record)
                result.append((parameter, record))
        return result

    def apply_load(self, records, *, extra=()):
        # The complete plan (including all scalar/device conversions) already
        # succeeded before touching live dictionaries or positional buffers.
        state = defaultdict(dict)
        state.update(extra)
        state.update(records)
        self.public_state = state
        self._known_records = {}
        for group in self.execution_groups():
            for index, parameter in enumerate(group["params"]):
                if not self._record(parameter):
                    self._reset(group, index)
            self.sync_group(group)
        self.sync_n_step()


def state_owner(optimizer, torch, reset_buffer):
    owner = vars(optimizer).get("_torch_adam_state_owner")
    if owner is None:
        owner = AdamState(optimizer, PublicStepPolicy(torch), reset_buffer)
        optimizer._torch_adam_state_owner = owner
    for group in owner.execution_groups():
        if group.get("_torch_adam_state_owner") is not owner:
            owner.sync_group(group, validate=False)
    return owner


def reconcile_optimizer_state(optimizer):
    """Rebind existing Adam moments after public state storage migration.

    This opt-in compatibility bridge is not installed on ``torch``. The caller
    must be outside optimizer execution. It accepts only an already constructed
    AdamState with unchanged parameter topology and complete existing records.
    Public dictionaries, tensors, gradients and scalar steps are never replaced
    or modified. No Tensor construction, scalar readback, reset, move or GC is
    performed. Deleted/emptied initialized state and topology changes must use
    their ordinary optimizer APIs instead of this narrow migration boundary.
    """
    owner = vars(optimizer).get("_torch_adam_state_owner")
    if type(owner) is not AdamState or owner.optimizer is not optimizer:
        raise TypeError("reconciliation requires this optimizer's existing AdamState")
    state, groups = owner.public_state, optimizer.param_groups
    if type(state) not in (dict, defaultdict):
        raise TypeError("Adam public state must remain a real dict or defaultdict")
    if type(state) is defaultdict and state.default_factory is not dict:
        raise TypeError("Adam public state defaultdict factory must remain dict")
    if type(groups) is not list or type(owner._groups) is not list:
        raise TypeError("Adam parameter groups must retain their existing list topology")
    if len(groups) != len(owner._groups) or type(owner._known_records) is not dict:
        raise ValueError("Adam group topology or state bookkeeping changed")
    torch = owner.policy.torch
    parameters, plan = {}, []
    # Validate ALL groups and records before rebinding even the first buffer.
    for public_group, group in zip(groups, owner._groups):
        if (type(public_group) is not dict or type(group) is not AdamExecutionGroup
                or group.owner is not owner or group.public_group is not public_group
                or type(group.storage) is not dict
                or group.storage.get("_torch_adam_state_owner") is not owner):
            raise ValueError("Adam execution groups must match their existing public groups")
        params = public_group.get("params")
        previous = group.storage.get("_torch_adam_params")
        if (type(params) not in (list, tuple) or type(previous) is not tuple
                or len(params) != len(previous)
                or any(current is not old for current, old in zip(params, previous))):
            raise ValueError("Adam parameter topology changed before reconciliation")
        for key in ("m", "values"):
            if type(group.storage.get(key)) is not list or len(group.storage[key]) != len(params):
                raise ValueError("Adam execution moment slots must already exist")
        for index, parameter in enumerate(params):
            if not isinstance(parameter, torch.Tensor) or id(parameter) in parameters:
                raise ValueError("Adam parameters must be unique existing public Tensors")
            parameters[id(parameter)] = parameter
            record = state.get(parameter)
            if record is None and parameter not in state:
                if parameter in owner._known_records:
                    raise ValueError("initialized Adam state was deleted")
                continue  # Preserve genuinely absent, uninitialized state.
            if type(record) is not dict:
                raise TypeError("Adam inner state must remain a real dict")
            if not record:
                if parameter in owner._known_records:
                    raise ValueError("initialized Adam state was cleared")
                continue  # No defaultdict insertion or lazy tensor creation.
            for key in ("step", "exp_avg", "exp_avg_sq"):
                if key not in record:
                    raise KeyError(key)
            step = record["step"]
            if (not isinstance(step, torch.Tensor) or tuple(step.shape) != ()
                    or step.requires_grad or step.dtype not in (torch.float32, torch.float64)):
                raise ValueError("Adam step must remain a detached floating scalar Tensor")
            if parameter.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                raise ValueError("Adam reconciliation requires floating parameters")
            for key in ("exp_avg", "exp_avg_sq"):
                moment = record[key]
                if not isinstance(moment, torch.Tensor):
                    raise TypeError("Adam moment must remain a public Tensor")
                if (tuple(moment.shape) != tuple(parameter.shape)
                        or moment.dtype != parameter.dtype or moment.requires_grad):
                    raise ValueError("Adam moment shape/dtype/detachment is incompatible")
            if record["exp_avg"].device != record["exp_avg_sq"].device:
                raise ValueError("Adam moment pair must share the migrated device")
            plan.append((group, index, parameter, record))
    for mapping in (state, owner._known_records):
        if any(parameters.get(id(parameter)) is not parameter for parameter in mapping):
            raise ValueError("Adam state contains parameters outside the existing topology")
    # Direct metadata rebinding avoids sync_group's reset/step-slot resize paths.
    # Drop the prior inner-dict reference too, including whole-record replacement.
    for group, index, parameter, record in plan:
        group.storage["m"][index] = record["exp_avg"]
        group.storage["values"][index] = record["exp_avg_sq"]
        owner._known_records[parameter] = record
    return None
