"""Native ZeRO stage 1: shard optimizer state, retain replicated parameters.

The base Optimizer owns backward, accumulation and DDP reduction. Existing
native algorithms update only owned parameters; no optimizer math is copied.
"""

import hashlib
import itertools
import json
from copy import deepcopy

import jittor as jt
from jittor._core.dtypes import dtype_name
from jittor.optim import Adam, AdamW, SGD
from jittor.optim.base import Optimizer
from jittor.serialization.sharded import RankLocal


__all__ = ["ZeroRedundancyOptimizer"]
_INSTANCES = itertools.count()
_ALGORITHMS = (Adam, AdamW, SGD)
_STATE_FIELDS = frozenset(("grads", "m", "values", "momentum_initialized", "n_step"))


def _name(value):
    return value.__module__ + "." + value.__qualname__


def _metadata(groups):
    return [[{"shape": list(p.shape), "dtype": dtype_name(p.dtype)}
             for p in group["params"]] for group in groups]


def _owners(groups, world):
    """Greedily assign whole parameters; stable ties use group/parameter order."""
    loads = [0] * world
    result = [[None] * len(group["params"]) for group in groups]
    entries = [(int(p.numel()), gi, pi) for gi, group in enumerate(groups)
               for pi, p in enumerate(group["params"])]
    for size, gi, pi in sorted(entries, key=lambda value: (-value[0], value[1], value[2])):
        rank = min(range(world), key=lambda index: (loads[index], index))
        result[gi][pi] = rank
        loads[rank] += size
    return result


class ZeroRedundancyOptimizer(Optimizer):
    """Replicated native training with rank-owned Adam/AdamW/SGD state.

    Multi-rank use requires a native WORLD process group with a Store. Call
    backward/step on every rank in the same order. Groups and parameter order
    are fixed after construction; parameter aliases must occur once in groups.
    This is stage 1: parameters and gradients remain replicated. It neither
    implements FSDP/TP nor shards a parameter internally. Checkpoints retain
    rank-owned state and currently require the same world size and partition.
    """

    def __init__(self, params, optimizer_class=Adam, *, timeout=60, **kwargs):
        from jittor import distributed as dist

        if optimizer_class not in _ALGORITHMS:
            raise NotImplementedError("native ZeRO supports exactly Adam, AdamW and SGD")
        if timeout <= 0:
            raise ValueError("native ZeRO timeout must be positive")
        parameters = list(params)
        if not parameters:
            raise ValueError("native ZeRO needs at least one parameter group")
        groups = ([dict(group, params=list(group["params"])) for group in parameters]
                  if isinstance(parameters[0], dict) else [{"params": parameters}])
        if any(_STATE_FIELDS.intersection(group) for group in groups):
            raise ValueError("native ZeRO expects fresh parameter groups without optimizer state")
        flat = [p for group in groups for p in group["params"]]
        if not flat or any(not isinstance(p, jt.Var) for p in flat):
            raise ValueError("native ZeRO requires native tensor parameters")
        if len({id(p) for p in flat}) != len(flat):
            raise ValueError("native ZeRO parameter aliases must appear only once in groups")
        self._rank = dist.get_rank()
        self._world = dist.get_world_size()
        self._store = dist.get_default_store()
        self._timeout = timeout
        if self._world > 1 and self._store is None:
            raise ValueError("multi-rank native ZeRO requires a native WORLD Store")
        if self._world > 1 and dist.get_backend() != "nccl":
            raise NotImplementedError("multi-rank native ZeRO currently requires CUDA/NCCL WORLD")
        if self._world > 1 and any(jt.core.dispatch_context([p]) != ("cuda", dist.get_device())
                                   for p in flat):
            raise ValueError("native ZeRO parameters must be on the WORLD CUDA device")
        self._prefix = "native-zero/{}/".format(next(_INSTANCES))
        self._sequence = itertools.count()
        self._optimizer_class = optimizer_class
        self._owners = _owners(groups, self._world)
        self._indices = [[i for i, owner in enumerate(owners) if owner == self._rank]
                         for owners in self._owners]
        self._parameter_ids = [[id(p) for p in group["params"]] for group in groups]
        self._parameter_metadata = _metadata(groups)
        # Validate the replicated protocol before allocating any algorithm state.
        signature = {"algorithm": _name(optimizer_class), "kwargs": kwargs,
                     "groups": [{k: v for k, v in group.items() if k != "params"}
                                for group in groups], "parameters": self._parameter_metadata,
                     "trainability": [[bool(p.requires_grad) for p in group["params"]]
                                      for group in groups],
                     "owners": self._owners}
        self._agree(signature)
        local_groups = [dict(group, params=[group["params"][i] for i in indices])
                        for group, indices in zip(groups, self._indices)]
        # This is the only algorithm instance. Non-owned moments are never made.
        self._local_optimizer = optimizer_class(local_groups, **kwargs)
        super().__init__(groups, self._local_optimizer.lr)
        for key, value in self._local_optimizer.defaults.items():
            if key not in ("param_groups", "n_step", "param_sync_iter"):
                setattr(self, key, value)
        if self._world > 1:
            for parameter in flat:
                self._broadcast(parameter, 0)

    @property
    def local_optimizer(self):
        """The existing native algorithm instance, containing only owned state."""
        return self._local_optimizer

    def state_partition(self):
        """Return host ownership metadata; no tensor state is gathered or cloned."""
        return {"rank": self._rank, "world_size": self._world,
                "owners": [list(owners) for owners in self._owners],
                "local_indices": [list(indices) for indices in self._indices]}

    def _agree(self, signature):
        encoded = json.dumps(signature, sort_keys=True, separators=(",", ":"), allow_nan=False)
        if self._world == 1:
            return
        digest = hashlib.sha256(encoded.encode("utf8")).hexdigest().encode("ascii")
        key = self._prefix + str(next(self._sequence)) + "/"
        self._store.set(key + str(self._rank), digest)
        keys = [key + str(rank) for rank in range(self._world)]
        self._store.wait(keys, timeout=self._timeout)
        mismatch = any(value != digest for value in [self._store.get(name) for name in keys])
        # A server-hosting rank must not enter a GIL-holding collective while
        # a peer still needs an RPC reply. The native Store's arrive contract
        # flushes the peer's last reply BEFORE publishing its arrival marker.
        completed = [key + "read/" + str(rank) for rank in range(1, self._world)]
        if self._rank:
            self._store.arrive(completed[self._rank - 1])
        else:
            self._store.wait(completed, timeout=self._timeout)
            for name in keys + completed:
                self._store.delete_key(name)
        if mismatch:
            raise ValueError("native ZeRO parameter/group protocol differs across ranks")

    def _check_layout(self):
        from jittor import distributed as dist

        if (dist.get_rank(), dist.get_world_size()) != (self._rank, self._world):
            raise ValueError("native ZeRO process-group rank/world changed")
        ids = [[id(p) for p in group["params"]] for group in self.param_groups]
        if ids != self._parameter_ids or _metadata(self.param_groups) != self._parameter_metadata:
            raise ValueError("native ZeRO parameter binding/shape/dtype changed")
        self._agree({"trainability": [[bool(p.requires_grad) for p in group["params"]]
                                     for group in self.param_groups],
                     "groups": [{k: v for k, v in group.items()
                                 if k not in _STATE_FIELDS and k != "params"}
                                for group in self.param_groups],
                     "defaults": {k: v for k, v in self.defaults.items()
                                  if k != "param_groups"}})

    def add_param_group(self, group):
        raise NotImplementedError("native ZeRO parameter groups are fixed after construction")

    def backward(self, loss, retain_graph=False):
        self._check_layout()
        return super().backward(loss, retain_graph)

    def _broadcast(self, parameter, owner):
        from jittor import distributed as dist

        trainable = bool(parameter.requires_grad)
        synchronized = dist.broadcast(parameter, src=owner)
        if synchronized is not parameter:
            parameter.update(synchronized)
        parameter.start_grad() if trainable else parameter.stop_grad()
        # Different owners have different update graphs. Materialize each
        # broadcast in parameter order rather than relying on graph scheduling
        # to issue collectives in the same order on every rank.
        parameter.sync()

    def step(self, loss=None, retain_graph=False):
        self._check_layout()
        self.pre_step(loss, retain_graph)
        local = self._local_optimizer
        for key, value in self.defaults.items():
            if key != "param_groups":
                setattr(local, key, value)
        for group, owned, indices in zip(self.param_groups, local.param_groups, self._indices):
            if "grads" not in group:
                raise RuntimeError("native ZeRO step requires backward first")
            owned["grads"] = [group["grads"][i] for i in indices]
            for key in list(owned):
                if key != "params" and key not in _STATE_FIELDS and key not in group:
                    del owned[key]
            for key, value in group.items():
                if key != "params" and key not in _STATE_FIELDS:
                    owned[key] = value
        if self._world > 1:
            # Collective gradients belong to all ranks. Finish their exact
            # roots before only owners build update graphs: otherwise a lazy
            # constant-gradient producer can enter a later owner's sync while
            # non-owners have already advanced to parameter broadcasts.
            gradients = [g for group in self.param_groups
                         for p, g in zip(group["params"], group["grads"])
                         if p.requires_grad]
            jt.sync(gradients, True, False)
            self._agree({"collective_gradients_ready": self.n_step})
        local.step()  # No backward, and therefore no second gradient reduction.
        for group, owned, owners in zip(self.param_groups, local.param_groups, self._owners):
            group["n_step"] = owned["n_step"]
            for parameter, owner in zip(group["params"], owners):
                if self._world > 1:
                    self._broadcast(parameter, owner)
        self.post_step()

    def state_dict(self):
        defaults = dict(self.defaults)
        local = self._local_optimizer.state_dict()
        local_defaults = dict(local["defaults"])
        local_defaults["param_groups"] = [{k: v for k, v in group.items() if k != "params"}
                                          for group in local_defaults["param_groups"]]
        defaults["zero_state"] = RankLocal({
            "version": 1, "rank": self._rank, "world_size": self._world,
            "owners": deepcopy(self._owners), "parameters": deepcopy(self._parameter_metadata),
            "algorithm": _name(self._optimizer_class), "state": {"defaults": local_defaults}})
        return {"defaults": defaults}

    def validate_state_dict(self, state):
        """Validate rank-owned state before a training checkpoint mutates owners."""
        try:
            zero = state["defaults"]["zero_state"]
            if isinstance(zero, RankLocal):
                zero = zero.value
            expected = {"version": 1, "rank": self._rank, "world_size": self._world,
                        "owners": self._owners, "parameters": self._parameter_metadata,
                        "algorithm": _name(self._optimizer_class)}
            if any(zero[key] != value or type(zero[key]) is not type(value)
                   for key, value in expected.items()):
                raise ValueError("native ZeRO checkpoint partition/algorithm mismatch")
            global_groups = state["defaults"]["param_groups"]
            if (len(global_groups) != len(self.param_groups) or any(
                    {"m", "values", "momentum_initialized"}.intersection(group)
                    for group in global_groups)):
                raise ValueError("native ZeRO checkpoint has unexpected replicated algorithm state")
            groups = zero["state"]["defaults"]["param_groups"]
            if len(groups) != len(self._indices):
                raise ValueError("native ZeRO checkpoint local group count mismatch")
            for saved, target in zip(groups, self._local_optimizer.param_groups):
                gradients = saved.get("grads", [])
                if "grads" in saved and (len(gradients) != len(target["params"]) or any(
                        not isinstance(value, jt.Var) or tuple(value.shape) != tuple(parameter.shape)
                        for value, parameter in zip(gradients, target["params"]))):
                    raise ValueError("native ZeRO checkpoint owned gradient shape mismatch")
                for field in _STATE_FIELDS - {"grads", "n_step"}:
                    if field not in target:
                        continue
                    values = saved[field]
                    if not isinstance(values, (list, tuple)) or len(values) != len(target[field]):
                        raise ValueError("native ZeRO checkpoint owned state length mismatch")
                    for value, template in zip(values, target[field]):
                        if isinstance(template, jt.Var):
                            if (not isinstance(value, jt.Var) or tuple(value.shape) != tuple(template.shape)
                                    or dtype_name(value.dtype) != dtype_name(template.dtype)):
                                raise ValueError("native ZeRO checkpoint owned state shape/dtype mismatch")
                        elif type(value) is not type(template):
                            raise ValueError("native ZeRO checkpoint state flag type mismatch")
                if type(saved.get("n_step", 0)) is not int or saved.get("n_step", 0) < 0:
                    raise ValueError("native ZeRO checkpoint step count mismatch")
        except (KeyError, TypeError) as error:
            raise ValueError("invalid native ZeRO checkpoint state") from error
        return zero

    def load_state_dict(self, state):
        zero = self.validate_state_dict(state)
        defaults = {k: v for k, v in state["defaults"].items() if k != "zero_state"}
        # The native loader merges group fields. Here absence of an override
        # means use the saved algorithm default, not a stale target override.
        for targets, saved in ((self.param_groups, defaults["param_groups"]),
                               (self._local_optimizer.param_groups,
                                zero["state"]["defaults"]["param_groups"])):
            for target, source in zip(targets, saved):
                for key in list(target):
                    if key != "params" and key not in source:
                        del target[key]
        super().load_state_dict({"defaults": defaults})
        self._local_optimizer.load_state_dict(zero["state"])
        self.zero_grad()
