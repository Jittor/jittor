"""Native, equal-feature tensor-parallel linear layers and their adjoints.

Parameters are actual rank-local Vars, never full tensors marked as shards.
The initial scope is FP32 CPU singleton and CUDA/NCCL WORLD. Inputs and losses
must be replicated across tensor-parallel ranks; data parallelism is separate.
"""

import math

import jittor as jt
from jittor import distributed as dist
from jittor import init
from jittor.nn.functional.linear import linear
from jittor.nn.functional._amp import bias_for_compute_dtype


__all__ = ["ColumnParallelLinear", "RowParallelLinear"]


def _group():
    group = dist.get_default_group()
    if group.size() > 1 and dist.get_backend() != "nccl":
        raise NotImplementedError("native tensor parallelism requires CUDA/NCCL WORLD")
    return group


def _gather_last(value, group):
    axes = [value.ndim - 1] + list(range(value.ndim - 1))
    leading = value.transpose(axes)
    gathered = group.all_gather(leading)
    return gathered.transpose(list(range(1, value.ndim)) + [0])


class _Copy(jt.Function):
    def __init__(self, group):
        self.group = group

    def execute(self, value):
        return value

    def grad(self, gradient):
        return self.group.all_reduce(gradient, "sum")


class _Reduce(jt.Function):
    def __init__(self, group):
        self.group = group

    def execute(self, value):
        return self.group.all_reduce(value, "sum")

    def grad(self, gradient):
        # The output/loss is replicated. Each local partial output contributes
        # once, so a second all-reduce here would multiply every gradient.
        return gradient


class _Gather(jt.Function):
    def __init__(self, group):
        self.group = group

    def execute(self, value):
        self.width = value.shape[-1]
        return _gather_last(value, self.group)

    def grad(self, gradient):
        start = self.group.rank() * self.width
        return gradient[..., start:start + self.width]


class _Scatter(jt.Function):
    def __init__(self, group):
        self.group = group

    def execute(self, value):
        self.width = value.shape[-1] // self.group.size()
        start = self.group.rank() * self.width
        return value[..., start:start + self.width]

    def grad(self, gradient):
        return _gather_last(gradient, self.group)


def _spec(kind, shape, global_shape, axis, group):
    return {"version": 1, "kind": kind, "shape": tuple(shape),
            "global_shape": tuple(global_shape), "axis": axis,
            "rank": group.rank(), "world_size": group.size()}


def _mark(parameter, kind, global_shape, axis, group):
    parameter._jittor_tp_spec = _spec(kind, parameter.shape, global_shape, axis, group)
    return parameter


def _validate_optimizer_parameter(parameter):
    """Validate the explicit native optimizer exclusion; no math is replaced."""
    spec = parameter._jittor_tp_spec
    group = _group()
    if (tuple(parameter.shape) != spec["shape"] or group.rank() != spec["rank"]
            or group.size() != spec["world_size"]):
        raise ValueError("tensor-parallel parameter layout or WORLD changed")
    if getattr(parameter, "_jittor_ddp_state", None) is not None:
        raise NotImplementedError("native TP parameters cannot also be managed by DDP")
    if group.size() > 1 and jt.core.dispatch_context([parameter]) != ("cuda", dist.get_device()):
        raise ValueError("tensor-parallel parameter must remain on the WORLD CUDA device")


class _ParallelLinear(jt.Module):
    def _configure(self, in_features, out_features, partition_features):
        if (type(in_features) is not int or type(out_features) is not int
                or in_features <= 0 or out_features <= 0):
            raise ValueError("tensor-parallel feature sizes must be positive integers")
        self.in_features, self.out_features = in_features, out_features
        self._group = _group()
        if partition_features % self._group.size():
            raise ValueError("partitioned feature count must divide WORLD size")
        self._world = self._group.size()
        self._rank = self._group.rank()

    def _check(self, value, width):
        if value.ndim < 1 or value.shape[-1] != width:
            raise ValueError("tensor-parallel linear input has the wrong last dimension")
        if (self._group is not dist.get_default_group()
                or self._group.rank() != self._rank or self._group.size() != self._world):
            raise ValueError("tensor-parallel WORLD changed after construction")
        for parameter in self.parameters():
            _validate_optimizer_parameter(parameter)
        if self._world > 1 and jt.core.dispatch_context([value]) != ("cuda", dist.get_device()):
            raise ValueError("tensor-parallel input must be on the WORLD CUDA device")

    def partition(self):
        """Describe owned storage; the returned metadata contains no tensor aliases."""
        return {name: dict(parameter._jittor_tp_spec)
                for name, parameter in self.named_parameters()}


class ColumnParallelLinear(_ParallelLinear):
    """Shard weight rows/output features; optionally gather replicated output.

    Input is replicated. Its backward sums contributions from every output
    shard. With ``gather_output=False``, feed local output to a matching
    ``RowParallelLinear(input_is_parallel=True)``. Dimensions must divide WORLD.
    """

    def __init__(self, in_features, out_features, bias=True, *, gather_output=False):
        self._configure(in_features, out_features, out_features)
        self.gather_output = bool(gather_output)
        local = out_features // self._world
        bound = 1 / math.sqrt(in_features)
        self.weight = _mark(init.uniform((local, in_features), "float32", -bound, bound),
                            "column", (out_features, in_features), 0, self._group)
        self.bias = (_mark(init.uniform((local,), "float32", -bound, bound),
                           "column", (out_features,), 0, self._group) if bias else None)

    def execute(self, value):
        self._check(value, self.in_features)
        result = linear(_Copy(self._group)(value), self.weight, self.bias)
        return _Gather(self._group)(result) if self.gather_output else result


class RowParallelLinear(_ParallelLinear):
    """Shard weight columns/input features, then sum local partial outputs.

    Bias is replicated and added once after reduction. ``input_is_parallel``
    accepts a local feature shard; otherwise input is replicated and scattered.
    """

    def __init__(self, in_features, out_features, bias=True, *, input_is_parallel=False):
        self._configure(in_features, out_features, in_features)
        self.input_is_parallel = bool(input_is_parallel)
        local = in_features // self._world
        bound = 1 / math.sqrt(in_features)
        self.weight = _mark(init.uniform((out_features, local), "float32", -bound, bound),
                            "row", (out_features, in_features), 1, self._group)
        self.bias = None
        if bias:
            self.bias = _mark(init.uniform((out_features,), "float32", -bound, bound),
                              "replicated", (out_features,), None, self._group)
            if self._world > 1:
                self.bias.update(self._group.broadcast(self.bias, src=0))
                self.bias.start_grad()
                self.bias.sync()

    def execute(self, value):
        self._check(value, self.in_features // self._world
                    if self.input_is_parallel else self.in_features)
        local = value if self.input_is_parallel else _Scatter(self._group)(value)
        result = _Reduce(self._group)(linear(local, self.weight))
        return (result + bias_for_compute_dtype(result, self.bias)
                if self.bias is not None else result)
