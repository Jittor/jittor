"""Stateful normalization modules."""

import jittor as jt
from jittor import Module, init

from ..functional.normalization import fp32_guard


class BatchNorm(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        is_train=True,
        sync=True,
        track_running_stats=True,
        device=None,
        dtype=None,
    ):
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0
        self.running_mean = init.constant((num_features,), "float32", 0.0).stop_grad()
        self.running_var = init.constant((num_features,), "float32", 1.0).stop_grad()
        self.num_batches_tracked = init.constant((1,), "int32", 0.0).stop_grad()
        for buffer in (
            self.running_mean,
            self.running_var,
            self.num_batches_tracked,
        ):
            object.__setattr__(buffer, "is_buffer", True)
        object.__setattr__(self.num_batches_tracked, "persistent", False)

    def execute(self, x):
        dims = [0] + list(range(2, x.ndim))
        if self.is_train:
            if self.track_running_stats:
                self.num_batches_tracked.update(
                    self.num_batches_tracked + 1
                )
            xmean = jt.mean(x, dims=dims)
            x2mean = jt.mean(x * x, dims=dims)
            sync = self.sync and jt.in_mpi
            if sync:
                xmean = xmean.mpi_all_reduce("mean")
                x2mean = x2mean.mpi_all_reduce("mean")

            xvar = (x2mean - xmean * xmean).maximum(0.0)
            if sync:
                weight = self.weight / jt.sqrt(xvar + self.eps)
                bias = self.bias - xmean * weight
                norm_x = x * weight.broadcast(x, dims) + bias.broadcast(x, dims)
            else:
                fast = jt.nn._batch_norm_cuda(
                    x, self.weight, self.bias, self.eps
                )
                if fast is not None:
                    norm_x = fast
                else:
                    xhat = jt.nn._ln_normalize(x, dims, self.eps)
                    if self.affine:
                        shape = [1, self.num_features] + [1] * (x.ndim - 2)
                        norm_x = xhat * self.weight.reshape(shape) + self.bias.reshape(shape)
                    else:
                        norm_x = xhat

            self.running_mean.update(
                self.running_mean + (xmean.reshape((-1,)) - self.running_mean) * self.momentum
            )
            count = 1
            for dim in dims:
                count *= x.shape[dim]
            if sync:
                count *= jt.world_size
            run_var = xvar * (count / (count - 1)) if count > 1 else xvar
            self.running_var.update(
                self.running_var + (run_var.reshape((-1,)) - self.running_var) * self.momentum
            )
            return norm_x

        fast = jt.nn._batch_norm_eval_cuda(
            x, self.weight, self.bias,
            self.running_mean, self.running_var, self.eps,
        )
        if fast is not None:
            return fast
        weight = self.weight / jt.sqrt(self.running_var + self.eps)
        bias = self.bias - self.running_mean * weight
        return x * weight.broadcast(x, dims) + bias.broadcast(x, dims)


BatchNorm1d = BatchNorm
BatchNorm2d = BatchNorm
BatchNorm3d = BatchNorm


class InstanceNorm(Module):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        is_train=True,
        sync=True,
    ):
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        dims = list(range(2, x.ndim))
        xhat = jt.nn._ln_normalize(x, dims, self.eps)
        if not self.affine:
            return xhat
        shape = [1, self.num_features] + [1] * len(dims)
        return xhat * self.weight.reshape(shape) + self.bias.reshape(shape)


InstanceNorm1d = InstanceNorm
InstanceNorm2d = InstanceNorm
InstanceNorm3d = InstanceNorm


class LayerNorm(Module):
    def __init__(
        self,
        normalized_shape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = init.constant(normalized_shape, "float32", 1.0) if elementwise_affine else 1.0
        self.bias = (
            init.constant(normalized_shape, "float32", 0.0) if elementwise_affine and bias else 0.0
        )

    @fp32_guard
    def execute(self, x):
        dims = [-i for i in range(len(self.normalized_shape), 0, -1)]
        weight = 1.0 if self.weight is None else self.weight
        bias = 0.0 if self.bias is None else self.bias
        fast = jt.nn._layer_norm_cuda(
            x, self.normalized_shape, weight, bias, self.eps
        )
        if fast is not None:
            return fast
        fast = jt.nn._layer_norm_no_grad_cuda(x, self.normalized_shape, weight, bias, self.eps)
        if fast is not None:
            return fast
        xhat = jt.nn._ln_normalize(x, dims, self.eps)
        return xhat * weight + bias

    def reset_parameters(self):
        if isinstance(self.weight, jt.Var):
            self.weight.update(jt.ones_like(self.weight))
        if isinstance(self.bias, jt.Var):
            self.bias.update(jt.zeros_like(self.bias))


LayerNorm1d = LayerNorm
LayerNorm2d = LayerNorm
LayerNorm3d = LayerNorm


class GroupNorm(Module):
    def __init__(self, num_groups, num_channels, eps=1e-5, affine=True, is_train=True):
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        self.weight = init.constant((num_channels,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_channels,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        batch = x.shape[0]
        channels = self.num_channels
        assert channels % self.num_groups == 0, (
            "GroupNorm: num_channels (%s) must be divisible by num_groups (%s)"
            % (channels, self.num_groups)
        )
        fast = jt.nn._group_norm_cuda(x, self.num_groups, self.weight, self.bias, self.eps)
        if fast is not None:
            return fast
        grouped = x.reshape((batch, self.num_groups, channels // self.num_groups, -1))
        xhat = jt.nn._ln_normalize(grouped, [2, 3], self.eps).reshape(x.shape)
        if not self.affine:
            return xhat
        shape = [1, channels] + [1] * (x.ndim - 2)
        return xhat * self.weight.reshape(shape) + self.bias.reshape(shape)
