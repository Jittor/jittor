"""Stateful normalization modules."""

import jittor as jt
from jittor import Module, init

from ..functional.normalization import (
    _batch_norm_eval,
    _batch_norm_train,
    _unbiased,
)
from ... import _arg_policy


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
        # register_buffer, not a tagged assignment. Tagging the Var
        # (`object.__setattr__(buf, "is_buffer", True)`) records the classification
        # on the OBJECT, and the object does not survive being replaced: after
        # `bn.running_mean = jt.zeros(n)` -- which is how a checkpoint load, a dtype
        # cast or a hand-written reset writes it -- the new Var carries no tag and
        # running_mean became a trainable parameter, so the optimizer's weight decay
        # started dragging the running statistics towards zero. register_buffer
        # records the NAME on the module, which no reassignment can lose. That name
        # set is exactly the mechanism this bypassed.
        self.register_buffer(
            "running_mean",
            init.constant((num_features,), "float32", 0.0).stop_grad())
        self.register_buffer(
            "running_var",
            init.constant((num_features,), "float32", 1.0).stop_grad())
        # Kept non-persistent, as it has always been here: jittor's checkpoints do
        # not carry num_batches_tracked, and load_parameters/load_state_dict both
        # special-case the key rather than report it missing.
        self.register_buffer(
            "num_batches_tracked",
            init.constant((1,), "int32", 0.0).stop_grad(), persistent=False)

    def execute(self, x):
        # Parameters and buffers live here; the arithmetic lives in
        # nn.functional.normalization, which nn.functional.batch_norm also
        # calls. This used to be a second transcription, and its two training
        # branches did not even agree with each other: with MPI the statistics
        # came from E[x^2]-E[x]^2 and the output was a scale-shift of raw x
        # differentiated by composite autodiff, without MPI they came from the
        # two-pass formula and went through the stable closed-form backward.
        # Whether the job was distributed decided the numbers and the gradient.
        dims = [0] + list(range(2, x.ndim))
        if not self.is_train:
            return _batch_norm_eval(
                x, dims, self.running_mean, self.running_var,
                self.weight, self.bias, self.eps)
        sync = self.sync and jt.in_mpi
        norm_x, xmean, xvar = _batch_norm_train(
            x, dims, self.weight, self.bias, self.eps, sync=sync)
        if self.track_running_stats:
            self.num_batches_tracked.update(self.num_batches_tracked + 1)
        world_size = jt.world_size if sync else 1
        self.running_mean.update(
            self.running_mean
            + (xmean.reshape((-1,)) - self.running_mean) * self.momentum)
        self.running_var.update(
            self.running_var
            + (_unbiased(xvar, x, dims, world_size).reshape((-1,))
               - self.running_var) * self.momentum)
        return norm_x


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
        # This module has no running statistics: it normalises every sample with
        # its own spatial statistics in both train and eval mode. That makes
        # three of its parameters inert rather than merely unused.
        if momentum != 0.1:
            _arg_policy.ignored(
                "jittor.nn.InstanceNorm", "momentum", momentum,
                "momentum only ever weights a running-statistics update, and "
                "this module tracks no running statistics")
        if not is_train:
            _arg_policy.ignored(
                "jittor.nn.InstanceNorm", "is_train", is_train,
                "with no running statistics there is no eval-mode behaviour to "
                "switch to, so both modes normalise with per-sample statistics")
        if not sync:
            _arg_policy.ignored(
                "jittor.nn.InstanceNorm", "sync", sync,
                "the statistics are per sample and never cross rank "
                "boundaries, so there is nothing for this flag to turn off")
        self.sync = sync
        self.num_features = num_features
        self.is_train = is_train
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.weight = init.constant((num_features,), "float32", 1.0) if affine else 1.0
        self.bias = init.constant((num_features,), "float32", 0.0) if affine else 0.0

    def execute(self, x):
        return jt.nn.instance_norm(x, weight=self.weight, bias=self.bias,
                                   eps=self.eps)


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

    def execute(self, x):
        # fp32_guard lives on the functional; applying it here too would cast
        # twice.
        return jt.nn.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.eps,
                                self.elementwise_affine)

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
        if x.shape[1] != self.num_channels:
            raise ValueError(
                "GroupNorm: expected %s channels (num_channels), but got %s; "
                "input shape %s"
                % (self.num_channels, x.shape[1], tuple(x.shape)))
        return jt.nn.group_norm(x, self.num_groups, self.weight, self.bias,
                                self.eps)
