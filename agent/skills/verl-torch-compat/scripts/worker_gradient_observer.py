"""Opt-in SGD observer, loaded through verl's public optimizer_impl setting.

This changes no update rule. The callback runs after the worker's gradient
clipping, immediately before ordinary SGD consumes the same gradients. The
caller owns observation and must not mutate parameters or gradients. In
particular, this observer does not call backward, manufacture missing gradients,
unflatten FSDP internals, or prevent the worker from clearing gradients later.
"""

import torch


class ObservedSGD(torch.optim.SGD):
    """SGD with a required, explicitly configured pre-step diagnostic callback."""

    def __init__(self, params, *args, **kwargs):
        super().__init__(params, *args, **kwargs)
        self._gradient_observer = None

    def set_gradient_observer(self, callback):
        if not callable(callback):
            raise TypeError("gradient observer must be callable")
        self._gradient_observer = callback

    def step(self, closure=None):
        if closure is not None:
            # This probe observes the gradients produced by verl before step.
            # A closure could replace them afterwards, invalidating that claim.
            raise ValueError("ObservedSGD only supports the worker's closure-free step")
        if self._gradient_observer is None:
            raise RuntimeError("ObservedSGD requires set_gradient_observer before step")
        self._gradient_observer(self)
        return super().step()
