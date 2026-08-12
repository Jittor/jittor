"""Autograd compatibility helpers exposed through :mod:`jittor.nn`."""

import jittor as jt  # noqa: F401
from jittor_utils import LOG


def backward(v, *args, **kw):
    """The `backward` variable interface doesn't exist in Jittor.
    please use `optimizer.backward(loss)` or
    `optimizer.step(loss)` instead.
    For example, if your code looks like this::

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    It can be changed to this::

        optimizer.zero_grad()
        optimizer.backward(loss)
        optimizer.step()

    Or more concise::

        optimizer.step(loss)

    The step function will automatically zero grad and backward.
    """
    LOG.f(backward.__doc__)


__all__ = ["backward"]
