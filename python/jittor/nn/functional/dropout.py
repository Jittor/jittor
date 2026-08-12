"""Functional dropout implementations exposed through :mod:`jittor.nn`."""

import jittor as jt


def _check_probability(p):
    assert 0 <= p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)


def dropout(x, p=0.5, is_train=False, training=None):
    if training is not None:
        is_train = training
    _check_probability(p)
    output = x
    if p > 0 and is_train:
        if p == 1:
            output = output * jt.zeros(x.shape)
        else:
            noise = (jt.random(x.shape) > p).int()
            output = output * noise / (1.0 - p)
    return output.to(x.dtype)


def dropout2d(x, p=0.5, is_train=False):
    _check_probability(p)
    if x.dim() not in (3, 4):
        raise RuntimeError(
            "Expected 3D (unbatched) or 4D (batched) input to Dropout2d, "
            "but got input of size: {}".format(x.shape)
        )
    output = x
    if p > 0 and is_train:
        if p == 1:
            output = jt.zeros(x.shape)
        else:
            noise = (jt.random(x.shape[:-2]) > p).int()
            output = output * noise.broadcast(x.shape, dims=[-2, -1]) / (1.0 - p)
    return output


def droppath(x, p=0.5, is_train=False):
    if p == 0.0 or not is_train:
        return x
    keep_prob = 1 - p
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + jt.rand(shape, dtype=x.dtype)
    return x.divide(keep_prob) * random_tensor.floor()


__all__ = ["dropout", "dropout2d", "droppath"]
