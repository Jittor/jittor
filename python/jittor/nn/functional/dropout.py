"""Functional dropout implementations exposed through :mod:`jittor.nn`."""

import jittor as jt


def _check_probability(p):
    assert 0 <= p <= 1, "dropout probability has to be between 0 and 1, but got {}".format(p)


class _Dropout(jt.Function):
    """Dropout that keeps nothing for its backward but a one-byte mask.

    As operators the backward held the float32 draw the mask was compared
    from (the comparison fused into the product, so the mask itself was never
    stored) and the dropped tensor's input besides: three tensors where
    PyTorch keeps a byte an element -- 552 MB of a BERT-base training step's
    peak for the draws alone.
    """

    def execute(self, x, p):
        from jittor._core.var import _captured_keep
        mask = _captured_keep(x.shape, p)
        self.mask = mask if mask is not None else (jt.random(x.shape) > p).stop_fuse()
        self.scale = 1.0 / (1.0 - p)
        return (x * self.mask * self.scale).to(x.dtype)

    def grad(self, g):
        return (g * self.mask * self.scale).to(g.dtype), None


def dropout(x, p=0.5, is_train=False, training=None):
    if training is not None:
        is_train = training
    _check_probability(p)
    output = x
    if p > 0 and is_train:
        if p == 1:
            output = output * jt.zeros(x.shape)
        elif not x.is_stop_grad() and not jt.flags.no_grad:
            return _Dropout.apply(x, p)
        else:
            noise = jt.random(x.shape) > p
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
