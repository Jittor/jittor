"""Functional embedding implementations exposed through :mod:`jittor.nn`."""

import jittor as jt


def embedding(
    input,
    weight,
    padding_idx=None,
    max_norm=None,
    norm_type=2.0,
    scale_grad_by_freq=False,
    sparse=False,
):
    if max_norm is not None:
        indices = input.reshape((-1,)).unique()
        rows = weight[indices]
        norm = (rows.abs() ** norm_type).sum(dim=-1, keepdims=True) ** (1.0 / norm_type)
        scale = jt.ternary(
            norm > max_norm,
            max_norm / jt.maximum(norm, 1e-7),
            jt.ones_like(norm),
        )
        was_trainable = not weight.is_stop_grad()
        weight[indices] = (rows * scale).stop_grad()
        if was_trainable and weight.is_stop_grad():
            weight.start_grad()
        elif not was_trainable and not weight.is_stop_grad():
            weight.stop_grad()
    result = weight[input]
    if padding_idx is not None:
        keep = (input != padding_idx).unsqueeze(-1).float32()
        result = result * keep + (result * (1.0 - keep)).stop_grad()
    return result


def embedding_bag(
    input,
    weight,
    offsets=None,
    mode="mean",
    per_sample_weights=None,
):
    assert mode in ("sum", "mean", "max"), "unsupported mode {} in embedding_bag".format(mode)
    input = input if isinstance(input, jt.Var) else jt.array(input)
    if input.ndim == 1:
        assert offsets is not None, "offsets has to be provided when input is 1-D in embedding_bag"
        offsets = offsets if isinstance(offsets, jt.Var) else jt.array(offsets)
        ends = jt.concat(
            [offsets[1:], jt.array([input.shape[0]]).cast(offsets.dtype)],
            dim=0,
        )
        bags = []
        sample_weights = None
        if per_sample_weights is not None and mode == "sum":
            sample_weights = (
                per_sample_weights
                if isinstance(per_sample_weights, jt.Var)
                else jt.array(per_sample_weights)
            )
        for index in range(offsets.shape[0]):
            start = int(offsets[index].item())
            end = int(ends[index].item())
            values = weight[input[start:end]]
            if sample_weights is not None and mode == "sum":
                values = values * sample_weights[start:end].reshape((-1, 1))
            if mode == "max":
                bag = values.max(dim=0)
            elif mode == "mean":
                bag = values.mean(dim=0)
            else:
                bag = values.sum(dim=0)
            bags.append(bag.reshape((1, -1)))
        return jt.concat(bags, dim=0)

    assert input.ndim == 2, "input must be 1-D or 2-D in embedding_bag"
    values = weight[input]
    if per_sample_weights is not None and mode == "sum":
        sample_weights = (
            per_sample_weights
            if isinstance(per_sample_weights, jt.Var)
            else jt.array(per_sample_weights)
        )
        values = values * sample_weights.reshape(sample_weights.shape + (1,))
    if mode == "max":
        return values.max(dim=1)
    if mode == "mean":
        return values.mean(dim=1)
    return values.sum(dim=1)


__all__ = ["embedding", "embedding_bag"]
