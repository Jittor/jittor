"""Torch tensor indexing ownership."""
from jittor._core.dtypes import dtype_name as _jittor_dtype_name

def masked_scatter(input, mask, source):
    """Copy ``source`` into the True positions of ``mask``, out of place.

    ``source`` is consumed in row-major order and ``mask`` broadcasts to
    ``input.shape``. Differentiable w.r.t. both operands -- the Qwen-VL path
    scatters vision-tower image_embeds into the text inputs_embeds, and grads
    must reach the ViT. Implemented as gather(source, running-count-of-True)
    then where(mask), avoiding any sliced in-place write (a jittor no-view
    no-op).
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    broadcast_mask = mask
    if tuple(broadcast_mask.shape) != tuple(input.shape):
        broadcast_mask = broadcast_mask.broadcast(input.shape)
    broadcast_mask = broadcast_mask.bool()
    flat_mask = broadcast_mask.reshape(-1)
    # index into source.flatten() for each position = (#True strictly before it)
    picked = flat_mask.int32().cumsum(0) - 1
    # clamp: the out-of-range entries sit where the mask is False and are dropped
    picked = picked.maximum(0).minimum(source.numel() - 1)
    gathered = source.reshape(-1)[picked].reshape(input.shape)
    if _jittor_dtype_name(gathered.dtype) != _jittor_dtype_name(input.dtype):
        gathered = gathered.cast(_jittor_dtype_name(input.dtype))
    return _owner.jt.ternary(broadcast_mask, gathered, input)


def masked_scatter_(input, mask, source):
    """In-place ``masked_scatter``, keeping ``input``'s object identity.

    Writes back through ``assign()`` so the same Var -- and any module
    attribute holding it -- reflects the update.
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    input.assign(_owner.masked_scatter(input, mask, source))
    return input


def unfold(input, dimension, size, step):
    """Return sliding windows along ``dimension`` as a new trailing dim.

    ``out[..., i, ..., j] == input[..., i * step + j, ...]``. This is a
    materialized reindex, not the stride view Torch returns.
    """
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    rank = input.ndim
    axis = dimension if dimension >= 0 else dimension + rank
    count = (input.shape[axis] - size) // step + 1
    out_shape = list(input.shape)
    out_shape[axis] = count
    out_shape.append(size)
    source = [f"i{k}" for k in range(rank)]
    source[axis] = f"i{axis}*{step}+i{rank}"   # window pos + within-window
    return input.reindex(out_shape, source)


def addcmul(input, tensor1, tensor2, value=1):
    """Return ``input + value * (tensor1 * tensor2)``."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return input + value * (tensor1 * tensor2)


def addcdiv(input, tensor1, tensor2, value=1):
    """Return ``input + value * (tensor1 / tensor2)``."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return input + value * (tensor1 / tensor2)


def broadcast_to(input, shape):
    """Expand ``input`` to ``shape`` without copying, as Torch does."""
    from importlib import import_module as _import_module
    _owner = _import_module(__package__)
    return input.broadcast(shape)
