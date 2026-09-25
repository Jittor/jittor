"""One operator for a whole parameter list's AdamW update on CUDA.

The per-parameter update is about ten graph nodes: decay, two moment updates,
two bias corrections, a square root, a division, the step, and the casts
between them. A diffusers UNet has 450 parameter tensors, so an AdamW step
built 4500 nodes in Python and sent them through the executor every
iteration -- more host time than the whole forward pass. PyTorch's default
AdamW on CUDA is a ``foreach`` kernel: a handful of launches for the list.

This hands the list to ``fused_adamw`` (src/ops/composite/fused_adamw_op.cc),
whose CUDA kernel updates every tensor of a launch from one argument table.
The arithmetic is the one ``optimizer_api._adam_step`` uses per parameter.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import register_kernel

_DTYPES = ("float32", "float16", "bfloat16")


def _supports(entries, *args, **kwargs):
    """Dense tensors of one supported dtype per entry: the kernel writes raw pointers."""
    if not entries:
        return False
    for parameter, moment, variance, gradient, _ in entries:
        dtype = _jittor_dtype_name(parameter.dtype)
        if dtype not in _DTYPES:
            return False
        for tensor in (parameter, moment, variance, gradient):
            if _jittor_dtype_name(tensor.dtype) != dtype or not tensor._storage_is_contiguous():
                return False
    return True


def _cuda_fused_adamw_updates(entries, lr, beta1, beta2, weight_decay, eps):
    """(new parameter, new moment, new variance) per entry, in entry order.

    An entry is (parameter, moment, variance, gradient, steps taken so far);
    this is the update for step ``steps + 1``, as the per-parameter path
    computes it. Entries are grouped by step and dtype, one operator each.
    """
    results = [None] * len(entries)
    groups = {}
    for index, entry in enumerate(entries):
        key = (int(entry[4]), _jittor_dtype_name(entry[0].dtype))
        groups.setdefault(key, []).append(index)
    for (steps, _), indices in groups.items():
        step = jt.array(float(steps + 1), dtype="float32").stop_grad()
        count = len(indices)
        out = jt.fused_adamw(
            [entries[i][0] for i in indices], [entries[i][1] for i in indices],
            [entries[i][2] for i in indices], [entries[i][3] for i in indices],
            step, float(lr), float(beta1), float(beta2), float(weight_decay), float(eps))
        for position, index in enumerate(indices):
            results[index] = (out[position], out[count + position], out[2 * count + position])
    return results


register_kernel("optim.adamw_fused", "cuda", _cuda_fused_adamw_updates, supports=_supports)
