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
from jittor._runtime.dispatch import register_kernel

_DTYPES = frozenset(("float32", "float16", "bfloat16"))
# Keep each mapped op small enough for nvcc and the JIT argument table.  The
# CUDA kernel can process up to 36 tensors per launch; batching 32 entries
# per op halves launch and graph-node overhead vs 16 while keeping JIT
# compilation bounded.
_MAX_FUSED_ENTRIES = 32
_NATIVE_DTYPE_GETTER = None
_ALLOWED_NATIVE_DTYPES = None


def _bind_native_dtypes():
    global _NATIVE_DTYPE_GETTER, _ALLOWED_NATIVE_DTYPES
    cls = jt.core.Var
    getter = getattr(cls, "_frontend_native_dtype", None) or cls.__dict__["dtype"]
    _NATIVE_DTYPE_GETTER = getter.__get__
    _ALLOWED_NATIVE_DTYPES = (
        jt.core.NanoString("float32"),
        jt.core.NanoString("float16"),
        jt.core.NanoString("bfloat16"),
    )


def _supports(entries, *args, **kwargs):
    """Dense tensors of one supported dtype per entry: the kernel writes raw pointers."""
    if not entries:
        return False
    get_dtype = _NATIVE_DTYPE_GETTER
    if get_dtype is None:
        _bind_native_dtypes()
        get_dtype = _NATIVE_DTYPE_GETTER
    f32, f16, bf16 = _ALLOWED_NATIVE_DTYPES
    for parameter, moment, variance, gradient, _ in entries:
        dtype = get_dtype(parameter)
        if dtype != f32 and dtype != f16 and dtype != bf16:
            return False
        if (not parameter._storage_is_contiguous()
                or get_dtype(moment) != dtype or not moment._storage_is_contiguous()
                or get_dtype(variance) != dtype or not variance._storage_is_contiguous()
                or get_dtype(gradient) != dtype or not gradient._storage_is_contiguous()):
            return False
    return True


def _cuda_fused_adamw_updates(entries, lr, beta1, beta2, weight_decay, eps):
    """(new parameter, new moment, new variance) per entry, in entry order.

    An entry is (parameter, moment, variance, gradient, steps taken so far);
    this is the update for step ``steps + 1``, as the per-parameter path
    computes it. Entries are grouped by step and dtype, one operator each.

    ``lr`` may instead be a two-element float32 Var ``[step, lr]``, which the
    kernel reads on the device and which then stands for every entry's step:
    what a captured step passes, so that a replay advances the count and
    follows the learning rate without the optimizer running.
    A seven-element header also carries found-inf, both betas, weight decay
    and epsilon, allowing schedulers to change them without recapture.
    """
    hyper = lr if isinstance(lr, jt.Var) else None
    results = [None] * len(entries)
    get_dtype = _NATIVE_DTYPE_GETTER
    if get_dtype is None:
        _bind_native_dtypes()
        get_dtype = _NATIVE_DTYPE_GETTER
    f32, f16, bf16 = _ALLOWED_NATIVE_DTYPES
    groups = {}
    for index, entry in enumerate(entries):
        dt = get_dtype(entry[0])
        dt_key = 0 if dt == f32 else (1 if dt == f16 else (2 if dt == bf16 else str(dt)))
        key = (0 if hyper is not None else int(entry[4]), dt_key)
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [index]
        else:
            bucket.append(index)
    for (steps, _), indices in groups.items():
        if hyper is not None:
            step, lr = hyper, 0.0
        else:
            step = jt.array(float(steps + 1), dtype="float32").stop_grad()
        for start in range(0, len(indices), _MAX_FUSED_ENTRIES):
            batch = indices[start:start + _MAX_FUSED_ENTRIES]
            count = len(batch)
            out = jt.fused_adamw(
                [entries[i][0] for i in batch], [entries[i][1] for i in batch],
                [entries[i][2] for i in batch], [entries[i][3] for i in batch],
                step, float(lr), float(beta1), float(beta2), float(weight_decay), float(eps))
            for position, index in enumerate(batch):
                results[index] = (out[position], out[count + position], out[2 * count + position])
    return results


#: Takes ``[step, lr]`` as a device Var in place of the learning rate.
_cuda_fused_adamw_updates.accepts_live_step = True
_cuda_fused_adamw_updates.accepts_live_hyperparameters = True

register_kernel("optim.adamw_fused", "cuda", _cuda_fused_adamw_updates, supports=_supports)
