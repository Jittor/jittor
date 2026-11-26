"""ROCm multi-tensor AdamW for FP32 master parameters."""

import math

import jittor as jt


_MAX_ENTRIES = 32
_dtype_getter = None
_float32 = None


def _supports_adamw(entries, *args, **kwargs):
    if not entries:
        return False
    global _dtype_getter, _float32
    if _dtype_getter is None:
        cls = jt.core.Var
        getter = getattr(cls, "_frontend_native_dtype", None) or cls.__dict__["dtype"]
        _dtype_getter = getter.__get__
        _float32 = jt.core.NanoString("float32")
    for p, m, v, g, _ in entries:
        if (not p._storage_is_contiguous()
                or any(_dtype_getter(t) != _float32 for t in (p, m, v, g))):
            return False
    return True


def _dense(tensor):
    return tensor if tensor._storage_is_contiguous() else tensor.contiguous()


def fused_adamw_updates(entries, lr, beta1, beta2, weight_decay, eps):
    """Return the three updated tensors for each entry in input order."""
    rate, beta1, beta2 = float(lr), float(beta1), float(beta2)
    decay, eps = float(weight_decay), float(eps)
    buckets = {}
    results = [None] * len(entries)
    for index, entry in enumerate(entries):
        buckets.setdefault(int(entry[4]) + 1, []).append(index)
    for step, indices in buckets.items():
        # Compute the sensitive bias corrections in host FP64. The accelerator
        # consumes only FP32 scalars, including on devices without FP64 math.
        header = jt.array([step, rate, 0., beta1, beta2, decay, eps,
                           rate / (1. - beta1 ** step),
                           math.sqrt(1. - beta2 ** step)], dtype="float32").stop_grad()
        for start in range(0, len(indices), _MAX_ENTRIES):
            batch = indices[start:start + _MAX_ENTRIES]
            count = len(batch)
            out = jt.fused_adamw(
                [entries[i][0] for i in batch],
                [_dense(entries[i][1]) for i in batch],
                [_dense(entries[i][2]) for i in batch],
                [_dense(entries[i][3]) for i in batch],
                header, rate, beta1, beta2, decay, eps)
            for pos, index in enumerate(batch):
                results[index] = (out[pos], out[count + pos], out[2 * count + pos])
    return results


def install_kernels():
    from jittor._runtime.dispatch import register_kernel
    register_kernel("optim.adamw_fused", "rocm", fused_adamw_updates,
                    supports=_supports_adamw, priority=100)
