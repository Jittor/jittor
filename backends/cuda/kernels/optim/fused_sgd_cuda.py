"""One kernel launch for a whole parameter list's SGD update.

The portable update is two elementwise ops per parameter plus a holder
rebind, and a transformer has a lot of parameters: an 8-layer d512 model has
96, and the update loop measured 1.11 ms of a 7.41 ms training step -- the
optimizer, not the model. PyTorch does not pay that because its SGD is a
`foreach` kernel: one launch for the whole list.

This is the same idea: ``fused_sgd`` (src/ops/composite/fused_sgd_op.cc)
walks the whole list from one argument table per launch, and writes the
parameters and velocities in place.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import register_kernel


def _supports_fused_sgd(tensors, *args, **kwargs):
    """float32, dense, and allocated: this writes raw pointers.

    `tensors` is every Var the kernel dereferences -- parameters, gradients and
    velocities -- not the parameter list. The kernel's argument struct declares
    all three families as `float*`, so a float16 gradient against a float32
    parameter is a compile error, not a slow path, and that pair is exactly what
    `auto_mixed_precision_level` 4/5/6 produce. See the call site in
    `jittor/optim/algorithms/sgd.py`.
    """
    if not tensors:
        return False
    for p in tensors:
        if not isinstance(p, jt.Var):
            return False
        if _jittor_dtype_name(p.dtype) != "float32":
            return False
        if not p._storage_is_contiguous():
            return False
    return True


def _fused_sgd_cuda(entries, lr, momentum, weight_decay, dampening, nesterov):
    """`entries` is a list of (param, grad, velocity). Returns [(new_p, new_v)].

    `lr` may be a one-element float32 Var instead of a number: the kernel then
    reads the rate on the device (see `accepts_live_lr`).

    ``fused_sgd`` (src/ops/composite/fused_sgd_op.cc) updates the parameters
    and velocities in place: its outputs share the inputs' storage. The
    velocity is handed back as the same Var, so nothing rebinds it.
    """
    live = [lr] if isinstance(lr, jt.Var) else []
    params = [e[0] for e in entries]
    grads = [e[1] for e in entries]
    vels = [e[2] for e in entries]
    outs = jt.fused_sgd(params, vels, grads, 0.0 if live else float(lr),
                        float(momentum), float(weight_decay), float(dampening),
                        bool(nesterov), False, live)
    return list(zip(outs[:len(entries)], vels))


#: Takes the learning rate as a device Var as well as a number.
_fused_sgd_cuda.accepts_live_lr = True

register_kernel("optim.sgd_fused", "cuda", _fused_sgd_cuda,
                dtypes=("float32",), supports=_supports_fused_sgd)

__all__ = ["_fused_sgd_cuda"]
