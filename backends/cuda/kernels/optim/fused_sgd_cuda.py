"""One kernel launch for a whole parameter list's SGD update.

The portable update is two elementwise ops per parameter plus a holder
rebind, and a transformer has a lot of parameters: an 8-layer d512 model has
96, and the update loop measured 1.11 ms of a 7.41 ms training step -- the
optimizer, not the model. PyTorch does not pay that because its SGD is a
`foreach` kernel: one launch for the whole list.

This is the same idea. Every parameter's pointer, gradient pointer, output
pointer and length travel in one by-value argument struct, and one kernel
walks them all. The struct is what bounds the batch: CUDA gives a kernel 4 KB
of parameter space, so the tensors are processed in chunks that fit.
"""

import jittor as jt
from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import register_kernel


#: Tensors per launch. Each one costs three pointers and a length in the
#: argument struct (28 bytes), and a kernel's parameter space is 4 KB; 96 is
#: 2688 bytes, which leaves room for the scalars and the ABI's own overhead.
_CHUNK = 96


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


def _source(count, momentum, weight_decay, dampening, nesterov):
    """File-scope CUDA for exactly this configuration.

    The coefficients are baked in as literals rather than passed: they do not
    change between steps, and a branch on `momentum == 0` inside the inner
    loop would be evaluated once per element.
    """
    plain = momentum == 0 and dampening == 0 and not nesterov
    wd = f"{float(weight_decay):.9e}f"
    mom = f"{float(momentum):.9e}f"
    damp = f"{float(dampening):.9e}f"
    # dp: the gradient, with weight decay folded in when there is any.
    dp = "g" if weight_decay == 0 else f"fmaf(p, {wd}, g)"
    if plain:
        body = f"""
                float p = arg.param[t][i];
                float g = arg.grad[t][i];
                float dp = {dp};
                arg.dst[t][i] = p - dp * lr;"""
    else:
        step = f"arg.vel[t][i] = v = fmaf({mom}, arg.vel[t][i], dp * (1.0f - {damp}));"
        use = f"fmaf({mom}, v, dp)" if nesterov else "v"
        body = f"""
                float p = arg.param[t][i];
                float g = arg.grad[t][i];
                float dp = {dp};
                float v;
                {step}
                arg.dst[t][i] = p - ({use}) * lr;"""
    vel = "" if plain else f"float* vel[{count}];"
    # No member may be called `out`: the code op's JIT template does
    # `#define out out0` around the body, so `args.out[k]` would be rewritten
    # to `args.out0[k]` and nvcc reports a struct with no such member.
    return f"""
    struct FusedSgdArgs {{
        float* param[{count}];
        float* grad[{count}];
        float* dst[{count}];
        {vel}
        int len[{count}];
    }};
    __global__ static void fused_sgd_kernel(FusedSgdArgs arg, const float* lr_ptr, float lr) {{
        // A captured step passes the rate as a device Var, which a replay
        // refreshes; otherwise it is the literal below.
        if (lr_ptr) lr = *lr_ptr;
        const int t = blockIdx.y;
        const int n = arg.len[t];
        const int stride = blockDim.x * gridDim.x;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {{
            {body}
        }}
    }}
    """


def _launch(count, plain):
    vel = "" if plain else "        args.vel[k] = nullptr;\n"
    return vel


def _fused_sgd_cuda(entries, lr, momentum, weight_decay, dampening, nesterov):
    """`entries` is a list of (param, grad, velocity). Returns [(new_p, new_v)].

    `lr` may be a one-element float32 Var instead of a number: the kernel then
    reads the rate on the device (see `accepts_live_lr`).
    """
    plain = momentum == 0 and dampening == 0 and not nesterov
    live = lr if isinstance(lr, jt.Var) else None
    results = []
    for start in range(0, len(entries), _CHUNK):
        chunk = entries[start:start + _CHUNK]
        count = len(chunk)
        params = [e[0] for e in chunk]
        grads = [e[1] for e in chunk]
        vels = [e[2] for e in chunk]
        inputs = params + grads + ([] if plain else vels)
        lr_ptr = "nullptr"
        if live is not None:
            lr_ptr = f"in{len(inputs)}_p"
            inputs = inputs + [live]
        setup = []
        for k in range(count):
            setup.append(f"args.param[{k}] = in{k}_p;")
            setup.append(f"args.grad[{k}] = in{count + k}_p;")
            setup.append(f"args.dst[{k}] = out{k}_p;")
            if not plain:
                setup.append(f"args.vel[{k}] = in{2 * count + k}_p;")
            setup.append(f"args.len[{k}] = {int(params[k].numel())};")
        longest = max(int(p.numel()) for p in params)
        blocks = max(1, min(256, (longest + 255) // 256))
        # The struct and the kernel go in the header: `cuda_src` is spliced
        # into the body of `CodeOp::jit_run`, and a definition there is block
        # scope -- nvcc answers "a block-scope function may only have extern
        # storage class". Only the launch belongs in `cuda_src`.
        header = _source(count, momentum, weight_decay, dampening, nesterov)
        body = f"""
        FusedSgdArgs args;
        {chr(10).join('        ' + line for line in setup)}
        fused_sgd_kernel<<<dim3({blocks}, {count}), 256>>>(
            args, {lr_ptr}, {0.0 if live is not None else float(lr):.9e}f);
        """
        outs = jt.code(
            [p.shape for p in params],
            [p.dtype for p in params],
            inputs,
            cuda_header=header,
            cuda_src=body,
        )
        if not isinstance(outs, (list, tuple)):
            outs = [outs]
        # The velocity is written in place by the kernel, so it is handed back
        # unchanged rather than as a new Var.
        results.extend(zip(outs, vels))
    return results


#: Takes the learning rate as a device Var as well as a number.
_fused_sgd_cuda.accepts_live_lr = True

register_kernel("optim.sgd_fused", "cuda", _fused_sgd_cuda,
                dtypes=("float32",), supports=_supports_fused_sgd)

__all__ = ["_fused_sgd_cuda"]
