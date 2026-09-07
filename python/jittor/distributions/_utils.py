# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************


import math


def _norm_sample_shape(sample_shape):
    ''' Normalize a torch-style sample_shape (None / int / tuple / list /
    jt.NanoVector / torch.Size) to a plain tuple of ints. '''
    if sample_shape is None:
        return ()
    if isinstance(sample_shape, int):
        return (sample_shape,)
    return tuple(int(s) for s in sample_shape)

def _prod(shape):
    p = 1
    for d in shape:
        p *= d
    return p

def _bshape(*params):
    ''' Broadcast the parameter shapes to obtain batch_shape (torch semantics).
    A single-element parameter (python number, or a length-1 Var that jittor uses
    to stand in for a 0-d scalar) contributes () -- see the module note: jittor has
    no 0-d Var so a scalar and a 1-element batch are indistinguishable, and we pick
    the scalar reading so Normal(jt.array(0.5), ...).sample((n,)) is (n,), not (n,1).'''
    shapes = []
    for p in params:
        if hasattr(p, "shape"):
            s = tuple(p.shape)
            shapes.append(() if _prod(s) == 1 else s)   # length-1 Var == scalar
        else:
            shapes.append(())                            # python number == scalar
    out = ()
    for s in shapes:
        out = _broadcast_two(out, s)
    return out

def _broadcast_two(a, b):
    ''' numpy/torch broadcast of two shape tuples. '''
    res = []
    for i in range(1, max(len(a), len(b)) + 1):
        da = a[-i] if i <= len(a) else 1
        db = b[-i] if i <= len(b) else 1
        if da == 1:
            res.append(db)
        elif db == 1 or da == db:
            res.append(da)
        else:
            raise ValueError(f"incompatible parameter shapes for broadcast: {a} vs {b}")
    return tuple(reversed(res))

def _full_shape(sample_shape, batch_shape, event_shape=()):
    ''' torch's sample_shape + batch_shape + event_shape. A scalar (empty
    batch+event) collapses to (1,) because jittor has no 0-d Var. '''
    out = _norm_sample_shape(sample_shape) + tuple(batch_shape) + tuple(event_shape)
    return out if len(out) > 0 else (1,)

def _broadcast_var(value, shape):
    import jittor as jt
    value = value if isinstance(value, jt.Var) else jt.array(value)
    cur = tuple(value.shape)
    if cur == tuple(shape) or not shape:
        return value
    if _prod(cur) == 1:
        value = value.reshape((1,) * len(shape))
    return value.broadcast(shape)

def broadcast_all(*values):
    batch_shape = _bshape(*values)
    return tuple(_broadcast_var(value, batch_shape) for value in values)

def simple_presum(x):
    import jittor as jt
    src = '''
__inline_static__
@python.jittor.auto_parallel(1)
void kernel(int n0, int i0, in0_type* x, in0_type* out, int nl) {
    out[i0*(nl+1)] = 0;
    for (int i=0; i<nl; i++)
        out[i0*(nl+1)+i+1] = out[i0*(nl+1)+i] + x[i0*nl+i];
}
kernel(in0->num/in0->shape[in0->shape.size()-1], 0, in0_p, out0_p, in0->shape[in0->shape.size()-1]);
    '''
    return jt.code(x.shape[:-1]+(x.shape[-1]+1,), x.dtype, [x],
        cpu_src=src, cuda_src=src)

def _logsigmoid(z):
    # stable log(sigmoid(z)) = min(z,0) - log(1+exp(-|z|))
    import jittor as jt
    return jt.minimum(z, 0.0) - jt.safe_log(1.0 + jt.exp(-jt.abs(z)))

def _softplus(z):
    # stable log(1+exp(z)) = max(z,0) + log(1+exp(-|z|))
    import jittor as jt
    return jt.maximum(z, 0.0) + jt.safe_log(1.0 + jt.exp(-jt.abs(z)))

def _log_temperature(temperature):
    import jittor as jt
    if isinstance(temperature, jt.Var):
        return jt.safe_log(temperature)
    return math.log(float(temperature))

def _no_closed_form(cls_name, name):
    raise NotImplementedError(
        f"{cls_name}.{name} has no closed form (torch.distributions raises here "
        f"too). The discrete parent's {name} describes a different random "
        f"variable and would be silently wrong.")

_LOG2PI = math.log(2 * math.pi)

def _as_var(x):
    import jittor as jt
    return x if isinstance(x, jt.Var) else jt.array(x, dtype="float32")

def _lgamma(x):
    from jittor import lgamma, digamma
    return lgamma.apply(_as_var(x))

def _digamma(x):
    from jittor import lgamma, digamma
    return digamma.apply(_as_var(x))
