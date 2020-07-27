import jittor as jt
import numpy as np

#TODO:full_matrices=1
def svd(x):
    from functools import partial
    import copy
    def T(x):
        return np.swapaxes(x, -1, -2)
    _dot = partial(np.einsum, '...ij,...jk->...ik')

    def forward_code(np, data):
        a = data["inputs"][0]
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, tu)
        np.copyto(s, ts)
        np.copyto(v, tv)

    def backward_code(np, data):
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        u, s, v = data["f_outputs"]
        v = T(v)
        m, n = inp.shape[-2:]
        k = np.min((m, n))
        i = np.reshape(np.eye(k), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (k, k))))
        if out_index == 0:
            f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
            gu = dout
            utgu = _dot(T(u), gu)
            t = (f * (utgu - T(utgu))) * s[..., np.newaxis, :]
            t = _dot(_dot(u, t), T(v))
            if m > n:
                i_minus_uut = (np.reshape(np.eye(m), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (m, m)))) -
                               _dot(u, np.conj(T(u))))
                t = t + T(_dot(_dot(v / s[..., np.newaxis, :], T(gu)), i_minus_uut))
            np.copyto(out, t)
        elif out_index == 1:
            gs = dout
            t = i * gs[..., :, np.newaxis]
            t = _dot(_dot(u, t), T(v))
            np.copyto(out, t)
        elif out_index == 2:
            f = 1 / (s[..., np.newaxis, :] ** 2 - s[..., :, np.newaxis] ** 2 + i)
            gv = dout
            vtgv = _dot(T(v), gv)
            t = s[..., :, np.newaxis] * (f * (vtgv - T(vtgv)))
            t = _dot(_dot(u, t), T(v))
            if m < n:
                i_minus_vvt = (np.reshape(np.eye(n), np.concatenate((np.ones(inp.ndim - 2, dtype=int), (n, n)))) -
                               _dot(v, np.conj(T(v))))
                t = t + T(_dot(_dot(u / s[..., np.newaxis, :], T(gv)), i_minus_vvt))
            np.copyto(out, t)

    s = jt.array(x.shape).data.tolist()
    m, n = x.shape[-2:]
    k = np.min((m, n))
    k = int(k)
    s1 = copy.deepcopy(s)
    s1[-1] = k
    s2 = copy.deepcopy(s)
    s2[-2] = k
    s3 = s[:-2]
    s3.append(k)
    u, s, v = jt.numpy_code(
        [s1, s3, s2],
        [x.dtype, x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return u, s, v

def eigh(x):
    from functools import partial
    import copy
    def T(x):
        return np.swapaxes(x, -1, -2)
    _dot = partial(np.einsum, '...ij,...jk->...ik')

    def forward_code(np, data):
        a = data["inputs"][0]
        w, v = data["outputs"]
        tw, tv = np.linalg.eigh(a, UPLO='L')
        np.copyto(w, tw)
        np.copyto(v, tv)

    def backward_code(np, data):
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        w, v = data["f_outputs"]
        k = int(inp.shape[-1])
        w_repeated = np.repeat(w[..., np.newaxis], k, axis=-1)
        if out_index == 0:
            t = _dot(v * dout[..., np.newaxis, :], T(v))
            np.copyto(out, t)
        elif out_index == 1:
            if np.any(dout):
                off_diag = np.ones((k, k)) - np.eye(k)
                F = off_diag / (T(w_repeated) - w_repeated + np.eye(k))
                t = _dot(_dot(v, F * _dot(T(v), dout)), T(v))
                np.copyto(out, t)

    s = jt.array(x.shape).data.tolist()
    sw = s[:-2]
    sw.append(s[-1])
    sv = copy.deepcopy(s)
    w, v = jt.numpy_code(
        [sw, sv],
        [x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return w, v

def inv(x):
    from functools import partial
    def T(x):
        return np.swapaxes(x, -1, -2)
    _dot = partial(np.einsum, '...ij,...jk->...ik')

    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        dout = data["dout"]
        out = data["outputs"][0]
        lmx = data["f_outputs"]
        mx = lmx[0]
        t = -_dot(_dot(T(mx), dout), T(mx))
        np.copyto(out, t)

    lmx = jt.numpy_code(
        [x.shape],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    mx = lmx[0]
    return mx

def pinv(x):
    from functools import partial
    def T(x):
        return np.swapaxes(x, -1, -2)
    _dot = partial(np.einsum, '...ij,...jk->...ik')

    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        lmx = data["f_outputs"]
        mx = lmx[0]
        t = T(
            -_dot(_dot(mx, T(dout)), mx)
            + _dot(_dot(_dot(mx, T(mx)), dout), np.eye(inp.shape[-2]) - _dot(inp, mx))
            + _dot(_dot(_dot(np.eye(mx.shape[-2]) - _dot(mx, inp), dout), T(mx)), mx)
        )
        np.copyto(out, t)

    lmx = jt.numpy_code(
        [x.shape],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    mx = lmx[0]
    return mx

def slogdet(x):
    from functools import partial
    def T(x):
        return np.swapaxes(x, -1, -2)
    _dot = partial(np.einsum, '...ij,...jk->...ik')
    def forward_code(np, data):
        a = data["inputs"][0]
        sign, m_a = data["outputs"]
        sign_, t_a = np.linalg.slogdet(a)
        np.copyto(m_a, t_a)
        np.copyto(sign, sign_)

    def backward_code(np, data):
        dout = data["dout"]
        out = data["outputs"][0]
        inp = data["inputs"][0]
        out_index = data["out_index"]
        if out_index == 0:
            np.copyto(out, 0)
        if out_index == 1:
            t = np.reshape(dout, np.shape(dout) + (1, 1))
            t = t * T(np.linalg.inv(inp))
            np.copyto(out, t)

    s = jt.array(x.shape).data.tolist()
    det_s = s[:-2]
    if len(det_s) == 0:
        det_s.append(1)
    sign, mx = jt.numpy_code(
        [det_s, det_s],
        [x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return sign, mx
