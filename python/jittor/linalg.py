# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Haoyang Peng <2247838039@qq.com>
#     Guowei Yang <471184555@qq.com>
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
from functools import partial
from .nn import ComplexNumber

def complex_inv(x:ComplexNumber):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).

    TODO: Faster Implementation; Check backward.
    """
    assert isinstance(x, ComplexNumber), "complex_inv is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_inv"

    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)

        a = _stack_to_complex(data["inputs"][0])
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, _complex_to_stack(t_a))


    def backward_code(np, data):
        def T(x):
            return np.conj(np.swapaxes(x, -1, -2))
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = _stack_to_complex(data["dout"])
        out = data["outputs"][0]
        mx = _stack_to_complex(data["f_outputs"][0])
        t = -_dot(_dot(T(mx), dout), T(mx))
        np.copyto(out, _complex_to_stack(t))

    lmx = jt.numpy_code(
        x.value.shape,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )

    return ComplexNumber(lmx, is_concat_value=True)

def complex_eig(x:ComplexNumber):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return:w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    assert isinstance(x, ComplexNumber), "complex_eig is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for complex_eig"
    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        a = _stack_to_complex(data["inputs"][0])
        w, v = data["outputs"]
        tw, tv = np.linalg.eig(a)
        np.copyto(w, _complex_to_stack(tw))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        raise NotImplementedError

    sw = x.shape[:-2] + x.shape[-1:] + (2,)
    sv = x.value.shape
    w, v = jt.numpy_code(
        [sw, sv],
        [x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(w, is_concat_value=True), ComplexNumber(v, is_concat_value=True)

def complex_qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q is orthogonal matrix and R is upper-triangle matrix.
    :param x (...,M,M):
    :return:q,r as the result of qr factorization.They are both in the shape of (...,M,M).
    """
    assert isinstance(x, ComplexNumber), "linalg_qr is implemented for nn.ComplexNumber"
    assert x.real.dtype == jt.float32 and x.imag.dtype == jt.float32, "real and imag in ComplexNumber should be jt.float32"
    assert x.shape[-2] == x.shape[-1], "only square matrix is supported for linalg_qr"
    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        a = _stack_to_complex(data["inputs"][0])
        qr = data["outputs"][0]
        Q, R = np.linalg.qr(a)
        QR = np.stack([Q, R], axis=0)
        np.copyto(qr, _complex_to_stack(QR))

    def backward_code(np, data):
        # reference: https://github.com/tencent-quantum-lab/tensorcircuit/blob/master/tensorcircuit/backends/pytorch_ops.py
        def H(x):
            return np.conj(np.swapaxes(x, -1, -2))
        def _TriangularSolve(x, r):
            return H(np.linalg.solve(r, H(x)))
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        _diag = partial(np.einsum, '...ii->...i')

        dout = data["dout"]
        out = data["outputs"][0]
        qr = data["f_outputs"][0]
        dout = _stack_to_complex(dout)
        dq, dr = dout[0], dout[1]
        qr = _stack_to_complex(qr)
        q, r = qr[0], qr[1]


        qdq = _dot(H(q), dq)
        qdq_ = qdq - H(qdq)
        rdr = _dot(r, H(dr))
        rdr_ = rdr - H(rdr)
        tril = np.tril(qdq_ + rdr_)

        grad_a = _dot(q, dr + _TriangularSolve(tril, r))
        grad_b = _TriangularSolve(dq - _dot(q, qdq), r)
        ret = grad_a + grad_b

        m = rdr - H(qdq)
        eyem = np.zeros_like(m)
        _diag(eyem)[:] = _diag(m)
        correction = eyem - np.real(eyem)
        ret = ret + _TriangularSolve(_dot(q, H(correction)), r)
        
        ret = _complex_to_stack(ret)
        np.copyto(out,ret)

    qr = jt.numpy_code(
        (2,) + x.value.shape,
        x.value.dtype,
        [x.value],
        forward_code,
        [backward_code],
    )
    q, r = qr[0], qr[1]
    return ComplexNumber(q, is_concat_value=True), ComplexNumber(r, is_concat_value=True)

def complex_svd(x:ComplexNumber):
    r'''
    calculate the Singular Value Decomposition of x.It follows the below fomula:
    x = usv*
    only support full matrices == False ver now, which means:
    x's shape (...,M,K)
    u's shape (...,M,K)
    s's shape (...,K)
    v's shape (...,K,N)
    where K is min(M,N).
    :param x:
    :return:u,s,v.
    '''
    def forward_code(np, data):
        def _stack_to_complex(x):
            return x[..., 0] + 1j * x[..., 1]
        def _complex_to_stack(x):
            return np.stack([np.real(x), np.imag(x)], axis=-1)
        a = _stack_to_complex(data["inputs"][0])
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, _complex_to_stack(tu))
        np.copyto(s, _complex_to_stack(ts))
        np.copyto(v, _complex_to_stack(tv))

    def backward_code(np, data):
        raise NotImplementedError

    m, n = x.shape[-2:]
    k = min(m, n)
    s1 = list(x.shape)
    s1[-1] = k
    s2 = list(x.shape)
    s2[-2] = k
    s3 = list(x.shape)[:-2]
    s3.append(k)
    s1.append(2)
    s2.append(2)
    s3.append(2)
    u, s, v = jt.numpy_code(
        [s1, s3, s2],
        [x.value.dtype, x.value.dtype, x.value.dtype],
        [x.value],
        forward_code,
        [backward_code],
    )
    return ComplexNumber(u, is_concat_value=True), \
            ComplexNumber(s, is_concat_value=True), \
            ComplexNumber(v, is_concat_value=True)

#TODO:full_matrices=1
def svd(x):
    r'''
    calculate the Singular Value Decomposition of x.It follows the below fomula:
    x = usv*
    only support full matrices == False ver now, which means:
    x's shape (...,M,K)
    u's shape (...,M,K)
    s's shape (...,K)
    v's shape (...,K,N)
    where K is min(M,N).
    :param x:
    :return:u,s,v.
    '''
    if isinstance(x, ComplexNumber):
        return complex_svd(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        u, s, v = data["outputs"]
        #TODO:remove copyto
        tu, ts, tv = np.linalg.svd(a, full_matrices=0)
        np.copyto(u, tu)
        np.copyto(s, ts)
        np.copyto(v, tv)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
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

    m, n = x.shape[-2:]
    k = min(m, n)
    s1 = list(x.shape)
    s1[-1] = k
    s2 = list(x.shape)
    s2[-2] = k
    s3 = list(x.shape)[:-2]
    s3.append(k)
    u, s, v = jt.numpy_code(
        [s1, s3, s2],
        [x.dtype, x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return u, s, v

def eig(x):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return (ComplexNumber):w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    if isinstance(x, ComplexNumber):
        return complex_eig(x)
    return complex_eig(ComplexNumber(x))

def eigh(x):
    r"""
    calculate the eigenvalues and eigenvectors of x.
    :param x (...,M,M):
    :return:w, v.
    w (...,M) : the eigenvalues.
    v (...,M,M) : normalized eigenvectors.
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        w, v = data["outputs"]
        tw, tv = np.linalg.eigh(a, UPLO='L')
        np.copyto(w, tw)
        np.copyto(v, tv)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
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

    sw = x.shape[:-2] + x.shape[-1:]
    sv = x.shape
    w, v = jt.numpy_code(
        [sw, sv],
        [x.dtype, x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return w, v


def inv(x):
    r"""
    calculate the inverse of x.
    :param x (...,M,M):
    :return:x^-1 (...,M,M).
    """
    if isinstance(x, ComplexNumber):
        return complex_inv(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.inv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
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
    r"""
    calculate the pseudo-inverse of a x.
    :param x (...,M,N)
    :return: x's pinv (...N,M)
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        m_a = data["outputs"][0]
        t_a = np.linalg.pinv(a)
        np.copyto(m_a, t_a)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
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
    sw = list(x.shape[:-2]) + [x.shape[-1]] + [x.shape[-2]]
    lmx = jt.numpy_code(
        [sw],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    mx = lmx[0]
    return mx


def det(x):
    r"""
    calculate the determinant of x.
    :param x (...,M,M):
    :return:|x| (...,1)
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.det(a)
        np.copyto(L, tL)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        inp = data["inputs"][0]
        n_d = np.reshape(dout, np.shape(dout) + (1, 1))
        n_o = np.reshape(f_out, np.shape(f_out) + (1, 1))
        s = n_d * n_o * T(np.linalg.inv(inp))
        np.copyto(out, s)

    s = x.shape
    x_s = s[:-2]
    if len(s) == 2:
        x_s.append(1)
    l_det = jt.numpy_code(
        [x_s],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    det = l_det[0]
    return det


def slogdet(x):
    r"""
    calculate the sign and log of the determinant of x.
    :param x (...,M,M):
    :return sign, x's logdet.
    sign array decides the sign of determinant and their values can be -1,0,1.Only Real number now.0 means det is 0 and logdet is -inf.
    logdet in shape (...,1).
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        sign, m_a = data["outputs"]
        sign_, t_a = np.linalg.slogdet(a)
        np.copyto(m_a, t_a)
        np.copyto(sign, sign_)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
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

    s = x.shape
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


def cholesky(x):
    r"""
    do Cholesky decomposition of x in the form of below formula:
    x = LL^T
    x must be a Hermite and positive-definite matrix. L is a lower-triangular matrix.
    :param x (...,M,M):
    :return: L (...,M,M).
    """
    def forward_code(np, data):
        a = data["inputs"][0]
        L = data["outputs"][0]
        tL = np.linalg.cholesky(a)
        np.copyto(L, tL)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        solve_trans = lambda a, b: np.linalg.solve(T(a), b)
        phi = lambda X: np.tril(X) / (1. + np.eye(X.shape[-1]))

        def conjugate_solve(L, X):
            return solve_trans(L, T(solve_trans(L, T(X))))

        s = conjugate_solve(f_out, phi(np.einsum('...ki,...kj->...ij', f_out, dout)))
        s = (s + T(s)) / 2.
        np.copyto(out, s)

    lL = jt.numpy_code(
        [x.shape],
        [x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    L = lL[0]
    return L


def solve(a,b):
    r"""
    Solve a linear matrix equation Ax = B.This is done by calculating x = A^-1B.So A must not be singular.
    :param a:(...,M,M)
    :param b:(...,M)
    :return:solution of Ax = b formula.x in the shape of (...M)
    """
    def forward_code(np, data):
        a, b = data["inputs"]
        L = data["outputs"][0]
        ans = np.linalg.solve(a, b)
        np.copyto(L, ans)

    def backward_code1(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        dout = data["dout"]
        out = data["outputs"][0]
        f_out = data["f_outputs"][0]
        inp = data["inputs"][0]
        updim = lambda x: x if x.ndim == a.ndim else x[..., None]
        t = -_dot(updim(np.linalg.solve(T(inp), dout)), T(updim(f_out)))
        np.copyto(out, t)

    def backward_code2(np, data):
        out = data["outputs"][0]
        np.copyto(out, 0)

    l_ans = jt.numpy_code(
        [b.shape],
        [b.dtype],
        [a, b],
        forward_code,
        [backward_code1, backward_code2],
    )
    ans = l_ans[0]
    return ans


def qr(x):
    r"""
    do the qr factorization of x in the below formula:
    x = QR where Q is orthogonal matrix and R is upper-triangle matrix.
    :param x (...,M,M):
    :return:q,r as the result of qr factorization.They are both in the shape of (...,M,M).
    """
    if isinstance(x, ComplexNumber):
        return complex_qr(x)
    def forward_code(np, data):
        a = data["inputs"][0]
        q, r = data["outputs"]
        Q, R = np.linalg.qr(a)
        np.copyto(q,Q)
        np.copyto(r,R)

    def backward_code(np, data):
        def T(x):
            return np.swapaxes(x, -1, -2)
        _dot = partial(np.einsum, '...ij,...jk->...ik')
        _harmard = partial(np.einsum, '...ij,...ij->...ij')
        dout = data["dout"]
        out = data["outputs"][0]
        q, r = data["f_outputs"]
        out_index = data["out_index"]
        #pl = np.tril(np.ones((inp.shape[-1],inp.shape[-1])))-diags
        if out_index == 0: # Q_TERM
            q_t = _dot(T(q),dout)
            rhs_solve = q_t - T(q_t)
            rhs_solve = T(np.tril(rhs_solve,-1))
            qsolve = np.linalg.solve(r,rhs_solve)
            qsolve = T(qsolve)
            tq = _dot(q,qsolve)
            np.copyto(out,tq)
        else: #R_TERM
            r_t = _dot(r ,T(dout))
            rhs_solve = r_t - T(r_t)
            rhs_solve = np.tril(rhs_solve,-1)
            rhs_solve = T(rhs_solve)
            r_solve = np.linalg.solve(r,rhs_solve)
            tr = _dot(q,(T(r_solve) + dout))
            np.copyto(out,tr)

    q, r = jt.numpy_code(
        [x.shape,x.shape],
        [x.dtype,x.dtype],
        [x],
        forward_code,
        [backward_code],
    )
    return q, r


def einsum(string, *args):
    r"""
    do the einsum operation. Using the implementation in https://github.com/HIPS/autograd
    :param string, args:
    :return: return values depend on the input string kinds.
    """
    import numpy as np_cpu
    if string == "i,j->ij":
        return args[0].broadcast((args[0].shape[0], args[1].shape[0]), dims=[1]).multiply(args[1])
    def forward_code(np, data):
        out = data["outputs"][0]
        npout = np.einsum(string, *data["inputs"])
        np.copyto(out, npout)

    def backward_code(np, data, argnum=0):
        real_len = len(data["inputs"]) - 2
        operands = data["inputs"][:real_len]
        _ops = operands
        if np_cpu is not np:
            # fake a numpy array
            _ops = [ np_cpu.zeros((1,)*o.ndim) for o in _ops ]
        in_subs, out_subs, _ = np_cpu.core.einsumfunc._parse_einsum_input([string] + _ops)
        dout = data["dout"]
        out_index = data["out_index"]
        out = data["outputs"][0]
        inp = data["inputs"][argnum]
        c = data["f_outputs"]

        in_subs_list = in_subs.split(',')
        op_num = argnum
        subs_wrt = in_subs_list[op_num]
        rest_of_ops = operands[:op_num] + operands[op_num+1:]
        rest_of_subs = in_subs_list[:op_num] + in_subs_list[op_num+1:]
        other_named_subs = set(''.join([out_subs] + rest_of_subs))
        naked_summed = [(i, sub) for i, sub in enumerate(subs_wrt)
                        if sub not in other_named_subs]
        if naked_summed:
            naked_summed_dims, ones_subs = zip(*naked_summed)
            ones_subs = ''.join(ones_subs)
            ones = np_cpu.ones(np_cpu.array(operands[op_num].shape)[list(naked_summed_dims)])
            new_input_subs = ','.join([out_subs, ones_subs] + rest_of_subs)
            new_operands = [dout, ones] + rest_of_ops
        else:
            new_input_subs = ','.join([out_subs] + rest_of_subs)
            new_operands = [dout] + rest_of_ops

        new_subscripts = new_input_subs + '->' + subs_wrt
        x = np.einsum(new_subscripts, *new_operands)
        while np.ndim(x) > np.ndim(inp):
            x = np.sum(x, axis=broadcast_idx)
            for axis, size in enumerate(inp.shape):
                if size == 1:
                    x = np.sum(x, axis=axis, keepdims=True)
        np.copyto(out, x)
    
    def einsum_outshape(einsum_expr, inputs):
        shps = np_cpu.concatenate([in_.shape for in_ in inputs])
        p = einsum_expr.replace(" ", "").split(',')
        s = p[:-1] + p[-1].split('->')
        rec_shape = []
        ellip_expr = None
        const_rep = '1234567890' # assume tensor shape no more than 10 dimensions
        for idx, expr in enumerate(s[:-1]):
            if "..." in expr:
                assert "..." in s[-1]
            else:
                continue
            shp = inputs[idx].shape
            ellipsis_pos = len(expr.replace("...", ""))
            nellip_expr = const_rep[0 : len(shp) - ellipsis_pos]
            if ellip_expr is None:
                ellip_expr = nellip_expr
            else:
                assert ellip_expr == nellip_expr, "Please keep broadcast ellipsis record the same ellipsis."
            s[idx] = expr.replace("...", ellip_expr)
        if ellip_expr:
            s[-1] = s[-1].replace("...", ellip_expr)
        if s[-1]=='':
            return ()
        else:
            inop = list(map(list,s))
            return tuple(shps[(np_cpu.concatenate(inop[:-1])[:,None]==inop[-1]).argmax(0)].astype(np_cpu.int64))

    output_shape = [int(x) for x in einsum_outshape(string, args)]
    backwards = [partial(backward_code, argnum=idx) for idx in range(len(args))]
    a = jt.numpy_code(
        [output_shape],
        [args[0].dtype],
        args,
        forward_code,
        backwards,
    )[0]
    return a