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
"""Linear algebra public API, re-exported from its domain owners."""
from .decompositions import (
    svd,
    svdvals,
    eig,
    eigh,
    eigvalsh,
    cholesky,
    qr,
)
from .solving import (
    inv,
    inv_ex,
    pinv,
    matrix_power,
    det,
    slogdet,
    solve,
)
from .norms import (
    matrix_rank,
    matrix_norm,
    vector_norm,
    norm,
    cond,
)
from .contractions import (
    einsum,
)
from .results import INVEX, SVD

_COMPLEX_EXPORTS = (
    "complex_inv", "complex_eig", "complex_eigh", "complex_qr",
    "complex_svd", "complex_pinv",
)


def __getattr__(name):
    # Legacy ComplexNumber annotations need nn, which imports this facade.
    if name in _COMPLEX_EXPORTS:
        from importlib import import_module
        return getattr(import_module(__name__ + ".complex"), name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def __dir__():
    return sorted(set(globals()) | set(__all__))


__all__ = [
    'complex_inv',
    'complex_eig',
    'complex_eigh',
    'complex_qr',
    'complex_svd',
    'complex_pinv',
    'svd',
    'svdvals',
    'eig',
    'eigh',
    'eigvalsh',
    'inv',
    'inv_ex',
    'pinv',
    'matrix_power',
    'matrix_rank',
    'matrix_norm',
    'vector_norm',
    'norm',
    'cond',
    'det',
    'slogdet',
    'cholesky',
    'solve',
    'qr',
    'einsum',
    "INVEX",
    "SVD",
]
