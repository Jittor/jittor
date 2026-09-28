"""Transposing a rank-0 var is the identity, on every device.

`TransposeOp` refused it -- `infer_shape`'s `USER_CHECK(xdim)` -- so every
scalar-shaped `einops.rearrange` died with
`transpose_op.cc:61: [check failed: xdim]`. Both references call it the
identity: NumPy's `transpose` returns shape `()`, and torch's `.T` warns
"This function is the identity in these cases". Torch still rejects a
*non-empty* permutation of a rank-0 tensor, and so does this.
"""

import numpy as np
import jittor as jt

def test_rank0_transpose_is_identity():
    for use_cuda in (0, 1) if jt.has_cuda else (0,):
        with jt.flag_scope(use_cuda=use_cuda):
            x = jt.array(np.float32(3.5))
            assert x.ndim == 0, x.shape
            y = x.transpose()
            assert y.ndim == 0, (use_cuda, y.shape)
            assert float(y.item()) == 3.5
            z = jt.transpose(x, ())
            assert z.ndim == 0 and float(z.item()) == 3.5
            # 非空排列仍应报错
            try:
                jt.transpose(x, (0,))
                raise AssertionError("rank-0 加非空排列应当报错")
            except RuntimeError:
                pass
