# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""torch.real/torch.imag must recognize a BatchedVar (jittor-core-gaps.md §3.4).

``compat/torch/installers/numerical/complex.py``'s ``real``/``imag`` aliases
only checked ``isinstance(input, (jt.nn.ComplexNumber, jt.Var))`` -- a native
``jt.vmap`` ``BatchedVar`` is neither, so ``torch.real(bv)`` silently returned
``bv`` itself unchanged (mislabeled as "the real part") and ``torch.imag(bv)``
silently returned an all-zero result even for a genuinely complex ``bv``: a
silent wrong-answer bug, not the fail-fast this project otherwise requires.
``BatchedVar.real``/``.imag`` already delegate correctly (python/jittor/
vmap.py's ``_apply_real``/``_apply_imag``); this was purely a recognition gap
in the compat alias.
"""
import numpy as np
import torch


def test_real_imag_recognize_batched_var():
    import jittor as jt

    x = jt.array((np.random.RandomState(0).randn(4, 3)
                  + 1j * np.random.RandomState(1).randn(4, 3)).astype("complex64"))

    real_out = jt.vmap(lambda v: torch.real(v))(x)
    imag_out = jt.vmap(lambda v: torch.imag(v))(x)

    np.testing.assert_allclose(real_out.numpy(), x.numpy().real, atol=1e-5)
    np.testing.assert_allclose(imag_out.numpy(), x.numpy().imag, atol=1e-5)
    # the pre-fix bug: imag() on a BatchedVar returned all zeros regardless
    # of the actual imaginary part.
    assert np.max(np.abs(imag_out.numpy())) > 0.0
