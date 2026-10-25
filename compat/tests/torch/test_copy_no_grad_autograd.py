# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Dun Liang <randonlang@gmail.com>.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``copy_`` under ``no_grad`` leaves the destination a trainable leaf.

KI-COMPAT-007: a leaf ``w`` with ``requires_grad=True`` that is filled by
``w.copy_(x)`` inside ``torch.no_grad()`` stayed a leaf in name only -- a later
backward through ``w`` left ``w.grad`` as ``None`` when ``x`` required grad,
while an ``x`` that did not require grad worked.  The failing shape is a source
that required grad but was not a leaf (``x = a * 2``): the old in-place path
rebound ``w`` onto the source's graph, so gradients reached ``a`` and ``w`` was
no longer a leaf.  PyTorch accumulates into ``w.grad`` in both cases, because
``copy_`` writes values into the leaf's own storage and does not rebind it to
the source.  The tests pin the value copy, the leaf identity, the gradient, and
the absence of storage aliasing.

Run::  JITTOR_TORCH_SHIM=1 python -m pytest -q compat/tests/torch/test_copy_no_grad_autograd.py
"""

import numpy as np
import pytest
import torch

from _helpers.device_types import instantiate_device_type_tests


def _copy_under_no_grad(device, source_requires_grad):
    # Device-type instantiation only carries ``test*`` members to the generated
    # class, so this helper stays at module level.
    w = torch.randn(4, 8, device=device, requires_grad=True)
    x = torch.randn(4, 8, device=device, requires_grad=source_requires_grad)
    expected = x.detach().cpu().numpy().copy()
    with torch.no_grad():
        returned = w.copy_(x)
    assert returned is w
    assert w.is_leaf and w.requires_grad
    assert w.grad is None
    np.testing.assert_array_equal(w.detach().cpu().numpy(), expected)
    (w * 2.0).sum().backward()
    assert w.grad is not None, "copy_ under no_grad dropped the leaf's gradient"
    np.testing.assert_allclose(w.grad.cpu().numpy(), np.full((4, 8), 2.0))


class TestCopyNoGradAutograd:
    def test_grad_enabled_source(self, device):
        _copy_under_no_grad(device, True)

    def test_grad_disabled_source(self, device):
        _copy_under_no_grad(device, False)

    def test_nonleaf_grad_source_does_not_rebind_the_leaf(self, device):
        # The original KI-COMPAT-007 shape: ``x`` requires grad but is not a
        # leaf.  Rebinding ``w`` onto x's graph made ``w`` a non-leaf, so the
        # backward consumed ``w`` as an intermediate and left ``w.grad`` None
        # while sending the gradient to x's source.
        source = torch.randn(4, 8, device=device, requires_grad=True)
        w = torch.randn(4, 8, device=device, requires_grad=True)
        x = source * 2.0
        with torch.no_grad():
            w.copy_(x)
        assert w.is_leaf and w.requires_grad
        (w * 2.0).sum().backward()
        assert w.grad is not None, "copy_ rebound the leaf onto the source graph"
        np.testing.assert_allclose(w.grad.cpu().numpy(), np.full((4, 8), 2.0))
        assert source.grad is None, "the no_grad copy must not propagate to the source"

    def test_gradient_accumulates_across_backwards(self, device):
        w = torch.randn(4, 8, device=device, requires_grad=True)
        x = torch.randn(4, 8, device=device, requires_grad=True)
        with torch.no_grad():
            w.copy_(x)
        (w * 2.0).sum().backward()
        (w * 3.0).sum().backward()
        np.testing.assert_allclose(w.grad.cpu().numpy(), np.full((4, 8), 5.0))

    def test_copy_does_not_alias_the_source(self, device):
        # The value is written into w's storage; rebinding w to x's node is what
        # made the gradient disappear.  Mutating x afterwards must not be seen.
        w = torch.randn(4, 8, device=device, requires_grad=True)
        x = torch.ones(4, 8, device=device)
        with torch.no_grad():
            w.copy_(x)
            x.add_(5.0)
        np.testing.assert_array_equal(w.detach().cpu().numpy(), np.ones((4, 8)))

    def test_inplace_copy_on_leaf_outside_no_grad_still_raises(self, device):
        w = torch.randn(4, 8, device=device, requires_grad=True)
        x = torch.randn(4, 8, device=device)
        with pytest.raises(RuntimeError, match="leaf"):
            w.copy_(x)


instantiate_device_type_tests(TestCopyNoGradAutograd, globals(), only_for=("cpu", "cuda"))
