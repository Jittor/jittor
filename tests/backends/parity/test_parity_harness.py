# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The device-parity harness must be able to compare *every* output shape.

``test_device_parity`` projects each floating output onto a fixed random
cotangent to get one scalar loss per sample, then differentiates it. The
projection was drawn with ``RandomState(seed).randn(*output.shape)``, and for a
**0-d output** that is ``randn()`` with no arguments -- which returns a python
float, not an array, so the next call (``.astype("float32")``) raised
``AttributeError``.

Every operator whose output is a scalar therefore never got as far as its
backward comparison: ``sum``, ``trace``, and every loss (``cross_entropy``,
``nll_loss``, ``kl_div``, ``mse_loss``, ``l1_loss``, ...). The failure was loud
rather than silent -- but only on a machine with an accelerator, and the whole
battery is ``skipUnless`` an accelerator, so the reports that mattered read
"skipped". 114 consecutive hand-offs from one partition recorded "this machine
has no CUDA" for a machine with eight usable GPUs; nothing was checking, so this
class of operator has had no CPU-vs-accelerator backward evidence at all.

These cases run on the CPU alone, so they hold on a CPU-only box too: the
harness is checked where it can be checked, rather than only where it is used.

Run::  JITTOR_TORCH_SHIM=1 python -m pytest tests/backends/parity/test_parity_harness.py
"""
import numpy as np

from opinfo.database import op_db

from backends.parity import test_device_parity as parity


#: Operators from ``op_db`` whose forward collapses to a single number.
#: Everything here reaches ``_run``'s cotangent draw with an empty shape.
SCALAR_OUTPUT_OPERATORS = ("sum", "trace", "mse_loss", "l1_loss")


def test_scalar_output_operators_reach_their_backward():
    """The regression: a 0-d forward must still produce gradients.

    Before the fix this raised ``AttributeError: 'float' object has no
    attribute 'astype'`` inside the harness, so these operators' gradients were
    never computed on either device and the comparison that is the point of the
    battery never happened.

    One case over all four operators rather than a parametrized one per
    operator: ``tests/structure/test_pytest_contract.py`` reserves module-level
    ``test_*`` parameters for fixtures.
    """
    checked = []
    for name in SCALAR_OUTPUT_OPERATORS:
        op = next((candidate for candidate in op_db
                   if candidate.full_name == name), None)
        if op is None or not op.supports_autograd:
            continue
        found = False
        for sample in op.sample_inputs("cpu", "float32", requires_grad=True):
            forward, grads = parity._run(op, sample, use_cuda=0)
            if not forward or np.asarray(forward[0]).shape != ():
                continue
            found = True
            assert grads is not None, (
                "%s has a 0-d forward and produced no gradients; the harness "
                "stopped before the backward comparison" % name)
            assert grads, "%s produced an empty gradient list" % name
            for index, grad in enumerate(grads):
                assert np.all(np.isfinite(grad)), \
                    "%s gradient %d is not finite" % (name, index)
        assert found, ("%s produced no 0-d sample, so this case would check "
                       "nothing" % name)
        checked.append(name)
    assert checked, "no operator from SCALAR_OUTPUT_OPERATORS is in op_db"


def test_the_cotangent_projection_is_unchanged_for_non_scalar_outputs():
    """The fix must not renumber the projection for the shapes that worked.

    ``randn(*shape)`` and ``standard_normal(shape)`` are the same draw from the
    same seeded generator, so every non-scalar output keeps the exact cotangent
    it had. If that were not so, every cached or previously recorded gradient
    would move and the change would look like a numerical regression.
    """
    for shape in ((3,), (2, 3), (2, 3, 4)):
        legacy = np.random.RandomState(1234).randn(*shape).astype("float32")
        current = parity._cotangent(shape, 0)
        assert current.dtype == np.dtype("float32")
        assert current.shape == shape
        assert np.array_equal(current, legacy)


def test_the_cotangent_is_an_array_for_a_scalar_output():
    cotangent = parity._cotangent((), 0)
    assert isinstance(cotangent, np.ndarray)
    assert cotangent.shape == ()
    assert cotangent.dtype == np.dtype("float32")
    assert np.isfinite(cotangent)


def test_the_cotangent_differs_per_output_index():
    """Two outputs must not be projected onto the same direction.

    A shared projection would cancel a bug that swaps two outputs.
    """
    first = parity._cotangent((4,), 0)
    second = parity._cotangent((4,), 1)
    assert not np.array_equal(first, second)
