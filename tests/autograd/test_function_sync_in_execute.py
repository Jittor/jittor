# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers: Jittor Group
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
# A `jt.Function` whose `execute` syncs keeps its second derivative.
#
# `grad` computes from the inputs `execute` saved, and those are taped copies
# that `_rebind_saved_inputs` swaps back onto the caller's Vars so the result
# differentiates again. Syncing inside `execute` -- the input, or an output,
# which materializes the input it depends on -- runs the tape first, and a
# tape that has run can no longer name its input. Such a saved input was
# skipped: the second derivative was 0, or a missing-path error in strict
# mode, while the forward and first derivative stayed right.
import numpy as np
import pytest
import jittor as jt

from _helpers import capability


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if (
        request.param == "cuda"
        and not capability.check_accelerator("cuda", backend=jt).enabled
    ):
        pytest.skip("CUDA is unavailable in this build")
    with jt.flag_scope(use_cuda=int(request.param == "cuda")):
        yield request.param


def _cube(saved_kind, sync):
    class Cube(jt.Function):
        def execute(self, x):
            if saved_kind == "direct":
                self.saved = x
            elif saved_kind == "nested":
                self.saved = {"outer": [(x,)]}
            else:  # shared and cyclic: one Var in several places, a dict holding itself
                self.saved = {"outer": [(x,)], "aliases": [x, x]}
                self.saved["self"] = self.saved
            result = x * x * x
            if sync == "input":
                jt.sync([x], device_sync=True)
            elif sync == "output":
                jt.sync([result], device_sync=True)
            elif sync == "both":
                jt.sync([x, result], device_sync=True)
            return result

        def grad(self, g):
            if saved_kind == "direct":
                x = self.saved
            else:
                x = self.saved["outer"][0][0]
                if saved_kind == "shared_cycle":
                    assert self.saved["self"] is self.saved
                    assert self.saved["aliases"][0] is x is self.saved["aliases"][1]
            return (3 * x * x).conj() * g
    return Cube


@pytest.mark.parametrize("sync", ["none", "input", "output", "both"])
@pytest.mark.parametrize("saved_kind", ["direct", "nested", "shared_cycle"])
@pytest.mark.parametrize("dtype", ["float32", "complex64"])
def test_second_derivative_after_a_sync_in_execute(device, dtype, saved_kind, sync):
    raw = np.array([2 + 1j, -1.5 + 0.5j] if dtype == "complex64" else [2, -1.5], dtype=dtype)
    x = jt.array(raw, dtype=dtype)
    with jt.flag_scope(missing_grad_error=1):
        y = _cube(saved_kind, sync).apply(x)
        first = jt.grad(y.real.sum(), x)
        second = jt.grad(first.real.sum(), x)
    for got, expected in zip((y, first, second),
                             (raw ** 3, np.conj(3 * raw ** 2), 6 * np.conj(raw))):
        np.testing.assert_allclose(got.numpy(), expected, rtol=2e-6, atol=2e-6)


def test_the_callers_var_is_untouched(device):
    # The saved input is moved onto a new holder of the caller's Var; the
    # caller's own holder keeps its Var and value.
    x = jt.array([2.0, -1.5])
    before = x.id
    first = jt.grad(_cube("nested", "both").apply(x).sum(), x)
    jt.grad(first.sum(), x).sync()
    assert x.id == before
    np.testing.assert_allclose(x.numpy(), [2.0, -1.5])
