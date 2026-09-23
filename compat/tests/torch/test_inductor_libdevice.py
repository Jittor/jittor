"""Inductor's libdevice import delegates to executable Triton device math."""

import importlib

import numpy as np
import pytest


def test_inductor_libdevice_uses_upstream_math_identity():
    helpers = importlib.import_module("torch._inductor.runtime.triton_helpers")
    pytest.importorskip("triton.language.extra", reason="requires upstream Triton device math")
    from triton.language.extra import libdevice

    assert helpers.libdevice is libdevice
    with pytest.raises(AttributeError):
        getattr(helpers, "nonexistent_kernel_helper")


@pytest.mark.cuda
def test_inductor_libdevice_counts_nans_on_cuda():
    from torch._inductor.runtime.triton_helpers import libdevice
    import jittor as jt
    import triton
    import triton.language as tl

    assert jt.flags.use_cuda == 1
    assert getattr(triton.runtime.jit.JITFunction, "_jittor_bridge", False)

    @triton.jit
    def count_nans(values, counts, n_columns: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        columns = tl.arange(0, BLOCK)
        values_block = tl.load(values + row * n_columns + columns,
                               mask=columns < n_columns, other=0)
        count = tl.sum(libdevice.isnan(values_block).to(tl.int32), axis=0)
        tl.store(counts + row, count)

    inputs = np.array([[np.nan, 1., np.inf, -np.inf, np.nan],
                       [0., -1., np.inf, 2., 3.],
                       [np.nan, np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    values = jt.array(inputs)
    counts = jt.zeros((3,), dtype="int32")
    count_nans[(3,)](values, counts, n_columns=5, BLOCK=8)
    np.testing.assert_array_equal(counts.numpy(), np.isnan(inputs).sum(axis=1))
