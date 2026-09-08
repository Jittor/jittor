"""Views share storage with the tensor they were sliced from.

Two claims, both of which jittor answered with a silently wrong result before
5.02:

* an in-place write through a view reaches the base (``v = y[1:4]``;
  ``v.assign(v + 100)`` must change ``y``).  Jittor's ``Var`` is a graph node
  and ``VarHolder`` is the name bound to it, so "in place" means the base's
  *name* is rebound to a ``setitem`` over the slice -- there is no eager buffer
  mutation anywhere in this framework.  What matters to a caller is that the
  base observes the write, and that it observes it at any view depth and for
  any basic-index expression, not only for the chains of single integers that
  ``check_cascade_setitem`` used to infer from the op graph.
* an expanded tensor is a stride-0 view and does not allocate its logical
  footprint, including when it is explicitly synchronized before consumption.
"""

import numpy as np
import pytest

import jittor as jt


def test_slice_view_assign_writes_through_to_base():
    y = jt.array(np.arange(10, dtype="float32"))
    v = y[1:4]
    v.assign(v + 100)
    np.testing.assert_array_equal(
        y.numpy(), [0, 101, 102, 103, 4, 5, 6, 7, 8, 9])
    np.testing.assert_array_equal(v.numpy(), [101, 102, 103])


def test_row_view_assign_writes_through_to_base():
    y = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    row = y[1]
    row.assign(row * 0)
    expected = np.arange(12, dtype="float32").reshape(3, 4)
    expected[1] = 0
    np.testing.assert_array_equal(y.numpy(), expected)


def test_write_through_survives_two_view_levels():
    y = jt.array(np.arange(24, dtype="float32").reshape(2, 3, 4))
    inner = y[1][2]
    inner.assign(inner + 1000)
    expected = np.arange(24, dtype="float32").reshape(2, 3, 4)
    expected[1, 2] += 1000
    np.testing.assert_array_equal(y.numpy(), expected)


def test_write_through_is_not_limited_to_integer_indices():
    # The op-graph walk `check_cascade_setitem` replaced only chains whose every
    # step was a single integer, so a slice anywhere in the chain silently
    # dropped the write.
    y = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    block = y[1:3]
    block.assign(block * 0 - 1)
    expected = np.arange(12, dtype="float32").reshape(3, 4)
    expected[1:3] = -1
    np.testing.assert_array_equal(y.numpy(), expected)


def test_setitem_on_a_view_reaches_the_base():
    y = jt.array(np.arange(12, dtype="float32").reshape(3, 4))
    row = y[1]
    row[2] = 90.0
    expected = np.arange(12, dtype="float32").reshape(3, 4)
    expected[1, 2] = 90
    np.testing.assert_array_equal(y.numpy(), expected)


def test_advanced_indexing_is_a_copy_not_a_view():
    # torch parity: only basic indexing produces a view.  A gather by index Var
    # must stay a copy, or optimizer state chains across every generation of a
    # densifying model.
    y = jt.array(np.arange(10, dtype="float32"))
    picked = y[jt.array(np.array([1, 3, 5]))]
    picked.assign(picked + 100)
    np.testing.assert_array_equal(y.numpy(), np.arange(10, dtype="float32"))


def test_assign_on_a_non_view_still_rebinds():
    y = jt.array(np.arange(4, dtype="float32"))
    z = y + 0
    z.assign(z + 1)
    np.testing.assert_array_equal(y.numpy(), np.arange(4, dtype="float32"))
    np.testing.assert_array_equal(z.numpy(), np.arange(4, dtype="float32") + 1)


def test_a_view_of_a_dead_base_still_assigns_locally():
    # The view holds the base weakly on purpose: keeping it alive would change
    # every lived-Var count in the tree and buy nothing observable, because a
    # base nobody names can only be read back through this view.
    y = jt.array(np.arange(10, dtype="float32"))
    v = y[1:4]
    del y
    v.assign(v + 100)
    np.testing.assert_array_equal(v.numpy(), [101, 102, 103])


def test_expand_does_not_materialize_when_it_is_consumed():
    a = jt.ones((4096, 1))
    a.sync()
    with jt.flag_scope(use_stat_allocator=1):
        before = jt.flags.stat_allocator_total_alloc_byte
        total = (a.expand(4096, 4096) * 2).sum()
        total.sync()
        grew = jt.flags.stat_allocator_total_alloc_byte - before
    assert float(total.numpy()) == 4096 * 4096 * 2
    # 4096*4096*4 bytes is 64 MB; a fused consumer must not pay it.
    assert grew < 4096 * 4096, grew


def test_expand_still_materializes():
    a = jt.ones((4096, 1))
    a.sync()
    b = a.expand(4096, 4096)
    b.sync()
    # Read the two pointers out and drop the Vars before asserting: an xfailing
    # assertion keeps its frame alive inside the stored exception, and this file
    # would then report two leaked hold vars at every run.
    base, expanded = a.raw_ptr, b.raw_ptr
    del a, b
    assert expanded == base
