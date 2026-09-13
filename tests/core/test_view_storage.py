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


def test_update_writes_direct_and_chained_views_through_to_root():
    y = jt.zeros((2, 3, 4))
    direct = y[0]
    sibling = y[1]
    chained = y[1][2]

    direct._update(jt.ones(direct.shape) * 3)
    chained._update(jt.ones(chained.shape) * 7)

    expected = np.zeros((2, 3, 4), dtype="float32")
    expected[0] = 3
    expected[1, 2] = 7
    np.testing.assert_array_equal(y.numpy(), expected)
    np.testing.assert_array_equal(sibling.numpy(), expected[1])


def test_view_update_preserves_rhs_gradient_and_nonview_update_contract():
    root = jt.zeros((2, 3))
    rhs = jt.array(np.array([2.0, 3.0, 4.0], dtype="float32"))
    rhs.start_grad()
    root[1]._update(rhs * 2)
    np.testing.assert_array_equal(jt.grad(root.sum(), rhs).numpy(), [2, 2, 2])

    ordinary = jt.zeros((3,))
    identity = id(ordinary)
    ordinary._update(rhs + 5)
    assert id(ordinary) == identity
    np.testing.assert_array_equal(ordinary.numpy(), [7, 8, 9])

    same = jt.array(np.array([4.0, 5.0, 6.0], dtype="float32"))
    same_identity = id(same)
    same._update(same)
    assert id(same) == same_identity
    np.testing.assert_array_equal(same.numpy(), [4, 5, 6])

    # A value that previously served as an ordinary update destination must
    # remain usable as the RHS of a later view update.
    staged = jt.zeros((3,))
    staged._update(rhs + 1)
    target = jt.zeros((2, 3))
    target[0]._update(staged)
    np.testing.assert_array_equal(target.numpy()[0], [3, 4, 5])


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
        before = jt.introspection.counters.allocator.allocated_bytes
        total = (a.expand(4096, 4096) * 2).sum()
        total.sync()
        grew = jt.introspection.counters.allocator.allocated_bytes - before
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
