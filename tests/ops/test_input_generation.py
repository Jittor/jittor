"""Contracts for ``make_tensor``: the data a test gets must belong to that test.

The default seed used to come from a process-level counter
(``itertools.count(0x5EED)``), so which numbers a case received depended on how
many ``make_tensor`` calls had happened earlier in the same process. Three things
followed, and all three showed up in practice:

* a case that failed in a full run got *different* data when re-run with ``-k``,
  so the failure could not be reproduced;
* inserting or deleting any test shifted the inputs of every test after it, which
  turns an unrelated edit into a data change across the whole suite;
* running under xdist or in random order was not possible at all.

The seed is now a pure function of the test's own nodeid, the draw ordinal within
that test, and the requested shape/dtype/range.
"""

import numpy as np

from _helpers import common as cu
from _helpers.common import make_tensor, to_numpy


def _draw(test_id, *shape, **kw):
    cu.begin_test_inputs(test_id)
    return to_numpy(make_tensor(*shape, **kw))


def test_inputs_depend_only_on_the_test_that_asked_for_them():
    """-k and a full run must hand the same case the same numbers."""
    alone = _draw("tests/ops/test_x.py::TestX::test_a", 3, 4, dtype=cu.float32)

    # Simulate the rest of a full run happening first: a different test drawing a
    # different number of tensors of different shapes.
    cu.begin_test_inputs("tests/ops/test_x.py::TestX::test_earlier")
    make_tensor(7, dtype=cu.float64)
    make_tensor(2, 2, dtype=cu.int32)
    make_tensor(5, 5, 5, dtype=cu.float32)

    in_full_run = _draw("tests/ops/test_x.py::TestX::test_a", 3, 4, dtype=cu.float32)
    np.testing.assert_array_equal(alone, in_full_run)


def test_two_different_tests_get_different_data():
    left = _draw("tests/ops/test_x.py::TestX::test_a", 4, 4, dtype=cu.float32)
    right = _draw("tests/ops/test_x.py::TestX::test_b", 4, 4, dtype=cu.float32)
    assert not np.array_equal(left, right)


def test_successive_draws_inside_one_test_differ():
    """Otherwise both operands of a binary op would be the same tensor."""
    cu.begin_test_inputs("tests/ops/test_x.py::TestX::test_a")
    first = to_numpy(make_tensor(4, 4, dtype=cu.float32))
    second = to_numpy(make_tensor(4, 4, dtype=cu.float32))
    assert not np.array_equal(first, second)


def test_the_seed_does_not_move_between_processes():
    """``hash()`` on a str is salted per process; the seed must not be."""
    assert cu.stable_seed("a", 1, (2, 3)) == cu.stable_seed("a", 1, (2, 3))
    assert cu.stable_seed("a", 1, (2, 3)) != cu.stable_seed("a", 2, (2, 3))
    # Pinned literal: if this changes, every generated input in the suite changed,
    # which is a decision, not a refactor.
    assert cu.stable_seed("pin", 0, (2, 2), "float32") == 1629849870


def test_a_failing_test_can_report_the_seeds_it_drew():
    cu.begin_test_inputs("tests/ops/test_x.py::TestX::test_a")
    make_tensor(3, dtype=cu.float32)
    make_tensor(2, 2, dtype=cu.int64)
    drawn = cu.drawn_inputs()
    assert len(drawn) == 2
    for entry in drawn:
        assert "seed=" in entry and "shape=" in entry and "dtype=" in entry


def test_an_explicit_seed_still_wins():
    cu.begin_test_inputs("tests/ops/test_x.py::TestX::test_a")
    pinned = to_numpy(make_tensor(3, 3, dtype=cu.float32, seed=1234))
    cu.begin_test_inputs("tests/ops/test_x.py::TestX::test_b")
    again = to_numpy(make_tensor(3, 3, dtype=cu.float32, seed=1234))
    np.testing.assert_array_equal(pinned, again)


def test_bfloat16_inputs_keep_the_declared_dtype():
    cu.begin_test_inputs("tests/ops/test_input_generation.py::test_bfloat16")
    value = make_tensor(4, dtype=cu.bfloat16)
    assert str(value.dtype) == cu.bfloat16
