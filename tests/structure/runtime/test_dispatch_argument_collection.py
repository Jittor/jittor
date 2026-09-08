# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What `dispatch_context` finds in a kernel's arguments, and in what order.

Task 3.21 rewrote the walk: it used to define a closure per call and recurse
once per *value*, which cost a measured 2.1 of its 3.2 us and dominated the
Python side of building a graph.  The walk now costs one call per container.

The behaviour it has to keep was, before this file, entirely unpinned -- the
cyclic-container branch had no test at all, so a rewrite that dropped the
guard, or that stopped looking inside nested containers, or that returned the
tensors in a different order, would have been silently accepted.  Order is
load-bearing: `core.dispatch_context` resolves the device from the first input
that has one, and `DispatchContext.dtypes` is matched positionally against a
kernel's declared dtypes.
"""
import unittest

import jittor as jt
from jittor._runtime.dispatch import dispatch_context


class TestArgumentCollection(unittest.TestCase):
    def _dtypes(self, *args, **kwargs):
        return dispatch_context(*args, **kwargs).dtypes

    def test_flat_arguments_are_collected_in_order_and_scalars_ignored(self):
        first = jt.zeros((2,), "float32")
        second = jt.zeros((2,), "float64")
        self.assertEqual(
            self._dtypes(first, 3, second, "text", 1e-5, None),
            ("float32", "float64"))

    def test_positional_arguments_come_before_keyword_arguments(self):
        positional = jt.zeros((2,), "float32")
        keyword = jt.zeros((2,), "int32")
        self.assertEqual(self._dtypes(positional, weight=keyword),
                         ("float32", "int32"))

    def test_tensors_inside_lists_tuples_and_dicts_are_found(self):
        a = jt.zeros((2,), "float32")
        b = jt.zeros((2,), "float64")
        c = jt.zeros((2,), "int32")
        d = jt.zeros((2,), "int64")
        self.assertEqual(self._dtypes([a, (b,)], {"k": c}, extra={"j": [d]}),
                         ("float32", "float64", "int32", "int64"))

    def test_nesting_is_followed_to_the_bottom(self):
        deep = jt.zeros((2,), "float64")
        nest = [[[[[deep]]]]]
        self.assertEqual(self._dtypes(nest), ("float64",))

    def test_the_same_container_twice_side_by_side_is_not_a_cycle(self):
        """It is reachable twice, not reachable from itself. The walk has to
        take its entry back out on the way up, and this is what says so."""
        shared = [jt.zeros((2,), "float32")]
        self.assertEqual(self._dtypes(shared, shared), ("float32", "float32"))
        self.assertEqual(self._dtypes([shared, shared]), ("float32", "float32"))

    def test_a_container_reachable_from_itself_is_reported(self):
        cycle = [jt.zeros((2,), "float32")]
        cycle.append(cycle)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            dispatch_context(cycle)

    def test_a_cycle_through_a_dict_is_reported(self):
        cycle = {"tensor": jt.zeros((2,), "float32")}
        cycle["self"] = cycle
        with self.assertRaisesRegex(ValueError, "cyclic"):
            dispatch_context(cycle)

    def test_a_cycle_two_containers_long_is_reported(self):
        outer = []
        inner = [outer]
        outer.append(inner)
        with self.assertRaisesRegex(ValueError, "cyclic"):
            dispatch_context(outer)

    def test_a_reported_cycle_leaves_no_state_behind(self):
        """The walk's cycle set is per call; a raised cycle must not make the
        next call see containers that are no longer being visited."""
        cycle = []
        cycle.append(cycle)
        for _ in range(3):
            with self.assertRaisesRegex(ValueError, "cyclic"):
                dispatch_context(cycle)
        shared = [jt.zeros((2,), "float32")]
        self.assertEqual(self._dtypes(shared, shared), ("float32", "float32"))

    def test_no_arguments_still_resolves_a_backend(self):
        context = dispatch_context()
        self.assertEqual(context.dtypes, ())
        self.assertTrue(context.backend)

    def test_a_backend_and_device_come_from_the_native_query(self):
        tensor = jt.zeros((2,), "float32")
        context = dispatch_context(tensor)
        self.assertEqual((context.backend, context.device_id),
                         tuple(jt.core.dispatch_context([tensor])))


if __name__ == "__main__":
    unittest.main()
