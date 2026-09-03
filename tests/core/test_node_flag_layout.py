# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What the Var/Op flag split has to keep true at run time.

``Node::flags`` is one word whose upper bits mean different things on a ``Var``
and on an ``Op`` (see ``NodeFlags`` in ``node.h`` and
``tests/structure/test_node_flag_kinds.py`` for the layout rules).  The union is
deliberate; what was not deliberate is code that read a bit off a node without
knowing which kind it had.

``executor.cc`` did exactly that.  ``run_sync``'s bfs queue holds vars and ops
together, and for every node it reached it did::

    need_opt += node->flags.get(NodeFlags::_has_gopt);
    ...
    for (Node* n : bfs_q)
        if (n->flags.get(NodeFlags::_has_gopt)) {
            n->op()->graph_optimize();      // on whatever answered yes

``_has_gopt`` is an Op flag.  Asking a var for it was only harmless because the
bit number it happened to occupy was unused in the Var layout -- an accident of
two hand-maintained lists of numbers, not a property anybody stated.  Once the
two layouts are packed independently, ``_has_gopt`` and the Var-only
``_first_order_only`` land on the same bit, and a var carrying that flag would
be handed to a virtual ``Op`` call.

That is what the first case here builds: a first-order-only var and a
graph-optimizable op (``setitem``) in one ``run_sync``.

It runs *only* in a child process, and that is not a preference.  Restoring the
kind-blind read and running the same graph in-process segfaults -- "fault pc
(no mapping)", a virtual call through a var's vtable slot -- and jittor's
process-level SIGCHLD action then takes the whole pytest session down with no
output at all (6.C31).  The child asserts the values as well as the survival so
there is nothing left for an in-process copy to add.

Run::  python -m pytest tests/core/test_node_flag_layout.py
"""

import unittest

import numpy as np
import jittor as jt

from _helpers.child_process import run_child_script


MIXED_GRAPH = r'''
import numpy as np
import jittor as jt

# A var with a Var-only flag whose bit number an Op-only flag also uses.
first_order_only = jt.array(np.arange(8, dtype="float32"))
first_order_only._set_first_order_only()

# ... and, in the same lazy graph, an op that really does want graph_optimize.
target = jt.array(np.arange(8, dtype="float32")) + 1
target[2:5] = 7.0

total = (first_order_only + target).sum()
print("RESULT", total.item(), list(target.numpy()))
'''


class TestNodeFlagLayoutAtRuntime(unittest.TestCase):
    def test_a_first_order_only_var_is_not_taken_for_an_optimizable_op(self):
        done = run_child_script(MIXED_GRAPH, text=True, merge_stderr=True,
                                name="node_flag_layout", crash_isolated=True)
        output = done.stdout
        self.assertEqual(done.returncode, 0,
                         "run_sync handed a var to Op::graph_optimize\n"
                         + output[-4000:])
        # x = 0..7 sums to 28; target is [1,2,7,7,7,6,7,8] and sums to 45.
        self.assertIn("RESULT 73.0", output, output[-4000:])
        self.assertIn("7.0, 7.0, 7.0", output, output[-4000:])

    def test_requires_grad_does_not_make_a_var_a_scalar(self):
        """The bug that started this: two meanings on one bit.

        ``_explicit_requires_grad`` and ``_is_scalar`` shared a bit for a while,
        so a ``requires_grad`` parameter looked like a Python scalar to
        ``binary_dtype_infer``: it dropped out of dtype promotion (float64 ->
        float32) and ``float_dtype``'s has_scalar branch skipped the amp
        override on every operator that touched it.  They are in the same enum
        now, so they cannot collide again by construction; this pins the
        behaviour the collision broke.
        """
        parameter = jt.array(np.ones(4), dtype="float64")
        other = jt.array(np.ones(4, "float32"))
        self.assertEqual(str((parameter * other).dtype), "float64")
        parameter.requires_grad = True
        self.assertEqual(str((parameter * other).dtype), "float64",
                         "requires_grad made the var read back as a scalar")
        self.assertTrue(parameter.requires_grad)


if __name__ == "__main__":
    unittest.main()
