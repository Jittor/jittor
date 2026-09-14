# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Every generated op maker that takes a ``VarSlices`` must adapt its indices.

``python/jittor/build/codegen.py`` emits one ``make_<op>`` per registered
operator and inserts the storage adaptation calls into it: ``Var*`` inputs go
through ``adapt_storage_input`` and ``VarSlices`` indices through
``adapt_index_storage``. The second one was written, committed, and then
emitted **nowhere**: the argument list is split off the C++ declaration, so
every argument after the first still carries the space that followed the comma,
and ``" VarSlices&& slices".startswith("VarSlices")`` is false. The call had no
callers for as long as it existed, and a broadcast index Var reached the
``getitem``/``setitem`` kernels, which read index Vars as if they were dense
(KI-OPS-009).

The behavioural gate is ``tests/ops/test_broadcast_index.py``. This file is the
label: it reads the generated source, so a maker that loses the call again says
which maker and not merely that indexing went wrong. It is also the half that
covers a *future* ``VarSlices`` op, which the behavioural test cannot reach.
"""

import glob
import os
import re
import unittest

import jittor as jt


def _generated_op_maker():
    matches = glob.glob(os.path.join(jt.compiler.cache_path, "**", "jit_op_maker.h"),
                        recursive=True)
    if not matches:
        raise unittest.SkipTest("no generated jit_op_maker.h under %s"
                                % jt.compiler.cache_path)
    with open(max(matches, key=os.path.getmtime), encoding="utf-8") as source:
        return source.read()


#: `VarPtr make_foo(...) {` up to the `auto _op = new` that ends the preamble.
_MAKER = re.compile(
    r"^\s*(?:VarPtr|vector<VarPtr>)\s+(make_\w+)\(([^)]*)\)\s*\{(.*?)auto _op = new",
    re.S | re.M)


class TestIndexStorageAdaptation(unittest.TestCase):

    def setUp(self):
        self.source = _generated_op_maker()
        self.makers = _MAKER.findall(self.source)
        self.assertTrue(self.makers, "no op makers were parsed out of the "
                                     "generated source; this test would pass "
                                     "without checking anything")

    def test_var_slices_makers_adapt_their_indices(self):
        seen = 0
        for name, args, body in self.makers:
            if "VarSlices" not in args:
                continue
            seen += 1
            argument = [a for a in args.split(",") if "VarSlices" in a][0]
            argument_name = argument.split()[-1].split("=")[0]
            with self.subTest(maker=name):
                self.assertIn(
                    "adapt_index_storage(%s," % argument_name, body,
                    "%s takes a VarSlices and never calls adapt_index_storage, "
                    "so a non-contiguous index Var (a broadcast, a strided "
                    "view) reaches its kernel, which reads index Vars as if "
                    "they were dense (KI-OPS-009)." % name)
        self.assertGreaterEqual(
            seen, 2, "expected at least the getitem and setitem makers to take "
                     "a VarSlices; found %d, so this test is no longer looking "
                     "at the ops it was written for" % seen)

    def test_var_inputs_still_adapt_their_storage(self):
        """The same whitespace bug would silence this one; it never did, and
        the fix must not make it start."""
        for name, args, body in self.makers:
            if name != "make_binary":
                continue
            for operand in ("x", "y"):
                with self.subTest(operand=operand):
                    self.assertIn("adapt_storage_input<BinaryOp>(%s," % operand, body)
            return
        self.fail("make_binary was not generated")


if __name__ == "__main__":
    unittest.main()
