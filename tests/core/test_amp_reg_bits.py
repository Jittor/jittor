# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``jt.flags.amp_reg``: what its bits mean, and who is allowed to change them.

Task 5.11. ``amp_reg`` is a bit field that decides the dtype of nearly every op.
Its bits were spelled as bare integers in six files (``amp_reg & 16``,
``amp_reg | 36``, ``amp_reg = 4``) with the meaning written down only inside the
flag's description string in ``src/var.cc``. Two things had gone wrong under
cover of that:

* ``jt.concat`` entered ``flag_scope(amp_reg=4)`` -- an ASSIGNMENT. For the
  whole call it replaced the caller's AMP policy with "keep_reduce and nothing
  else", so under ``auto_mixed_precision_level=6`` every concat quietly dropped
  three of the four bits the user had set and returned float32 in the middle of
  a float16 graph. concat contains no reduce at all, so the bit it forced was
  not even its own.

* ``array()`` and ``random()`` each carried a copy of the same "array-like
  preference" logic, and the copies had drifted: ``array()`` skipped
  one-element and non-float results, ``random()`` did not. ``jt.array([1.0])``
  came back float32 and ``jt.random((1,))`` float16, in the same scope.

Every expectation below was measured on the tree before and after the fix.

Run::  python -m pytest tests/core/test_amp_reg_bits.py
"""

import re
import unittest
from pathlib import Path

import numpy as np

import jittor as jt


class amp_level:
    """Set ``auto_mixed_precision_level`` and put it back.

    Not ``jt.flag_scope``: the level's setter writes ``amp_reg`` as a side
    effect, and a test that leaves either behind poisons every later file.
    """

    def __init__(self, level):
        self.level = level

    def __enter__(self):
        self.old_level = jt.flags.auto_mixed_precision_level
        self.old_reg = jt.flags.amp_reg
        jt.flags.auto_mixed_precision_level = self.level
        return self

    def __exit__(self, exc_type, exc, tb):
        jt.flags.auto_mixed_precision_level = self.old_level
        jt.flags.amp_reg = self.old_reg
        return False


class TestAmpBitNames(unittest.TestCase):
    def test_the_python_names_match_the_cpp_constants(self):
        header = (Path(jt.__file__).resolve().parent
                  / "src" / "misc" / "nano_string.h").read_text(encoding="utf-8")
        found = dict(
            (name, int(value)) for name, value in
            re.findall(r"constexpr\s+int\s+(amp_\w+)\s*=\s*(\d+)\s*;", header))
        self.assertTrue(found, "no amp_* constexpr found in nano_string.h")
        expected = {
            "amp_prefer32": jt.amp_flags.prefer32,
            "amp_prefer16": jt.amp_flags.prefer16,
            "amp_keep_reduce": jt.amp_flags.keep_reduce,
            "amp_keep_white": jt.amp_flags.keep_white,
            "amp_array_prefer": jt.amp_flags.array_prefer,
        }
        for name, value in expected.items():
            self.assertIn(name, found, f"{name} vanished from nano_string.h")
            self.assertEqual(found[name], value,
                             f"{name} drifted: C++ {found[name]} != Python {value}")

    def test_bit_5_is_the_one_reduce_op_reads_by_number(self):
        # src/ops/reduce_op.cc has no constant for it; it spells `amp_reg & 32`.
        source = (Path(jt.__file__).resolve().parent
                  / "src" / "ops" / "reduce_op.cc").read_text(encoding="utf-8")
        self.assertIn("amp_reg & %d" % jt.amp_flags.reduce16_no_fp32_acc, source)

    def test_the_level_mapping_is_what_the_names_say(self):
        # mirrors setter_auto_mixed_precision_level in src/var.cc
        f = jt.amp_flags
        expected = {
            0: 0,
            1: 0,
            2: 0,
            3: f.keep_reduce | f.keep_white,
            4: f.prefer16,
            5: f.prefer16 | f.array_prefer,
            6: f.prefer16 | f.array_prefer | f.keep_reduce | f.keep_white,
        }
        for level, reg in expected.items():
            with amp_level(level):
                self.assertEqual(jt.flags.amp_reg, reg,
                                 f"level {level} -> amp_reg {jt.flags.amp_reg}, "
                                 f"expected {reg}")


class TestTheRewrittenLiteralsAreTheSameNumbers(unittest.TestCase):
    """Every bare integer this task replaced must still be the same number.

    Naming the bits is only an improvement if the names are right, and getting
    one wrong looks like nothing: ``36`` is ``keep_reduce | 32``, not
    ``keep_reduce | array_prefer`` (which is 20). No dtype assertion in this
    file would have caught that -- conv/matmul's scope only shows up on the
    float16 reduce path. So pin the arithmetic directly.
    """

    def test_conv_and_matmul_used_thirty_six(self):
        f = jt.amp_flags
        self.assertEqual(f.keep_reduce | f.reduce16_no_fp32_acc, 36)

    def test_concat_used_four(self):
        self.assertEqual(jt.amp_flags.keep_reduce, 4)

    def test_acl_used_thirty_two_plus_four(self):
        f = jt.amp_flags
        self.assertEqual(f.reduce16_no_fp32_acc | f.keep_reduce, 32 + 4)

    def test_array_and_random_used_sixteen_one_and_two(self):
        f = jt.amp_flags
        self.assertEqual((f.array_prefer, f.prefer32, f.prefer16), (16, 1, 2))

    def test_no_bare_amp_reg_integer_is_left_in_the_python_tree(self):
        """``amp_reg`` must never meet a numeric literal in Python again.

        AST, not grep: the docstrings in this task deliberately quote the old
        spellings, and a text scan would flag those instead of real code.
        """
        import ast

        root = Path(jt.__file__).resolve().parent
        offenders = []

        def names(node):
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return node.attr
            return None

        def is_int(node):
            return isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool)

        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                hit = False
                if isinstance(node, ast.BinOp) and isinstance(
                        node.op, (ast.BitOr, ast.BitAnd, ast.Add)):
                    sides = (node.left, node.right)
                    hit = (any(names(x) == "amp_reg" for x in sides)
                           and any(is_int(x) for x in sides))
                elif isinstance(node, ast.keyword):
                    hit = node.arg == "amp_reg" and is_int(node.value)
                elif isinstance(node, ast.AugAssign):
                    hit = names(node.target) == "amp_reg" and is_int(node.value)
                elif isinstance(node, ast.Assign):
                    hit = (any(names(t) == "amp_reg" for t in node.targets)
                           and is_int(node.value))
                if hit:
                    offenders.append("%s:%d" % (path.relative_to(root), node.lineno))
        self.assertEqual(
            offenders, [],
            "spell amp_reg bits with jt.amp_flags.<name>, not a number")


class TestConcatKeepsTheCallersPolicy(unittest.TestCase):
    def _pair(self):
        a = jt.array(np.ones((2, 3), dtype="float32"))
        b = jt.array(np.ones((2, 3), dtype="float32"))
        return a, b

    def test_concat_agrees_with_the_ops_around_it(self):
        # Measured before the fix: concat stayed float32 at levels 4, 5 and 6
        # while `a + b` in the same scope was float16.
        for level in (0, 3, 4, 5, 6):
            with amp_level(level):
                a, b = self._pair()
                self.assertEqual(str(jt.concat([a, b], dim=0).dtype),
                                 str((a + b).dtype),
                                 f"concat disagrees with binary ops at level {level}")

    def test_concat_is_float16_under_a_prefer16_policy(self):
        # the same statement, spelled as the value it must produce
        for level in (4, 5, 6):
            with amp_level(level):
                a, b = self._pair()
                self.assertEqual(str(jt.concat([a, b], dim=0).dtype), "float16",
                                 f"level {level}")

    def test_concat_does_not_leak_its_flag_scope(self):
        for level in (0, 4, 6):
            with amp_level(level):
                before = jt.flags.amp_reg
                a, b = self._pair()
                jt.concat([a, b], dim=0).sync()
                self.assertEqual(jt.flags.amp_reg, before)

    def test_concat_values_survive_the_dtype_change(self):
        with amp_level(6):
            a = jt.array(np.full((2, 3), 1.5, dtype="float32"))
            b = jt.array(np.full((2, 3), 2.5, dtype="float32"))
            out = jt.concat([a, b], dim=0)
            np.testing.assert_allclose(
                out.numpy(),
                np.concatenate([np.full((2, 3), 1.5), np.full((2, 3), 2.5)]),
                rtol=1e-3)


class TestArrayAndRandomAgree(unittest.TestCase):
    """The two copies of the array-like preference had drifted apart."""

    def test_a_one_element_result_is_not_downcast_by_either(self):
        # A one-element Var is a scalar to jittor's dtype inference, which skips
        # the preference for it; downcasting it here would make
        # `jt.array(1e-8) * x` disagree with `1e-8 * x`. array() knew that,
        # random() did not: it returned float16 where array() returned float32.
        for level in (5, 6):
            with amp_level(level):
                self.assertEqual(str(jt.array(np.ones(1, "float32")).dtype),
                                 "float32", f"array level {level}")
                self.assertEqual(str(jt.random((1,)).dtype),
                                 "float32", f"random level {level}")

    def test_a_multi_element_result_follows_the_preference_in_both(self):
        for level in (5, 6):
            with amp_level(level):
                self.assertEqual(str(jt.array(np.ones(4, "float32")).dtype),
                                 "float16", f"array level {level}")
                self.assertEqual(str(jt.random((4,)).dtype),
                                 "float16", f"random level {level}")

    def test_without_array_prefer_neither_is_touched(self):
        # level 4 is prefer16 WITHOUT array_prefer
        with amp_level(4):
            self.assertEqual(str(jt.array(np.ones(4, "float32")).dtype), "float32")
            self.assertEqual(str(jt.random((4,)).dtype), "float32")

    def test_a_non_float_array_is_never_downcast(self):
        for level in (5, 6):
            with amp_level(level):
                self.assertEqual(str(jt.array(np.ones(4, "int32")).dtype), "int32")


if __name__ == "__main__":
    unittest.main()
