# ***************************************************************
# Copyright (c) 2026 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Which JT_* macros the environment can turn on, declared in one place.

`cache_compile.cc` used to discover these by scanning every source it was
already reading for dependencies, and then rewriting the compiler command line
in place. Two unrelated jobs in one pass: deciding the command line (before a
compile) and collecting dependencies (only knowable after one). While they
shared a scanner, the compiler's own `-MD -MF` could not replace the
hand-written include scanner at all, because the first cold compile would have
gone out without its `-D`.

The list now lives in `compiler.JT_CONFIG_MACROS`. A declared list can go stale,
so this greps the sources and fails when it does.
"""

import os
import re
import unittest

import jittor.compiler as compiler


SOURCE_ROOTS = ("src", "extern")
SOURCE_SUFFIXES = (".cc", ".h", ".cu", ".cuh")
CONDITIONAL = re.compile(r"^\s*#\s*(?:ifdef|ifndef|if)\b([^\r\n]*)", re.M)
JT_MACRO = re.compile(r"\bJT_[A-Za-z0-9_]+\b")


def _macros_used_in_sources():
    found = set()
    root = os.path.dirname(compiler.__file__)
    for directory in SOURCE_ROOTS:
        for current, _dirs, files in os.walk(os.path.join(root, directory)):
            for name in files:
                if not name.endswith(SOURCE_SUFFIXES):
                    continue
                path = os.path.join(current, name)
                with open(path, encoding="utf-8", errors="replace") as handle:
                    for condition in CONDITIONAL.findall(handle.read()):
                        found.update(JT_MACRO.findall(condition))
    return found


class TestJtConfigMacros(unittest.TestCase):
    def test_the_declared_list_matches_the_sources(self):
        used = _macros_used_in_sources()
        declared = set(compiler.JT_CONFIG_MACROS)
        self.assertEqual(
            used - declared, set(),
            "a source tests one of these with #ifdef but compiler.py will "
            "never define it, so setting the environment variable does "
            "nothing: add it to JT_CONFIG_MACROS")
        self.assertEqual(
            declared - used, set(),
            "JT_CONFIG_MACROS names a macro no source tests any more; "
            "remove it")

    def test_an_unset_macro_adds_no_flag(self):
        self.assertEqual(compiler.jt_config_macro_flags({}), "")

    def test_save_mem_off_spellings_match_the_cache_config(self):
        for value in ("", "0", "false", "off", "no"):
            self.assertEqual(
                compiler.jt_config_macro_flags({"JT_SAVE_MEM": value}), "",
                value)

    def test_a_set_macro_becomes_a_define(self):
        flags = compiler.jt_config_macro_flags({"JT_SAVE_MEM": "1"})
        self.assertIn("-DJT_SAVE_MEM=1", flags)
        self.assertEqual(flags.count("-DJT_SAVE_MEM=1"), 1)
        # One token, and separated from its neighbours -- these are pasted
        # straight into a command line.
        self.assertTrue(flags.startswith(" ") and flags.endswith(" "))
        self.assertEqual(len(flags.split()), 1)

    def test_the_order_does_not_depend_on_the_environment(self):
        """The flags land in the cache key, so two processes that set the same
        macros must produce the same command line."""
        first = compiler.jt_config_macro_flags(
            {"JT_SAVE_MEM": "1", "JT_HAS_HALF_SIMD": "1"})
        second = compiler.jt_config_macro_flags(
            {"JT_HAS_HALF_SIMD": "1", "JT_SAVE_MEM": "1"})
        self.assertEqual(first, second)

    def test_nothing_is_set_by_default(self):
        """So this change moves no command line for anyone who has not opted
        in, which is why it costs no rebuild."""
        self.assertEqual(compiler.jt_config_macro_flags(os.environ), "")


if __name__ == "__main__":
    unittest.main()
