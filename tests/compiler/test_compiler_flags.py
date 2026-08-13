"""Regression tests for the shared compiler flag parser."""

from __future__ import print_function

import unittest

import jittor.compiler as compiler
import jittor_utils
from jittor_utils.compiler_flags import remove_flags, shsplit


class TestCompilerFlags(unittest.TestCase):
    def test_legacy_compiler_names_resolve_to_shared_helpers(self):
        self.assertIs(compiler.shsplit, shsplit)
        self.assertIs(compiler.remove_flags, remove_flags)
        self.assertIs(compiler.try_find_exe, jittor_utils.try_find_exe)

    def test_quoted_arguments_keep_embedded_spaces(self):
        flags = '-I"path with spaces" -DNAME="hello world" -shared input.cc'
        self.assertEqual(
            shsplit(flags),
            ['-I"path with spaces"', '-DNAME="hello world"', '-shared', 'input.cc'],
        )

    def test_remove_flags_preserves_unmatched_argument_spelling(self):
        flags = '-I"path with spaces" -L/tmp -shared input.cc output.o'
        self.assertEqual(
            remove_flags(flags, ["-L", "-shared", ".o"]),
            '-I"path with spaces" input.cc',
        )


if __name__ == "__main__":
    unittest.main()
