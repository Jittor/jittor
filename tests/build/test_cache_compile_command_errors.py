"""The compiler-command scanner is a caller boundary.

``cache_compile(cmd, ...)`` parses the command line it is handed. The command
line can carry user flags, so one this scanner cannot read is the caller's
error and must be catchable. Failures that are not about the command line -- a
compiler that produced no depfile, a file that cannot be opened, written or
renamed -- stay internal and are not exercised here.
"""

import unittest

import jittor as jt  # noqa: F401  (initialises jittor.build.compiler)
from jittor.build import compiler as compiler_state


CASES = {
    "unterminated_quote": ("g++ -o a.o 'a.cc", "unterminated quote"),
    "output_flag_without_value": (
        "g++ -o ", "ends after naming an output flag"),
    "duplicate_output": (
        "g++ -o a.o b.cc -o c.o", "exactly one output"),
    "include_flag_without_directory": (
        "g++ -o a.o b.cc -I ", "include flag ('-I') with no"),
    "no_input": ("g++ -o a.o", "at least one input file"),
    "no_output": ("g++ a.cc", "at least one input file"),
}


class TestCacheCompileCommandErrors(unittest.TestCase):
    def setUp(self):
        cc = compiler_state.jit_utils.cc
        if cc is None:
            self.skipTest("jit_utils_core is not built in this environment")
        self.cache_compile = cc.cache_compile

    def test_bad_command_is_a_catchable_user_error(self):
        for name, (cmd, sentence) in CASES.items():
            with self.subTest(name=name):
                with self.assertRaises(Exception) as caught:
                    self.cache_compile(cmd, "", "")
                message = str(caught.exception)
                self.assertNotIn("internal Jittor invariant", message)
                self.assertNotIn("report this issue", message)
                self.assertIn(sentence, message)


if __name__ == "__main__":
    unittest.main()
