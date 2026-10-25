"""Malformed ``jt.code`` source is a caller error, not an internal invariant.

``jt.code``'s ``cpu_src`` is public input. A misspelt ``@var``, an incomplete
``#`` directive or a wrong ``@if``/``@define`` arity used to escape the
``@``-parser as a plain ``LOGf``, so the reader got an internal-invariant note
and a bare ``file:line`` instead of the construct they wrote.
"""

import unittest

import jittor as jt
import numpy as np


_CASES = {
    "undefined_var": "@out(0)=@nosuchvar;",
    "bare_hash": "#",
    "bad_if_arity": "@out(0)=@if(1);",
    "undefined_macro": "@out(0)=@nosuchmacro();",
    "bad_strcmp_arity": "@out(0)=@strcmp(1);",
}


class TestJitSourceErrorCategory(unittest.TestCase):
    def test_bad_source_is_reported_as_the_callers_error(self):
        for name, source in _CASES.items():
            with self.subTest(name=name):
                with self.assertRaises(Exception) as caught:
                    jt.code([4], "float32", cpu_src=source).sync()
                message = str(caught.exception)
                self.assertNotIn("internal Jittor invariant", message)
                self.assertNotIn("report this issue", message)
                self.assertTrue(
                    any(token in message for token in (
                        "JIT variable", "Incomplete # directive", "@if",
                        "@define", "@is_def", "@strcmp", "@for", "JIT macro",
                        "Unbalanced")),
                    message,
                )

    def test_a_valid_source_still_compiles(self):
        value = jt.code([4], "float32", cpu_src="for (int i=0; i<out0_shape0; ++i) @out(i)=1.0;")
        np.testing.assert_array_equal(value.numpy(), [1.0] * 4)


if __name__ == "__main__":
    unittest.main()
