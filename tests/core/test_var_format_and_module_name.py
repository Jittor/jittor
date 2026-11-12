"""Python protocol details of Var and Module.

``format(var, spec)`` used to call ``item()`` for every Var, so ``f"{v}"`` of
a matrix raised; a multi-element Var now formats like ``object.__format__``.
``Module`` carried a never-called ``__name__`` stub method, so
``functools.wraps(module_instance)`` set ``__name__`` to a bound method and
failed.
"""
import functools
import unittest

import jittor as jt


class TestVarFormat(unittest.TestCase):
    def test_scalar_spec(self):
        self.assertEqual(f"{jt.array(1.2345):.2f}", "1.23")

    def test_multi_element(self):
        v = jt.ones((2, 2))
        self.assertEqual(f"{v}", str(v))
        with self.assertRaises(TypeError):
            f"{v:.2f}"


class TestModuleName(unittest.TestCase):
    def test_instances_have_no_name(self):
        module = jt.nn.Linear(2, 2)
        with self.assertRaises(AttributeError):
            module.__name__
        self.assertEqual(type(module).__name__, "Linear")

    def test_wraps_module_instance(self):
        module = jt.nn.Linear(2, 2)

        @functools.wraps(module)
        def wrapper():
            return None
        self.assertEqual(wrapper.__name__, "wrapper")


if __name__ == "__main__":
    unittest.main()
