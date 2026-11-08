"""Reading a backend hook answers from the registry, quickly when nothing is there."""
import unittest

import jittor as jt
from jittor._runtime import dispatch
from jittor.nn.backends import hooks


class TestBackendHookLookup(unittest.TestCase):

    def test_an_unregistered_hook_is_none(self):
        self.assertFalse(dispatch._is_registered("nn.constant_pad"))
        self.assertIsNone(hooks.acl_constant_pad)

    def test_a_registered_hook_is_returned_and_dispatches(self):
        backend = dispatch.dispatch_context().backend
        calls = []

        def kernel(x, padding, value):
            calls.append((tuple(padding), value))
            return x

        with dispatch.override_kernel("nn.constant_pad", backend, kernel):
            hook = hooks.acl_constant_pad
            self.assertIsNotNone(hook)
            x = jt.ones((2, 2))
            self.assertIs(hook(x, [1, 1], 0.0), x)
        self.assertEqual(calls, [((1, 1), 0.0)])
        self.assertIsNone(hooks.acl_constant_pad)

    def test_a_kernel_on_another_backend_is_not_this_ones_hook(self):
        backend = dispatch.dispatch_context().backend
        other = "acl" if backend != "acl" else "cpu"
        with dispatch.override_kernel("nn.constant_pad", other, lambda *a: None):
            self.assertTrue(dispatch._is_registered("nn.constant_pad"))
            self.assertIsNone(hooks.acl_constant_pad)

    def test_unknown_names_still_raise(self):
        with self.assertRaises(AttributeError):
            hooks.no_such_hook

    def test_pad_still_pads(self):
        x = jt.array([[1.0, 2.0], [3.0, 4.0]])
        out = jt.nn.pad(x, (1, 0), value=9.0)
        self.assertEqual(out.numpy().tolist(), [[9.0, 1.0, 2.0], [9.0, 3.0, 4.0]])


if __name__ == "__main__":
    unittest.main()
