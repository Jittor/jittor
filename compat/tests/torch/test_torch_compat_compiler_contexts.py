"""Eager context contracts required by compiler-facing libraries."""

import unittest

from torch._inductor.custom_graph_pass import CustomGraphPass
from torch._subclasses.fake_tensor import (
    FakeTensorMode,
    unset_fake_temporarily,
)


class TestFakeTensorContexts(unittest.TestCase):
    def test_custom_graph_pass_keeps_abstract_contract(self):
        with self.assertRaises(TypeError):
            CustomGraphPass()

        class CompletePass(CustomGraphPass):
            def __call__(self, graph):
                return graph

            def uuid(self):
                return "complete"

        graph_pass = CompletePass()
        marker = object()
        self.assertIs(graph_pass(marker), marker)
        self.assertEqual(graph_pass.uuid(), "complete")

    def test_fake_tensor_mode_returns_itself(self):
        mode = FakeTensorMode(allow_non_fake_inputs=True)
        with mode as active:
            self.assertIs(active, mode)

    def test_unset_fake_temporarily_is_reentrant(self):
        with unset_fake_temporarily() as outer:
            self.assertIsNone(outer)
            with unset_fake_temporarily() as inner:
                self.assertIsNone(inner)

    def test_contexts_do_not_swallow_exceptions(self):
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with unset_fake_temporarily(), FakeTensorMode():
                raise RuntimeError("sentinel")


if __name__ == "__main__":
    unittest.main()
