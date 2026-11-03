"""``torch.fx._pytree`` publishes the flatten-spec registry transformers uses.

``transformers.cache_utils`` registers a flatten spec for ``DynamicCache`` at
import time; without the submodule every ``AutoModel`` import (and the whole
``accelerate launch`` CLI) fails before any model is built.
"""

from __future__ import print_function

import unittest

import torch


class TestTorchFxPytree(unittest.TestCase):
    def test_publishes_the_registry_and_dict_flattening_helpers(self):
        pytree = torch.fx._pytree
        self.assertTrue(callable(pytree.register_pytree_flatten_spec))
        self.assertTrue(callable(pytree._dict_flatten_spec))

    def test_registers_a_flatten_spec_and_flattens_dict_in_context_order(self):
        class Node(object):
            pass

        def flatten_spec(node, spec):
            return [node]

        torch.fx._pytree.register_pytree_flatten_spec(Node, flatten_spec)
        self.assertIn(Node, torch.fx._pytree.SUPPORTED_NODES)
        self.assertIs(
            torch.fx._pytree.SUPPORTED_NODES[Node], flatten_spec)
        self.assertIn(Node, torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH)
        self.assertIsNone(
            torch.fx._pytree.SUPPORTED_NODES_EXACT_MATCH[Node])

        class Spec(object):
            context = ("second", "first")

        self.assertEqual(
            torch.fx._pytree._dict_flatten_spec(
                {"first": 1, "second": 2}, Spec()),
            [2, 1])


if __name__ == "__main__":
    unittest.main()
