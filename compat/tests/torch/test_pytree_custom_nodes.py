"""Custom pytree registration must affect traversal and reconstruction."""
from collections import OrderedDict, defaultdict
import os
import unittest

import torch
from torch.utils import _pytree as pytree


class TestCustomPytreeNodes(unittest.TestCase):
    def test_dict_key_order_and_key_types(self):
        keyed, context = pytree._dict_flatten_with_keys({"z": 2, "a": 3})
        self.assertEqual(context, ["z", "a"])
        self.assertEqual(keyed, [(pytree.MappingKey("z"), 2), (pytree.MappingKey("a"), 3)])
        self.assertIsInstance(pytree._list_flatten_with_keys([4])[0][0][0], pytree.SequenceKey)

    def test_registered_dict_subclass_roundtrip_and_mapping(self):
        class Output(dict):
            pass

        pytree.register_pytree_node(
            Output, pytree._dict_flatten,
            lambda values, keys: Output(zip(keys, values)),
            serialized_type_name="test.Output",
            flatten_with_keys_fn=pytree._dict_flatten_with_keys,
        )
        original = Output(hidden=[1, 2], pooler=(3,))
        leaves, spec = pytree.tree_flatten(original)
        self.assertEqual(leaves, [1, 2, 3])
        restored = pytree.tree_unflatten(leaves, spec)
        self.assertIs(type(restored), Output)
        self.assertEqual(restored, original)
        mapped = pytree.tree_map(lambda value: value * 10, original)
        self.assertIs(type(mapped), Output)
        self.assertEqual(mapped, Output(hidden=[10, 20], pooler=(30,)))
        keyed, keyed_spec = pytree.tree_flatten_with_path(original)
        self.assertEqual(keyed, [
            ((pytree.MappingKey("hidden"), pytree.SequenceKey(0)), 1),
            ((pytree.MappingKey("hidden"), pytree.SequenceKey(1)), 2),
            ((pytree.MappingKey("pooler"), pytree.SequenceKey(0)), 3),
        ])
        self.assertIs(type(pytree.tree_unflatten([v for _, v in keyed], keyed_spec)), Output)
        for path, expected in keyed:
            current = original
            for entry in path:
                current = entry.get(current)
            self.assertEqual(current, expected)

    def test_builtin_mapping_subclasses_preserve_type_and_factory(self):
        for value in (OrderedDict(z=[1], a=2), defaultdict(list, z=[1], a=2)):
            with self.subTest(kind=type(value).__name__):
                mapped = pytree.tree_map(lambda item: item + 1, value)
                self.assertIs(type(mapped), type(value))
                self.assertEqual(list(mapped.items()), [("z", [2]), ("a", 3)])
                if isinstance(mapped, defaultdict):
                    self.assertIs(mapped.default_factory, list)
                    self.assertEqual(mapped["missing"], [])

    def test_registration_is_effective_for_non_dict_nodes_and_gradients(self):
        class Pair:
            def __init__(self, left, right):
                self.left, self.right = left, right

        pytree.register_pytree_node(
            Pair, lambda pair: ([pair.left, pair.right], None),
            lambda values, _: Pair(*values),
            flatten_with_keys_fn=lambda pair: ([
                (pytree.GetAttrKey("left"), pair.left),
                (pytree.GetAttrKey("right"), pair.right),
            ], None),
        )
        device = os.environ.get("JITTOR_TEST_DEVICE", "cpu")
        x = torch.tensor([1., 2.], device=device, requires_grad=True)
        pair = pytree.tree_map(lambda value: value * 3, Pair(x, x * 2))
        self.assertIs(type(pair), Pair)
        (pair.left.sum() + pair.right.sum()).backward()
        self.assertTrue(str(x.grad.device).startswith(device))
        self.assertEqual(x.grad.detach().cpu().tolist(), [9., 9.])

    def test_duplicate_registration_is_rejected(self):
        class Node:
            pass

        def flatten(value):
            return [], None

        def unflatten(values, context):
            return Node()
        pytree.register_pytree_node(Node, flatten, unflatten)
        with self.assertRaises(ValueError):
            pytree.register_pytree_node(Node, flatten, unflatten)

    def test_missing_key_callback_is_rejected_on_path_traversal(self):
        class Node:
            pass

        pytree.register_pytree_node(Node, lambda value: ([1], None), lambda xs, ctx: Node())
        with self.assertRaises(ValueError):
            pytree.tree_flatten_with_path(Node())

    def test_malformed_leaf_count_is_rejected(self):
        _, spec = pytree.tree_flatten({"x": [1, 2]})
        for leaves in ([1], [1, 2, 3]):
            with self.subTest(leaves=leaves), self.assertRaises(ValueError):
                pytree.tree_unflatten(leaves, spec)

    def test_unregistered_dict_subclass_is_a_leaf(self):
        class Unregistered(dict):
            pass

        value = Unregistered(x=3)
        leaves, _ = pytree.tree_flatten(value)
        self.assertEqual(len(leaves), 1)
        self.assertIs(leaves[0], value)

    def test_legacy_registration_delegates_to_same_registry(self):
        class Legacy:
            pass

        pytree._register_pytree_node(Legacy, lambda node: ([7], None), lambda xs, ctx: Legacy())
        leaves, spec = pytree.tree_flatten(Legacy())
        self.assertEqual(leaves, [7])
        self.assertIs(type(pytree.tree_unflatten(leaves, spec)), Legacy)


if __name__ == "__main__":
    unittest.main()
