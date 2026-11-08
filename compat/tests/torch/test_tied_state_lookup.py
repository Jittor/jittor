"""Public lookup must accept every registered name of tied parameters/buffers."""

import unittest

import torch


class TiedState(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.first = torch.nn.Linear(3, 3, bias=False)
        self.second = torch.nn.Linear(3, 3, bias=False)
        self.second.weight = self.first.weight
        shared = torch.zeros(3)
        self.register_buffer("first_buffer", shared)
        self.register_buffer("second_buffer", shared)


class TestTiedStateLookup(unittest.TestCase):
    def test_parameter_name_enumeration_obeys_remove_duplicate(self):
        model = TiedState()
        self.assertEqual([name for name, _ in model.named_parameters()], ["first.weight"])
        self.assertEqual([name for name, _ in model.named_parameters(remove_duplicate=False)],
                         ["first.weight", "second.weight"])

    def test_tied_parameter_lookup_keeps_alias_identity(self):
        model = TiedState()
        self.assertIs(model.get_parameter("first.weight"), model.first.weight)
        self.assertIs(model.get_parameter("second.weight"), model.first.weight)
        with self.assertRaises(AttributeError):
            model.get_parameter("second_buffer")

    def test_buffer_name_enumeration_obeys_remove_duplicate(self):
        model = TiedState()
        self.assertEqual([name for name, _ in model.named_buffers()], ["first_buffer"])
        self.assertEqual([name for name, _ in model.named_buffers(remove_duplicate=False)],
                         ["first_buffer", "second_buffer"])

    def test_tied_buffer_lookup_keeps_alias_identity(self):
        model = TiedState()
        self.assertIs(model.get_buffer("first_buffer"), model.first_buffer)
        self.assertIs(model.get_buffer("second_buffer"), model.first_buffer)
        with self.assertRaises(AttributeError):
            model.get_buffer("second.weight")


if __name__ == "__main__":
    unittest.main()
