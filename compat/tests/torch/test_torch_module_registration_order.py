"""Submodules and parameters are listed in the order torch registers them.

torch registers a submodule or parameter when one is first assigned; a name
that held something else before -- ``self.mid_block = None``, filled in later,
as diffusers' UNets do -- is registered at the later assignment. The shim kept
the position of the first assignment, so an SD1.5 UNet listed its mid block
before its up blocks, and code that pairs two models' parameters by position
(a copy, an EMA, a checkpoint converter) paired the wrong tensors.
"""

import unittest

import torch
from torch import nn


class _Late(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Linear(2, 2)
        self.middle = None
        self.last = nn.Linear(2, 2)
        self.middle = nn.Linear(2, 2)
        self.scale = None
        self.scale = nn.Parameter(torch.ones(2))


class TestRegistrationOrder(unittest.TestCase):

    def test_a_late_submodule_is_listed_where_torch_lists_it(self):
        # What torch 2.11 lists for this module: its own parameters first, then
        # each child in registration order.
        names = [name for name, _ in _Late().named_parameters()]
        self.assertEqual(names, ["scale", "first.weight", "first.bias", "last.weight",
                                 "last.bias", "middle.weight", "middle.bias"])
        self.assertEqual([name for name, _ in _Late().named_children()],
                         ["first", "last", "middle"])

    def test_rebinding_a_submodule_keeps_its_place(self):
        model = _Late()
        model.first = nn.Linear(2, 2)
        names = [name for name, _ in model.named_children()]
        self.assertEqual(names, ["first", "last", "middle"])


if __name__ == "__main__":
    unittest.main()
