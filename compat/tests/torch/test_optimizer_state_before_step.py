"""``Optimizer.state`` follows torch's defaultdict(dict) read semantics.

torch's ``Optimizer.state`` is a ``defaultdict(dict)``: reading
``optimizer.state[param]`` for a parameter that has no state yet creates and
returns an empty mapping instead of raising ``KeyError``. DeepSpeed's ZeRO-2
mixed-precision linkage reads every optimizer group's entry right after
stepping only the first group (``_lazy_init_hp_params_optimizer_state``), so a
``KeyError`` there aborts the very first training step.

Run: ``python -m pytest compat/tests/torch/test_optimizer_state_before_step.py``
"""

import unittest

import torch


class TestOptimizerStateBeforeStep(unittest.TestCase):
    def test_reading_uninitialized_state_returns_an_empty_mapping(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        state = optimizer.state[parameter]
        self.assertEqual(dict(state), {})

        # A write through the returned mapping reaches the live state view.
        state["step"] = 1
        self.assertEqual(optimizer.state[parameter]["step"], 1.0)

    def test_reading_an_unstepped_group_after_stepping_another(self):
        """The per-group step order DeepSpeed's ZeRO-2 uses."""
        weight = torch.nn.Parameter(torch.ones(4))
        bias = torch.nn.Parameter(torch.ones(2))
        optimizer = torch.optim.AdamW(
            [{"params": [weight]}, {"params": [bias]}], lr=0.1)

        weight.grad = torch.ones(4)
        groups = optimizer.param_groups
        optimizer.param_groups = [groups[0]]
        try:
            optimizer.step()
        finally:
            optimizer.param_groups = groups

        # The second group has not stepped yet; reading its state must answer
        # with an empty mapping, as torch's defaultdict does.
        self.assertEqual(dict(optimizer.state[bias]), {})
        self.assertEqual(optimizer.state[weight]["step"], 1.0)
        self.assertIn("exp_avg", optimizer.state[weight])

    def test_unknown_parameter_still_raises(self):
        parameter = torch.nn.Parameter(torch.ones(3))
        optimizer = torch.optim.SGD([parameter], lr=0.1)
        stranger = torch.nn.Parameter(torch.ones(3))
        with self.assertRaises(KeyError):
            optimizer.state[stranger]


if __name__ == "__main__":
    unittest.main()
