"""Which leaves ``loss.backward()`` publishes ``.grad`` onto.

Jittor has no Python-side graph walk that can recover every leaf requiring
grad, so the shim keeps a registry of them. That registry used to be filled
only by whoever iterated ``named_parameters()``, which left a training loop
that used no optimizer -- backward, then read ``.grad`` -- with ``None`` on
every weight, silently, where Torch fills them. A module cannot contribute to
a loss without being called, so being called is what publishes its parameters.

Each case runs in its own interpreter on purpose. The registry is process
global and ``backward()`` deliberately prunes parameters outside an optimizer
once one exists, so a process that has built an optimizer -- as the rest of
this directory does at import time -- cannot observe the no-optimizer
contract these pin.
"""

import os
import unittest

from _helpers.child_process import run_python_child


_SCRIPT = """
import torch

layer = torch.nn.Linear(4, 3)
%(touch)s
(layer(torch.randn(2, 4)) ** 2).sum().backward()
filled = sum(1 for p in layer.parameters() if getattr(p, "grad", None) is not None)
print("FILLED", filled, len(list(layer.parameters())))
"""

_NESTED = """
import torch

model = torch.nn.Sequential(
    torch.nn.Linear(4, 5), torch.nn.ReLU(), torch.nn.Linear(5, 2))
(model(torch.randn(3, 4)) ** 2).sum().backward()
named = list(model.named_parameters())
filled = sum(1 for _, p in named if getattr(p, "grad", None) is not None)
print("FILLED", filled, len(named))
"""


def _run(script):
    finished = run_python_child(["-c", script], env={"JITTOR_TORCH_SHIM": "1"},
                                timeout=900)
    for line in finished.stdout.splitlines():
        if line.startswith("FILLED"):
            _, filled, total = line.split()
            return int(filled), int(total)
    raise AssertionError(
        "no result:\nstdout:\n%s\nstderr:\n%s"
        % (finished.stdout[-2000:], finished.stderr[-2000:]))


class TestBackwardPublishesModuleParameters(unittest.TestCase):
    def test_backward_fills_grad_without_touching_parameters_first(self):
        self.assertEqual(_run(_SCRIPT % {"touch": ""}), (2, 2))

    def test_touching_parameters_first_changes_nothing(self):
        self.assertEqual(
            _run(_SCRIPT % {"touch": "list(layer.parameters())"}), (2, 2))

    def test_a_nested_module_publishes_its_children(self):
        self.assertEqual(_run(_NESTED), (4, 4))


if __name__ == "__main__":
    unittest.main()
