"""Two adapters rebuild four-axis jagged positions; they must share the axis.

vLLM/TensorDict restore (``tensordict/consolidated_transport.py``) and the
verl upstream patch (``verl/jagged_positions.py``) both reconstruct a 4-axis
``position_ids`` nested tensor.  They were written independently and both
chose ``jagged_dim=2``; if one side later changes the axis, the same object
reaches the other adapter with a different layout and the failure is silent.

This is a tripwire, not a numerical proof: the two module test suites
(``test_four_axis_transport.py`` and ``test_jagged_positions.py``) own the
behavioral fixtures.  Here we pin the shared spelling that keeps them aligned.
"""

from pathlib import Path
import unittest


ADAPTERS = Path(__file__).resolve().parents[1]


def _source(relative):
    return (ADAPTERS / relative).read_text(encoding="utf-8")


class SharedJaggedContract(unittest.TestCase):
    def test_both_four_axis_rebuilds_use_jagged_dim_two(self):
        for relative in (
            "jittor_adapters/tensordict/consolidated_transport.py",
            "jittor_adapters/verl/jagged_positions.py",
        ):
            source = _source(relative)
            self.assertIn("nested_tensor_from_jagged", source, relative)
            self.assertIn("jagged_dim=2", source, relative)

    def test_both_take_offsets_from_the_nested_tensor(self):
        self.assertIn("offsets=offsets",
                      _source("jittor_adapters/tensordict/consolidated_transport.py"))
        self.assertIn("offsets=positions.offsets()",
                      _source("jittor_adapters/verl/jagged_positions.py"))


if __name__ == "__main__":
    unittest.main()
