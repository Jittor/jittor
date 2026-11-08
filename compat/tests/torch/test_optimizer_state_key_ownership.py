# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The Torch state key <-> native buffer mapping has one owner.

``optimizer_state.py`` is the live ``state`` view, ``state_dict()`` and
``load_state_dict()`` all translate the same names.  When the table lived in
each caller, adding a state key meant editing three places and one of them
silently went stale.  These tests pin the table's coverage and prove the
adapter kept no private copy.
"""

from pathlib import Path
import unittest

from jittor.compat import optimizer_kinds
from jittor.compat.torch.optimizer_state import (
    _NATIVE_TO_TORCH_STATE, _TORCH_TO_NATIVE_STATE,
)


class TestOptimizerStateKeyOwnership(unittest.TestCase):
    def test_every_known_optimizer_has_a_mapping(self):
        self.assertEqual(set(_TORCH_TO_NATIVE_STATE),
                         set(optimizer_kinds.KNOWN_KINDS))

    def test_the_two_tables_are_inverses(self):
        for kind, names in _TORCH_TO_NATIVE_STATE.items():
            self.assertEqual(_NATIVE_TO_TORCH_STATE[kind],
                             {native: torch for torch, native in names.items()},
                             kind)
            self.assertEqual(
                {torch: native for native, torch in _NATIVE_TO_TORCH_STATE[kind].items()},
                names, kind)

    def test_the_adapter_keeps_no_private_copy(self):
        source = (Path(__file__).resolve().parents[2]
                  / "torch" / "optimizer_api.py").read_text(encoding="utf-8")
        for literal in (
            '"exp_avg": "m"',
            '"exp_avg_sq": "values"',
            '"momentum_buffer": "values"',
            '"square_avg": "values"',
            '"sum": "values"',
            '"exp_avg_diff": "d"',
        ):
            self.assertNotIn(literal, source,
                             "optimizer_api still owns a state-name mapping")


if __name__ == "__main__":
    unittest.main()
