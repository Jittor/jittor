# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""FSDP2's flat-sharding policy is configurable, not two literals (8.11).

``_fsdp2_flat_enabled`` used to end in
``world_size <= 2 or total_numel <= 1_000_000``. Both numbers came from one set
of measurements on one machine, and behaviour changed abruptly at 3 ranks and
at 1.1M parameters with no way to try another value short of editing the
source. They are defaults now; the decision itself is unchanged, so nothing
that did not set the new variables behaves differently.
"""
import os
import unittest

from jittor.compat.fsdp2.common import _fsdp2_flat_enabled


class TestFsdp2FlatPolicy(unittest.TestCase):

    def setUp(self):
        self.saved = {
            name: os.environ.get(name)
            for name in ("JITTOR_FSDP2_FLAT",
                         "JITTOR_FSDP2_FLAT_MAX_WORLD_SIZE",
                         "JITTOR_FSDP2_FLAT_MAX_NUMEL")
        }
        for name in self.saved:
            os.environ.pop(name, None)
        self.addCleanup(self._restore)

    def _restore(self):
        for name, value in self.saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    def test_defaults_are_what_they_were(self):
        # The whole point is that this is a refactor, not a policy change.
        self.assertTrue(_fsdp2_flat_enabled(2, 10_000_000))
        self.assertFalse(_fsdp2_flat_enabled(4, 10_000_000))
        self.assertTrue(_fsdp2_flat_enabled(4, 1_000_000))
        self.assertFalse(_fsdp2_flat_enabled(4, 1_000_001))

    def test_the_rank_threshold_moves(self):
        os.environ["JITTOR_FSDP2_FLAT_MAX_WORLD_SIZE"] = "8"
        self.assertTrue(_fsdp2_flat_enabled(8, 10_000_000))
        self.assertFalse(_fsdp2_flat_enabled(9, 10_000_000))

    def test_the_size_threshold_moves(self):
        os.environ["JITTOR_FSDP2_FLAT_MAX_NUMEL"] = "50"
        self.assertTrue(_fsdp2_flat_enabled(16, 50))
        self.assertFalse(_fsdp2_flat_enabled(16, 51))

    def test_the_override_still_wins_over_both(self):
        os.environ["JITTOR_FSDP2_FLAT_MAX_WORLD_SIZE"] = "0"
        os.environ["JITTOR_FSDP2_FLAT_MAX_NUMEL"] = "0"
        os.environ["JITTOR_FSDP2_FLAT"] = "1"
        self.assertTrue(_fsdp2_flat_enabled(1024, 10 ** 12))
        os.environ["JITTOR_FSDP2_FLAT"] = "0"
        self.assertFalse(_fsdp2_flat_enabled(1, 1))

    def test_a_non_number_is_a_hard_error(self):
        """Silently falling back to the default is how a tuning run lies.

        Someone who mistypes the variable would otherwise get the default
        policy and a benchmark that says the threshold does not matter.
        """
        os.environ["JITTOR_FSDP2_FLAT_MAX_NUMEL"] = "1e6"
        with self.assertRaises(ValueError) as caught:
            _fsdp2_flat_enabled(4, 10)
        self.assertIn("JITTOR_FSDP2_FLAT_MAX_NUMEL", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
