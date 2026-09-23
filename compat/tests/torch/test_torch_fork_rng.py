"""`torch.random.fork_rng`: run a block, then put the RNG back.

MiniMax-H3's reference-to-video path wraps its sampling in it, and without it
the request died with `module 'torch.random' has no attribute 'fork_rng'` --
after the collective deadlock upstream of it was fixed, this was the next thing
in the way.
"""
import unittest

import jittor as jt
import numpy as np
import torch


def _draw(n=4):
    return np.asarray(jt.random((n,)).numpy()).copy()


class TestForkRng(unittest.TestCase):
    def _baseline(self):
        torch.manual_seed(7)
        return _draw()

    def test_it_exists_on_the_random_module(self):
        self.assertTrue(hasattr(torch.random, "fork_rng"))

    def test_the_stream_continues_as_if_the_block_had_not_run(self):
        expected = self._baseline()
        torch.manual_seed(7)
        with torch.random.fork_rng(devices=[]):
            jt.random((4,)).sync()
            jt.random((9,)).sync()
        np.testing.assert_allclose(_draw(), expected)

    def test_disabled_is_a_no_op(self):
        # torch lets a caller keep one code path and switch the forking off;
        # then the block's draws must count.
        expected = self._baseline()
        torch.manual_seed(7)
        with torch.random.fork_rng(devices=[], enabled=False):
            jt.random((4,)).sync()
        self.assertFalse(np.allclose(_draw(), expected))

    def test_an_exception_still_restores(self):
        # A block that raised has still consumed randomness. Leaving the stream
        # advanced would make the next draw depend on whether an unrelated
        # error happened.
        expected = self._baseline()
        torch.manual_seed(7)
        with self.assertRaises(ValueError):
            with torch.random.fork_rng(devices=[]):
                jt.random((4,)).sync()
                raise ValueError("boom")
        np.testing.assert_allclose(_draw(), expected)

    def test_it_can_be_built_then_entered_later(self):
        # Written as a class, not @contextmanager, so the state is captured on
        # __enter__ rather than when the object is made.
        expected = self._baseline()
        torch.manual_seed(7)
        manager = torch.random.fork_rng(devices=[])
        jt.random((3,)).sync()          # before entering: must count
        torch.manual_seed(7)
        with manager:
            jt.random((4,)).sync()
        np.testing.assert_allclose(_draw(), expected)

    @unittest.skipUnless(jt.has_cuda, "no CUDA device")
    def test_device_states_round_trip(self):
        with jt.flag_scope(use_cuda=1):
            torch.cuda.manual_seed(11)
            expected = _draw()
            torch.cuda.manual_seed(11)
            with torch.random.fork_rng(devices=[0]):
                jt.random((16,)).sync()
            np.testing.assert_allclose(_draw(), expected)


if __name__ == "__main__":
    unittest.main()
