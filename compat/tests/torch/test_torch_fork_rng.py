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

    def test_devices_may_be_device_objects(self):
        """The list is whatever torch accepts, not necessarily ints.

        MiniMax-H3's VAE passes `[torch.device('cuda:0')]`. Coercing with
        `int()` raised "int() argument must be ... not 'device'" and killed the
        request -- every earlier test here passed `[]` or `[0]`, which is how
        that got through.
        """
        spellings = [[], [torch.device("cpu")]]
        if jt.has_cuda:
            spellings += [[torch.device("cuda:0")], ["cuda:0"], [0]]
        for devices in spellings:
            with self.subTest(repr(devices)):
                with torch.random.fork_rng(devices=devices):
                    jt.random((2,)).sync()

    @unittest.expectedFailure
    def test_forking_away_from_a_fresh_seed(self):
        """Known gap: the CPU RNG state is the seed, not the position.

        `core.py::_get_rng_state` returns `[initial_seed()]` and
        `_set_rng_state` just re-seeds, so a restore **rewinds** to the start of
        the sequence instead of continuing from the fork point. When the fork
        happens right after `manual_seed`, rewinding and continuing coincide --
        which is why every other test in this file passes, and why this gap
        survived: they all capture state immediately after seeding.

        The CUDA side was given real position accounting (cuRAND offsets); the
        CPU side never was. Expected-failure rather than deleted so the
        limitation is visible and this flips the moment someone fixes it.
        """
        torch.manual_seed(7)
        jt.random((5,)).sync()          # advance away from the seed point
        expected = _draw()
        torch.manual_seed(7)
        jt.random((5,)).sync()
        with torch.random.fork_rng(devices=[]):
            jt.random((11,)).sync()
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
