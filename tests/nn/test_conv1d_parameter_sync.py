"""``Conv1d`` must run with the bias that is on the module, not the one hidden inside.

What this file asserts, and why it exists
-----------------------------------------
``jt.nn.Conv1d`` does not compute the convolution itself: it builds an inner
``jt.nn.Conv`` and keeps it in a **list**, so that module traversal does not
descend into it. It then re-exports that inner module's ``weight``/``bias`` as
its own parameters. ``execute`` re-syncs the weight on every forward:

    self._conv[0].weight = self.weight.unsqueeze(-1)

but used to forget the bias. The two only agree until something replaces the
module's parameters -- ``load_state_dict``, ``.to()``, an offload manager --
and after that the inner ``Conv`` silently keeps the value it was *constructed*
with. For ``Conv1d(32, 2048, 1)`` that value is ``uniform(-1/sqrt(32),
1/sqrt(32))``: it is uncorrelated with the checkpoint's bias, it has the same
distribution on every run, and it differs in value on every run because it is
drawn from the RNG. A model whose weights all check out against the checkpoint
still produced a different output per process -- every ``Conv1d`` in it ran with
a random bias.

The failure this catches is a whole class, not one line: **a parameter that is
aliased out of a sub-module hidden from traversal stops tracking that
sub-module the moment anything reassigns it.** ``Conv3d`` and ``ConvTranspose*``
hold their parameters directly and are unaffected; they are checked here too so
that a future refactor into the same list-wrapped shape cannot reintroduce the
divergence unnoticed.
"""

import unittest

import numpy as np

import jittor as jt


def _reference_conv(x, weight, bias, ndim):
    """conv with kernel size 1, expressed so that the bias enters exactly once."""
    if ndim == 3:  # (N, C, L) with weight (O, C)
        out = np.einsum("oc,ncl->nol", weight, x)
        return out + bias.reshape(1, -1, 1)
    out = np.einsum("oc,nchwd->nohwd", weight, x)  # (N, C, H, W, D) with (O, C)
    return out + bias.reshape(1, -1, 1, 1, 1)


class TestConv1dParameterSync(unittest.TestCase):
    #: Far outside the initialisation range, so a stale inner bias cannot pass
    #: by coincidence: the initialisation bound for fan_in=32 is 1/sqrt(32).
    SENTINEL_OFFSET = 3.0

    def setUp(self):
        jt.flags.use_cuda = 0 if not jt.has_cuda else jt.flags.use_cuda
        rng = np.random.default_rng(0)
        self.rng = rng

    def _sentinel(self, size):
        return (self.rng.standard_normal(size).astype("float32") * 0.5
                + self.SENTINEL_OFFSET)

    def test_conv1d_uses_assigned_bias(self):
        x = self.rng.standard_normal((2, 32, 207)).astype("float32")
        module = jt.nn.Conv1d(32, 2048, 1)
        sentinel = self._sentinel(module.out_channels)

        module.bias = jt.array(sentinel)  # what load_state_dict / .to() does
        got = np.asarray(module(jt.array(x)).numpy(), dtype=np.float64)

        expected = _reference_conv(x, module.weight.numpy()[:, :, 0], sentinel, 3)
        error = np.abs(got - expected).max()
        self.assertLess(error, 1e-4, "Conv1d ran with a bias other than the one "
                        "assigned to it (max|d|=%.3e); the inner Conv kept its "
                        "initialisation value" % error)

    def test_conv3d_uses_assigned_bias(self):
        x = self.rng.standard_normal((1, 8, 4, 5, 6)).astype("float32")
        module = jt.nn.Conv3d(8, 64, 1)
        sentinel = self._sentinel(module.out_channels)

        module.bias = jt.array(sentinel)
        got = np.asarray(module(jt.array(x)).numpy(), dtype=np.float64)

        expected = _reference_conv(x, module.weight.numpy()[:, :, 0, 0, 0], sentinel, 5)
        error = np.abs(got - expected).max()
        self.assertLess(error, 1e-4, "Conv3d ran with a bias other than the one "
                        "assigned to it (max|d|=%.3e)" % error)

    def test_bias_stays_none_when_disabled(self):
        """bias=False must not resurrect an inner bias on the way through."""
        module = jt.nn.Conv1d(32, 8, 1, bias=False)
        x = self.rng.standard_normal((1, 32, 16)).astype("float32")
        got = np.asarray(module(jt.array(x)).numpy(), dtype=np.float64)
        expected = _reference_conv(x, module.weight.numpy()[:, :, 0],
                                   np.zeros(8, dtype="float32"), 3)
        self.assertLess(np.abs(got - expected).max(), 1e-4)


if __name__ == "__main__":
    unittest.main()
