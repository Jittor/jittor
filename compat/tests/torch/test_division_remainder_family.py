"""`fmod`, `remainder`, `trunc`, `floor` and `floor_divide` against NumPy.

The family had no coverage at all, and one member was broken for every input:
`Tensor.fmod` was implemented as ``self - jt.trunc(self / other) * other`` but
`jt.trunc` does not exist -- `trunc` is installed onto the *Torch* namespace by
`core.install_misc`, never onto the Jittor one -- so every call raised
`AttributeError: trunc`. The implementation named a function in the wrong
namespace and nothing executed it until a user did.

The sign conventions are the point of these assertions. `fmod` truncates toward
zero and takes the sign of the *dividend*; `remainder` floors and takes the sign
of the *divisor*. The two agree whenever both operands are positive, so a test
that only used positive inputs would pass with the two swapped. Every case here
includes negative dividends and negative divisors for that reason.
"""

import unittest

import numpy as np
import pytest
import torch


def _cuda_available():
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


#: Chosen so truncation and flooring disagree: negative dividends, negative
#: divisors, an exact multiple, and a value below one in magnitude.
_DIVIDENDS = np.array([-5.0, -2.7, -0.5, 0.5, 1.5, 2.7, 6.0], dtype="float32")
_DIVISORS = (2.0, -2.0, 3.0, 0.75)


class _FamilyChecks(object):
    """Assertions shared by the CPU and CUDA classes; not collected itself."""

    device = "cpu"

    def _tensor(self, array):
        tensor = torch.tensor(array)
        return tensor.cuda() if self.device == "cuda" else tensor

    def _check(self, got, want, label):
        got = np.asarray(got.cpu().numpy(), dtype="float64")
        want = np.asarray(want, dtype="float64")
        self.assertEqual(got.shape, want.shape, label)
        scale = max(float(np.abs(want).max()), 1.0)
        error = float(np.abs(got - want).max() / scale)
        self.assertLess(error, 1e-6, "{0}: {1:.3e}\ngot  {2}\nwant {3}".format(
            label, error, got, want))

    def test_fmod_takes_the_sign_of_the_dividend(self):
        tensor = self._tensor(_DIVIDENDS)
        reference = _DIVIDENDS.astype("float64")
        for divisor in _DIVISORS:
            self._check(tensor.fmod(divisor), np.fmod(reference, divisor),
                        "fmod({0})".format(divisor))

    def test_remainder_takes_the_sign_of_the_divisor(self):
        tensor = self._tensor(_DIVIDENDS)
        reference = _DIVIDENDS.astype("float64")
        for divisor in _DIVISORS:
            self._check(tensor.remainder(divisor), np.mod(reference, divisor),
                        "remainder({0})".format(divisor))

    def test_fmod_and_remainder_disagree_on_mixed_signs(self):
        """The guard against implementing one as the other."""
        tensor = self._tensor(_DIVIDENDS)
        fmod = np.asarray(tensor.fmod(-2.0).cpu().numpy(), dtype="float64")
        remainder = np.asarray(tensor.remainder(-2.0).cpu().numpy(), dtype="float64")
        self.assertFalse(
            np.allclose(fmod, remainder),
            "fmod and remainder returned the same values for mixed signs, "
            "so one of them is implemented as the other",
        )

    def test_fmod_accepts_a_tensor_divisor(self):
        dividend = self._tensor(np.array([[7.0, -7.0], [3.5, -3.5]], dtype="float32"))
        divisor = self._tensor(np.array([[3.0, 3.0], [2.0, 2.0]], dtype="float32"))
        want = np.fmod(np.array([[7.0, -7.0], [3.5, -3.5]]),
                       np.array([[3.0, 3.0], [2.0, 2.0]]))
        self._check(dividend.fmod(divisor), want, "fmod(tensor)")

    def test_trunc_and_floor_differ_below_zero(self):
        tensor = self._tensor(_DIVIDENDS)
        reference = _DIVIDENDS.astype("float64")
        self._check(tensor.trunc(), np.trunc(reference), "trunc")
        self._check(tensor.floor(), np.floor(reference), "floor")
        self._check(torch.trunc(tensor), np.trunc(reference), "torch.trunc")

    def test_integer_floor_divide_matches_numpy(self):
        """The integer path, whose flooring fix KI-OPS-002 records as verified."""
        values = np.array([-5, -4, -3, -1, 1, 5, 7], dtype="int64")
        tensor = self._tensor(values)
        for divisor in (2, -2, 3):
            self._check(torch.floor_divide(tensor, divisor),
                        np.floor_divide(values, divisor),
                        "int floor_divide({0})".format(divisor))

    def test_div_rounding_modes_accept_tensors_and_python_scalars(self):
        values = np.array([-5, -4, 3, 6], dtype="int64")
        tensor = self._tensor(values)
        truncated = torch.div(tensor, 2, rounding_mode="trunc")
        floored = torch.divide(tensor, 2, rounding_mode="floor")

        self.assertEqual(str(truncated.dtype), "torch.int64")
        self.assertEqual(str(floored.dtype), "torch.int64")
        self._check(truncated, np.trunc(values / 2).astype("int64"),
                    "div rounding_mode=trunc")
        self._check(floored, np.floor(values / 2).astype("int64"),
                    "divide rounding_mode=floor")

        scalar = torch.div(8, 2, rounding_mode="trunc")
        self.assertEqual(scalar.ndim, 0)
        self.assertEqual(int(scalar), 4)

    @pytest.mark.xfail(strict=True, reason="KI-OPS-003: float operands are truncated to integers")
    def test_float_floor_divide_matches_numpy(self):
        """Float operands are cast to integers before dividing -- see KI-OPS-003.

        Strict, so that fixing the operator turns this red and the entry gets
        retired rather than the expectation quietly outliving the defect.
        """
        tensor = self._tensor(_DIVIDENDS)
        reference = _DIVIDENDS.astype("float64")
        for divisor in _DIVISORS:
            self._check(torch.floor_divide(tensor, divisor),
                        np.floor_divide(reference, divisor),
                        "float floor_divide({0})".format(divisor))


    def test_nansum_and_nanmean_ignore_nan(self):
        """`jt.nan_to_num` does not exist; both used to raise AttributeError."""
        values = np.array([[1.0, np.nan, 3.0], [np.nan, np.nan, 4.0]], dtype="float32")
        tensor = self._tensor(values)
        self.assertAlmostEqual(float(tensor.nansum().item()),
                               float(np.nansum(values)), places=5)
        self.assertAlmostEqual(float(tensor.nanmean().item()),
                               float(np.nanmean(values)), places=5)
        self.assertAlmostEqual(float(torch.nansum(tensor).item()),
                               float(np.nansum(values)), places=5)
        self.assertAlmostEqual(float(torch.nanmean(tensor).item()),
                               float(np.nanmean(values)), places=5)
        self._check(tensor.nansum(1), np.nansum(values, 1), "nansum(dim=1)")
        self._check(tensor.nanmean(1), np.nanmean(values, 1), "nanmean(dim=1)")


    def test_reducing_a_scalar_tensor(self):
        """Reducing over no dimensions returns the value, as PyTorch does.

        This was a strict expected failure until 2026-09-10 (KI-OPS-004). The
        reduce kernel emitted `index_t ystride-1 = 1;` for a rank-0 input --
        `@{DIM-1}` with `DIM` zero -- so the generated source did not compile
        and `loss.sum()` on an already-scalar loss died on both devices.

        The shape is asserted too. Returning `3.0` with shape `(1,)` would
        satisfy the values above and still break every caller that reduces
        without checking rank, which is the code this exists for.
        """
        scalar = self._tensor(np.float32(3.0))
        self.assertEqual(scalar.ndim, 0)
        for name in ("sum", "mean", "max", "min"):
            with self.subTest(reduction=name):
                result = getattr(scalar, name)()
                self.assertAlmostEqual(float(result.item()), 3.0, places=5)
                self.assertEqual(tuple(result.shape), (),
                                 "%s of a rank-0 tensor should stay rank-0" % name)


class TestDivisionRemainderFamilyCPU(_FamilyChecks, unittest.TestCase):
    device = "cpu"


@unittest.skipUnless(_cuda_available(), "cuda is required for the device half")
class TestDivisionRemainderFamilyCUDA(_FamilyChecks, unittest.TestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
