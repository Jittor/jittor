import unittest

import jittor as jt

from _helpers.gradcheck import GradcheckError, gradcheck


class _WrongSquare(jt.Function):
    """Square with an intentionally incorrect derivative for oracle testing."""

    def execute(self, x):
        self.x = x
        return x * x

    def grad(self, grad_output):
        return grad_output * 3.0 * self.x


class TestGradcheckOracle(unittest.TestCase):
    @jt.flag_scope(use_cuda=0)
    def test_intentionally_wrong_derivative_fails(self):
        x = jt.array([0.5, 1.5], dtype="float64")

        with self.assertRaisesRegex(GradcheckError, "Jacobian mismatch"):
            gradcheck(_WrongSquare.apply, x)


if __name__ == "__main__":
    unittest.main()
