"""Binary operators against foreign operands, and complex values without an order.

``Var * obj`` for an object Jittor cannot convert used to raise from the
generated number slot, so Python never tried ``obj.__rmul__`` -- a library
object that knows how to multiply itself with a Var never got the chance. Complex ``maximum``/``minimum`` failed inside the
kernel compiler, and ``jt.norm`` of a complex Var squared it instead of taking
``|x|**2``.
"""

import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase, get_all_device_types, use_cuda_for


class Reflected:
    def __rmul__(self, other):
        return ("rmul", other)

    def __radd__(self, other):
        return ("radd", other)

    def __rsub__(self, other):
        return ("rsub", other)


class TestReflectedOperands(JittorTestCase):
    def test_unknown_operand_defers_to_reflected_method(self):
        x = jt.ones(2)
        other = Reflected()
        self.assertEqual((x * other)[0], "rmul")
        self.assertEqual((x + other)[0], "radd")
        self.assertEqual((x - other)[0], "rsub")
        self.assertIs((x * other)[1], x)

    def test_unknown_operand_without_reflection_is_a_type_error(self):
        with self.assertRaises(TypeError):
            jt.ones(2) * object()
        with self.assertRaises(TypeError):
            object() + jt.ones(2)

    def test_supported_operands_are_unchanged(self):
        x = jt.array(np.array([1.0, 2.0], "float32"))
        np.testing.assert_allclose((x * 2).numpy(), [2.0, 4.0])
        np.testing.assert_allclose((3 * x).numpy(), [3.0, 6.0])
        np.testing.assert_allclose((x + [1.0, 1.0]).numpy(), [2.0, 3.0])


class TestComplexNormAndOrder(JittorTestCase):
    def _devices(self, body):
        for device in get_all_device_types():
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda_for(device)):
                    body()

    def test_complex_norm_is_real_euclidean(self):
        def body():
            z = np.array([[3 + 4j, 1j], [2 - 1j, 0]], np.complex64)
            value = jt.norm(jt.array(z), dim=-1)
            self.assertFalse(value.dtype.is_complex())
            np.testing.assert_allclose(value.numpy(), np.linalg.norm(z, axis=-1), rtol=1e-6)
            np.testing.assert_allclose(jt.array(z).norm(dim=0).numpy(),
                                       np.linalg.norm(z, axis=0), rtol=1e-6)
        self._devices(body)

    def test_complex_norm_gradient(self):
        def body():
            z = jt.array(np.array([3 + 4j, 1 - 2j], np.complex64))
            grad = jt.grad(jt.norm(z, dim=0), z).numpy()
            # d|z| with Jittor's conjugate-Wirtinger convention: z / |z|.
            expected = z.numpy() / np.linalg.norm(z.numpy())
            np.testing.assert_allclose(grad, expected, rtol=1e-5, atol=1e-6)
        self._devices(body)

    def test_complex_maximum_is_a_clear_error(self):
        z = jt.array(np.array([1 + 1j], np.complex64))
        for op in (jt.maximum, jt.minimum):
            with self.assertRaisesRegex(RuntimeError, "not defined for complex"):
                op(z, z)


if __name__ == "__main__":
    unittest.main()
