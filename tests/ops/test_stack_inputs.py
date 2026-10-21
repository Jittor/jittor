"""``jt.stack`` takes plain Var inputs without cloning them first.

Two or more inputs are copied into a new result anyway, so plain Vars are
read as they are; a lone input still goes through ``jt.array`` as before. The
change must not be visible: values, gradients, and -- for the inputs that are
no longer cloned -- independence of the result from them.
"""

from _helpers import capability as _test_capability

import unittest

import numpy as np

import jittor as jt


class _StackInputs:
    use_cuda = 0

    def setUp(self):
        self._scope = jt.flag_scope(use_cuda=self.use_cuda)
        self._scope.__enter__()

    def tearDown(self):
        self._scope.__exit__(None, None, None)

    def test_values_and_gradients(self):
        rng = np.random.RandomState(0)
        arrays = [rng.rand(3, 4).astype("float32") for _ in range(4)]
        for n in (1, 2, 3, 4):
            for dim in (0, 1, -1):
                with self.subTest(n=n, dim=dim):
                    xs = [jt.array(a) for a in arrays[:n]]
                    out = jt.stack(xs, dim)
                    np.testing.assert_array_equal(out.numpy(), np.stack(arrays[:n], dim))
                    grads = jt.grad((out * jt.arange(out.numel()).reshape(out.shape)).sum(), xs)
                    weights = np.arange(out.numel(), dtype="float32").reshape(out.shape)
                    for i, g in enumerate(grads):
                        np.testing.assert_array_equal(g.numpy(), np.take(weights, i, axis=dim))

    def test_the_result_is_independent_of_its_inputs(self):
        # A lone input is a clone + unsqueeze, whose storage sharing is
        # clone's contract and is not changed here.
        for n in (2, 3):
            with self.subTest(n=n):
                xs = [jt.array(np.full((2, 3), i, "float32")) for i in range(n)]
                out = jt.stack(xs)
                out.sync()
                for x in xs:
                    x.assign(x * 0 - 5)
                want = np.stack([np.full((2, 3), i, "float32") for i in range(n)])
                np.testing.assert_array_equal(out.numpy(), want)
                out.assign(out * 0 + 9)
                for x in xs:
                    np.testing.assert_array_equal(x.numpy(), np.full((2, 3), -5, "float32"))

    def test_one_dimensional_views(self):
        # Two or three 1-D inputs go to the stack kernel as they are, without
        # the reshape every other rank gets; views of a larger tensor must
        # still be read element by element.
        base = np.arange(40, dtype="float32").reshape(5, 8)
        b = jt.array(base)
        views = {
            "strided": (lambda: b.reshape([-1])[::3], base.reshape(-1)[::3]),
            "row": (lambda: b[2], base[2]),
            "column": (lambda: b[:, 3], base[:, 3]),
            "offset": (lambda: b.reshape([-1])[5:13], base.reshape(-1)[5:13]),
            "transposed": (lambda: b.transpose()[1], base.T[1]),
        }
        for name, (make, ref) in views.items():
            for n in (2, 3):
                for dim in (0, 1, -1):
                    with self.subTest(view=name, n=n, dim=dim), jt.no_grad():
                        x = make()
                        got = jt.stack([x, x * 2, x][:n], dim).numpy()
                        np.testing.assert_array_equal(got, np.stack([ref, ref * 2, ref][:n], dim))

    def test_mixed_inputs(self):
        out = jt.stack([jt.array([1.0, 2.0]), np.array([3.0, 4.0], "float32"), [5.0, 6.0]])
        np.testing.assert_array_equal(out.numpy(), np.array([[1, 2], [3, 4], [5, 6]], "float32"))


class TestStackInputsCPU(_StackInputs, unittest.TestCase):
    use_cuda = 0


@unittest.skipIf(not _test_capability.check_accelerator('cuda', backend=jt).enabled, "No CUDA found")
class TestStackInputsCUDA(_StackInputs, unittest.TestCase):
    use_cuda = 1


if __name__ == "__main__":
    unittest.main()
