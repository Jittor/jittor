"""A transpose's permutation is a runtime value, not part of its kernel.

The permutation used to be written into the JIT key of both transpose ops, so
every new permutation of a high-rank tensor compiled a kernel of its own (about
730 ms of g++ per permutation, against a 400 us computation, on a 6-qubit
density-matrix circuit). Correctness is checked for both forms of the fused
transpose -- the pure permute (loops over the output) and the form fused after
an elementwise producer (loops over the input) -- and for the gradient.
"""

import unittest

import numpy as np
import jittor as jt

from _helpers.common import JittorTestCase, get_all_device_types, use_cuda_for


RANK = 9


def kernels_compiled_by(fn):
    """How many ops this process had to compile while running ``fn``.

    Counted from the JIT compiler's own "Compile op" record rather than from
    the ``*.so`` files in the cache: a gate shares one cache between its
    workers, so another worker's compiles used to land in that directory
    during the measured window and fail this file.
    """
    with jt.log_capture_scope(log_silent=1, log_v=0,
                              log_vprefix="jit_compiler=100") as logs:
        fn()
    return sum(1 for record in logs if record["msg"].startswith("Compile op"))


class TestTransposeRuntimePermutation(JittorTestCase):
    def _devices(self, body):
        for device in get_all_device_types():
            with self.subTest(device=device):
                with jt.flag_scope(use_cuda=use_cuda_for(device)):
                    body()

    def _run(self, a, perm, perm_last):
        ja = jt.array(a)
        np.testing.assert_array_equal(ja.permute(*perm).numpy(), a.transpose(perm))
        # last axis kept: fuse_transpose as a pure permute, and fused after an
        # elementwise producer
        np.testing.assert_allclose((ja.permute(*perm_last) * 2).numpy(),
                                   a.transpose(perm_last) * 2)
        np.testing.assert_allclose((ja * 2).permute(*perm_last).numpy(),
                                   (a * 2).transpose(perm_last))
        weight = np.arange(a.size, dtype=a.dtype).reshape(a.transpose(perm).shape)
        grad = jt.grad((ja.permute(*perm) * jt.array(weight)).sum(), ja).numpy()
        np.testing.assert_allclose(grad, weight.transpose(np.argsort(perm)))

    def test_new_permutations_compile_nothing_after_warmup(self):
        def body():
            rng = np.random.default_rng(0)
            a = rng.standard_normal([2, 3] + [2] * (RANK - 2)).astype("float32")
            def perms():
                perm = [int(p) for p in rng.permutation(RANK)]
                perm_last = [int(p) for p in rng.permutation(RANK - 1)] + [RANK - 1]
                return perm, perm_last
            for _ in range(2):
                self._run(a, *perms())

            def more():
                for _ in range(4):
                    self._run(a, *perms())
            self.assertEqual(kernels_compiled_by(more), 0)
        self._devices(body)

    def test_non_square_shapes_and_complex(self):
        def body():
            rng = np.random.default_rng(1)
            a = (rng.standard_normal([2, 3, 4, 5]) + 1j * rng.standard_normal([2, 3, 4, 5]))
            a = a.astype("complex64")
            for perm in ([3, 1, 0, 2], [1, 0, 2, 3], [2, 3, 1, 0]):
                ja = jt.array(a)
                np.testing.assert_array_equal(ja.permute(*perm).numpy(), a.transpose(perm))
                np.testing.assert_allclose((ja * 2).permute(*perm).numpy(),
                                           (a * 2).transpose(perm), rtol=1e-6)
        self._devices(body)


if __name__ == "__main__":
    unittest.main()
