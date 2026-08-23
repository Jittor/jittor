"""Real deployed flash_attn adapter behavior on Jittor CUDA tensors."""

import importlib.util
import unittest

import numpy as np

import jittor as torch


_HAS_FLASH_ATTN = importlib.util.find_spec("flash_attn") is not None


def _sdpa_reference(query, key, value, scale=None):
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    scores = np.einsum("blhd,bshd->bhls", query, key) * factor
    scores -= scores.max(axis=-1, keepdims=True)
    probability = np.exp(scores)
    probability /= probability.sum(axis=-1, keepdims=True)
    return np.einsum("bhls,bshd->blhd", probability, value)


@unittest.skipUnless(_HAS_FLASH_ATTN, "flash_attn adapter is not installed")
@unittest.skipUnless(torch.compiler.has_cuda, "CUDA is required")
class TestFlashAttnCompat(unittest.TestCase):
    def setUp(self):
        self.random = np.random.RandomState(20260824)

    def _tensor(self, value):
        return torch.tensor(value, device="cuda")

    def test_dense_packed_forward_and_backward(self):
        import flash_attn

        q_array = self.random.randn(2, 3, 2, 4).astype("float32")
        k_array = self.random.randn(2, 3, 2, 4).astype("float32")
        v_array = self.random.randn(2, 3, 2, 4).astype("float32")
        with torch.flag_scope(use_cuda=1):
            q, k, v = (self._tensor(value) for value in (q_array, k_array, v_array))
            output = flash_attn.flash_attn_func(q, k, v)
            self.assertTrue(output.is_cuda)
            gradients = torch.grad(output.sum(), [q, k, v])
            packed = self._tensor(np.stack((q_array, k_array, v_array), axis=2))
            packed_output = flash_attn.flash_attn_qkvpacked_func(packed)
            got_output = output.numpy()
            got_packed = packed_output.numpy()
            got_gradients = [gradient.numpy() for gradient in gradients]

        expected = _sdpa_reference(q_array, k_array, v_array)
        np.testing.assert_allclose(got_output, expected, atol=2e-5, rtol=2e-5)
        np.testing.assert_allclose(got_packed, expected, atol=2e-5, rtol=2e-5)
        for gradient in got_gradients:
            self.assertTrue(np.isfinite(gradient).all())

    def test_varlen_segments_do_not_leak(self):
        import flash_attn

        q_array = self.random.randn(5, 2, 4).astype("float32")
        k_array = self.random.randn(5, 2, 4).astype("float32")
        v_array = self.random.randn(5, 2, 4).astype("float32")
        with torch.flag_scope(use_cuda=1):
            cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32, device="cuda")
            output = flash_attn.flash_attn_varlen_func(
                self._tensor(q_array),
                self._tensor(k_array),
                self._tensor(v_array),
                cu_seqlens,
                cu_seqlens,
                3,
                3,
            )
            self.assertTrue(output.is_cuda)
            got = output.numpy()

        expected = np.concatenate(
            (
                _sdpa_reference(q_array[None, :2], k_array[None, :2], v_array[None, :2])[0],
                _sdpa_reference(q_array[None, 2:], k_array[None, 2:], v_array[None, 2:])[0],
            ),
            axis=0,
        )
        np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)

    def test_math_fallback_honors_dropout_probability(self):
        import flash_attn

        value = np.ones((1, 3, 1, 4), dtype="float32")
        with torch.flag_scope(use_cuda=1):
            output = flash_attn.flash_attn_func(
                self._tensor(value),
                self._tensor(value),
                self._tensor(value),
                dropout_p=1.0,
            )
            got = output.numpy()

        np.testing.assert_array_equal(got, np.zeros_like(value))


if __name__ == "__main__":
    unittest.main()
