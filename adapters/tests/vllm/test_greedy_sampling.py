"""Tests for the exact all-greedy sampler fast path."""

import types
import unittest
import uuid

import numpy as np

import jittor as jt
from jittor.compat import transaction as tx

from jittor_adapters.vllm.buffers import patch_sampler


def _module(temperature=(0.0, 0.0), top_k=(7, 7), top_p=(1.0, 1.0)):
    class Sampler:
        def __init__(self):
            self.original_calls = 0
            self.processor_calls = 0
            self.sampling_states = types.SimpleNamespace(
                temperature=types.SimpleNamespace(
                    np=np.asarray(temperature, dtype="float32")),
                top_k=types.SimpleNamespace(np=np.asarray(top_k, dtype="int32")),
                top_p=types.SimpleNamespace(np=np.asarray(top_p, dtype="float32")),
                vocab_size=7,
            )

        def apply_sampling_params(self, logits, *args, **kwargs):
            assert kwargs == {"skip_top_k_top_p": True}
            self.processor_calls += 1
            return logits + 0.25

        def sample(self, *args, **kwargs):
            self.original_calls += 1
            return "original", None

    return types.SimpleNamespace(Sampler=Sampler)


def _call(instance):
    logits = jt.array([[1.0, 3.0, 2.0], [9.0, -1.0, 4.0]])
    indices = np.asarray([0, 1], dtype="int32")
    return instance.sample(logits, None, indices, None, None, None)


class TestGreedySamplingPatch(unittest.TestCase):
    def test_all_greedy_without_filters_uses_processed_argmax(self):
        module = _module()
        with tx.runtime_hook("greedy-sampler-" + uuid.uuid4().hex) as hook:
            self.assertTrue(patch_sampler(module))
            self.assertFalse(patch_sampler(module))
            sampler = module.Sampler()
            sampled, processed = _call(sampler)
            np.testing.assert_array_equal(sampled.numpy(), [1, 0])
            np.testing.assert_allclose(
                processed.numpy(), [[1.25, 3.25, 2.25], [9.25, -0.75, 4.25]])
            self.assertEqual(str(sampled.dtype), "int64")
            self.assertEqual(sampler.processor_calls, 1)
            self.assertEqual(sampler.original_calls, 0)
        hook.rollback()

    def test_random_or_filtered_batches_keep_vllm_sampler(self):
        cases = (
            _module(temperature=(0.0, 0.8)),
            _module(top_k=(7, 3)),
            _module(top_p=(1.0, 0.9)),
        )
        for module in cases:
            with tx.runtime_hook("greedy-sampler-" + uuid.uuid4().hex) as hook:
                self.assertTrue(patch_sampler(module))
                sampler = module.Sampler()
                sampled, processed = _call(sampler)
                self.assertEqual(sampled, "original")
                self.assertIsNone(processed)
                self.assertEqual(sampler.original_calls, 1)
                self.assertEqual(sampler.processor_calls, 0)
            hook.rollback()


if __name__ == "__main__":
    unittest.main()
