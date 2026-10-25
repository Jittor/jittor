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


def _decode_module():
    class SamplerOutput:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Sampler:
        def __init__(self):
            self.compute_nans = False
            self.original_call_count = 0
            self.original_sample_count = 0
            self.sampling_states = types.SimpleNamespace(
                temperature=types.SimpleNamespace(
                    np=np.asarray([0.0, 0.0], dtype="float32")),
                top_k=types.SimpleNamespace(
                    np=np.asarray([3, 3], dtype="int32")),
                top_p=types.SimpleNamespace(
                    np=np.asarray([1.0, 1.0], dtype="float32")),
                vocab_size=3,
                max_num_logprobs=lambda indices: -1,
            )
            self.logprob_token_ids_state = types.SimpleNamespace(
                max_num_token_ids=lambda indices: 0)

        def apply_sampling_params(self, logits, *args, **kwargs):
            return logits

        def sample(self, *args, **kwargs):
            self.original_sample_count += 1
            return "original-sample", None

        def __call__(self, logits, input_batch):
            self.original_call_count += 1
            return "original-call"

    return types.SimpleNamespace(
        Sampler=Sampler, SamplerOutput=SamplerOutput, NO_LOGPROBS=-1)


def _decode_input(prefilling=False, num_draft_tokens=0):
    return types.SimpleNamespace(
        num_draft_tokens=num_draft_tokens,
        is_prefilling_np=np.asarray([prefilling, prefilling]),
        idx_mapping_np=np.asarray([0, 1], dtype="int32"),
        expanded_idx_mapping=None,
        expanded_local_pos=None,
        positions=jt.array([0, 1], dtype="int32"),
        logits_indices=jt.array([0, 1], dtype="int32"),
        input_ids=jt.array([4, 5], dtype="int32"),
        seq_lens=jt.array([8, 9], dtype="int32"),
        num_reqs=2,
    )


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

    def test_decode_reuses_materialized_sample_counts(self):
        module = _decode_module()
        with tx.runtime_hook("decode-counts-" + uuid.uuid4().hex) as hook:
            self.assertTrue(patch_sampler(module))
            sampler = module.Sampler()
            logits = jt.array([[1.0, 3.0, 2.0], [9.0, -1.0, 4.0]])
            first = sampler(logits, _decode_input())
            second = sampler(logits, _decode_input())
            np.testing.assert_array_equal(first.sampled_token_ids.numpy(), [[1], [0]])
            np.testing.assert_array_equal(first.num_sampled.numpy(), [1, 1])
            np.testing.assert_array_equal(first.num_rejected.numpy(), [0, 0])
            self.assertIs(first.num_sampled, second.num_sampled)
            self.assertIs(first.num_rejected, second.num_rejected)
            self.assertEqual(sampler.original_call_count, 0)
            self.assertEqual(sampler.original_sample_count, 0)
        hook.rollback()

    def test_prefill_and_speculative_batches_keep_general_count_kernel(self):
        for input_batch in (_decode_input(prefilling=True),
                            _decode_input(num_draft_tokens=1)):
            module = _decode_module()
            with tx.runtime_hook("decode-fallback-" + uuid.uuid4().hex) as hook:
                self.assertTrue(patch_sampler(module))
                sampler = module.Sampler()
                logits = jt.array([[1.0, 3.0, 2.0], [9.0, -1.0, 4.0]])
                self.assertEqual(sampler(logits, input_batch), "original-call")
                self.assertEqual(sampler.original_call_count, 1)
            hook.rollback()

if __name__ == "__main__":
    unittest.main()
