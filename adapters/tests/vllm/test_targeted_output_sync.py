"""The vLLM output event must follow completion of its sampled-ID copy."""

import os
import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid
import numpy as np

from jittor.compat import transaction as tx
from jittor_adapters.vllm.buffers import (
    patch_model_runner_async_output_fetch,
    patch_model_runner_targeted_output_sync,
    patch_v2_runner_async_output_fetch,
)


class TargetedOutputSyncTests(unittest.TestCase):
    def test_syncs_only_host_copy_before_completed_event(self):
        events = []

        class HostCopy:
            def sync(self):
                events.append('copy.sync')

        host_copy = HostCopy()

        class SampledIds:
            shape = (1, 1)

            def to(self, device, non_blocking=False):
                events.append(('to', device, non_blocking))
                return host_copy

        class Stream:
            def wait_stream(self, other):
                events.append(('wait_stream', other))

        class Output:
            def __init__(self, *args, **kwargs):
                events.append('original')

        fake_torch = ModuleType('torch')
        fake_torch.cuda = SimpleNamespace(
            current_stream=lambda: 'main', stream=lambda stream: nullcontext())
        fake_jittor = ModuleType('jittor')
        fake_jittor.sync_all = lambda device_sync: events.append(
            ('sync_all', device_sync))
        with tx.runtime_hook('targeted-output-' + uuid.uuid4().hex) as hook:
            with patch.dict(os.environ, {
                'JITTOR_VLLM_EXPERIMENTAL_TARGETED_OUTPUT_SYNC': '1',
            }), patch.dict(sys.modules, {
                'torch': fake_torch, 'jittor': fake_jittor}):
                module = SimpleNamespace(AsyncGPUModelRunnerOutput=Output)
                self.assertTrue(patch_model_runner_targeted_output_sync(module))
                self.assertFalse(patch_model_runner_targeted_output_sync(module))
                result = Output(object(), SampledIds(), None, [], Stream(), 100)
                self.assertIs(result.sampled_token_ids_cpu, host_copy)
                self.assertEqual(events, [
                    ('wait_stream', 'main'), ('to', 'cpu', True),
                    ('sync_all', False), 'copy.sync'])
                result.async_copy_ready_event.synchronize()
                self.assertTrue(result.async_copy_ready_event.query())
                self.assertEqual(len(events), 4)
        hook.rollback()

    def test_unusual_outputs_use_original_constructor(self):
        calls = []

        class Output:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

        class SampledIds:
            shape = (1, 2)

        with tx.runtime_hook('targeted-fallback-' + uuid.uuid4().hex) as hook:
            with patch.dict(os.environ, {
                'JITTOR_VLLM_EXPERIMENTAL_TARGETED_OUTPUT_SYNC': '1',
            }):
                self.assertTrue(patch_model_runner_targeted_output_sync(
                    SimpleNamespace(AsyncGPUModelRunnerOutput=Output)))
                Output(object(), SampledIds(), None, [], object(), 100)
            self.assertEqual(len(calls), 1)
        hook.rollback()

    def test_default_off(self):
        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_TARGETED_OUTPUT_SYNC': '0',
        }):
            self.assertFalse(patch_model_runner_targeted_output_sync(
                SimpleNamespace(AsyncGPUModelRunnerOutput=object)))


class LegacyAsyncFetchTests(unittest.TestCase):
    def test_fetch_owns_host_bytes_and_waits_at_output_boundary(self):
        events = []
        returned_bytes = np.array([[42]], dtype=np.int64)

        class SampledIds:
            shape = (1, 1)

        class Output:
            def __init__(self, *args, **kwargs):
                events.append('original')

        fake_jittor = ModuleType('jittor')

        def fetch(value, callback):
            events.append('fetch')
            callback(returned_bytes)

        fake_jittor.fetch = fetch
        fake_jittor.sync_all = lambda wait: events.append(('sync_all', wait))
        with tx.runtime_hook('legacy-fetch-' + uuid.uuid4().hex) as hook:
            with patch.dict(os.environ, {
                'JITTOR_VLLM_EXPERIMENTAL_LEGACY_ASYNC_FETCH': '1',
            }), patch.dict(sys.modules, {'jittor': fake_jittor}), patch(
                'jittor_adapters.vllm.buffers._native_cuda_active',
                return_value=True,
            ):
                module = SimpleNamespace(AsyncGPUModelRunnerOutput=Output)
                self.assertTrue(patch_model_runner_async_output_fetch(module))
                result = Output(object(), SampledIds(), None, [], object(), 100)
                returned_bytes[0, 0] = -1
                result.async_copy_ready_event.synchronize()
                self.assertEqual(result.sampled_token_ids_cpu.tolist(), [[42]])
                self.assertEqual(events, ['fetch', ('sync_all', False)])
        hook.rollback()

    def test_fallback_and_default_off(self):
        calls = []

        class Output:
            def __init__(self, *args, **kwargs):
                calls.append((args, kwargs))

        class SampledIds:
            shape = (1, 2)

        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_LEGACY_ASYNC_FETCH': '0',
        }):
            self.assertFalse(patch_model_runner_async_output_fetch(
                SimpleNamespace(AsyncGPUModelRunnerOutput=Output)))
        with tx.runtime_hook('legacy-fallback-' + uuid.uuid4().hex) as hook:
            with patch.dict(os.environ, {
                'JITTOR_VLLM_EXPERIMENTAL_LEGACY_ASYNC_FETCH': '1',
            }), patch.dict(sys.modules, {'jittor': ModuleType('jittor')}), patch(
                'jittor_adapters.vllm.buffers._native_cuda_active',
                return_value=True,
            ):
                self.assertTrue(patch_model_runner_async_output_fetch(
                    SimpleNamespace(AsyncGPUModelRunnerOutput=Output)))
                Output(object(), SampledIds(), None, [], object(), 100)
                self.assertEqual(len(calls), 1)
        hook.rollback()


class V2AsyncFetchTests(unittest.TestCase):
    def test_fetches_tokens_and_counts_with_one_graph_submission(self):
        events = []
        token_bytes = np.array([[42]], dtype=np.int64)
        count_bytes = np.array([1], dtype=np.int32)

        class SampledIds:
            shape = (1, 1)

        sampled_ids = SampledIds()
        counts = object()
        sampler = SimpleNamespace(sampled_token_ids=sampled_ids,
                                  logprobs_tensors=None, num_nans=None)
        output = SimpleNamespace(prompt_logprobs_dict={})

        class AsyncOutput:
            def __init__(self, *args):
                events.append('original')

        fake_jittor = ModuleType('jittor')

        def fetch(value, callback):
            events.append('fetch')
            callback(token_bytes if value is sampled_ids else count_bytes)

        fake_jittor.fetch = fetch
        fake_jittor.sync_all = lambda wait: events.append(('sync_all', wait))
        with tx.runtime_hook('v2-fetch-' + uuid.uuid4().hex) as hook:
            with patch.dict(os.environ, {
                'JITTOR_VLLM_EXPERIMENTAL_V2_ASYNC_FETCH': '1',
            }), patch.dict(sys.modules, {'jittor': fake_jittor}), patch(
                'jittor_adapters.vllm.buffers._native_cuda_active',
                return_value=True,
            ):
                self.assertTrue(patch_v2_runner_async_output_fetch(
                    SimpleNamespace(AsyncOutput=AsyncOutput)))
                result = AsyncOutput(output, sampler, counts, object(), object())
                token_bytes[0, 0] = -1
                count_bytes[0] = -1
                result.copy_event.synchronize()
                self.assertEqual(result.sampled_token_ids.tolist(), [[42]])
                self.assertEqual(result.num_sampled_tokens_np.tolist(), [1])
                self.assertEqual(events, ['fetch', 'fetch', ('sync_all', False)])
        hook.rollback()

    def test_fallback_for_speculative_shape(self):
        events = []

        class AsyncOutput:
            def __init__(self, *args):
                events.append('original')

        sampler = SimpleNamespace(
            sampled_token_ids=SimpleNamespace(shape=(1, 2)),
            logprobs_tensors=None, num_nans=None)
        output = SimpleNamespace(prompt_logprobs_dict={})
        with tx.runtime_hook('v2-fallback-' + uuid.uuid4().hex) as hook:
            with patch.dict(os.environ, {
                'JITTOR_VLLM_EXPERIMENTAL_V2_ASYNC_FETCH': '1',
            }), patch.dict(sys.modules, {'jittor': ModuleType('jittor')}), patch(
                'jittor_adapters.vllm.buffers._native_cuda_active',
                return_value=True,
            ):
                self.assertTrue(patch_v2_runner_async_output_fetch(
                    SimpleNamespace(AsyncOutput=AsyncOutput)))
                AsyncOutput(output, sampler, object(), object(), object())
                self.assertEqual(events, ['original'])
        hook.rollback()


if __name__ == '__main__':
    unittest.main()
