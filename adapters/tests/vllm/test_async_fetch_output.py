"""Opt-in vLLM decode readback must use the callback-owned host snapshot."""

import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

import numpy as np

from jittor.compat import transaction as tx
from jittor_adapters.vllm import buffers
from jittor_adapters.vllm.buffers import patch_async_copy_to_np


class AsyncFetchOutputTests(unittest.TestCase):
    def test_h2d_batch_owns_mutable_numpy_source_until_flush(self):
        copied = []
        syncs = []
        fake_jittor = ModuleType('jittor')
        fake_jittor.flags = SimpleNamespace(use_acl=0)
        fake_jittor.compiler = SimpleNamespace(has_acl=0)
        fake_torch = ModuleType('torch')
        fake_torch.Tensor = type('Tensor', (), {})
        fake_torch.tensor = lambda source, **kwargs: copied.append(source) or object()
        fake_torch.cuda = SimpleNamespace(synchronize=lambda: syncs.append(True))

        class UvaBufferPool:
            def __init__(self, size, dtype, max_concurrency=None):
                self.dtype = dtype

            def copy_to_uva(self, x):
                raise AssertionError('the patch should replace this method')

        module = SimpleNamespace(UvaBufferPool=UvaBufferPool)
        with patch.dict(sys.modules, {'jittor': fake_jittor, 'torch': fake_torch}):
            with tx.runtime_hook('batch-h2d-source-' + uuid.uuid4().hex) as hook:
                self.assertTrue(buffers.patch_buffer_pool(module))
                pool = UvaBufferPool(1, 'int32')
                source = np.array([3], dtype=np.int32)
                buffers._H2D_SYNC_BATCH.sources = []
                try:
                    pool.copy_to_uva(source)
                    source[0] = 99
                    self.assertEqual(copied[0].tolist(), [3])
                    self.assertIs(copied[0], buffers._H2D_SYNC_BATCH.sources[0])
                    self.assertEqual(syncs, [])
                finally:
                    buffers._H2D_SYNC_BATCH.sources = None
                pool.copy_to_uva(source)
                self.assertEqual(syncs, [True])
            hook.rollback()

    def test_h2d_batch_syncs_once_at_input_boundary_and_cleans_up(self):
        syncs = []
        fake_jittor = ModuleType('jittor')
        fake_jittor.flags = SimpleNamespace(use_cuda=1, use_acl=0)
        fake_jittor.compiler = SimpleNamespace(has_cuda=1, has_acl=0)
        fake_jittor.sync_all = lambda device_sync=False: syncs.append(device_sync)

        class GPUModelRunner:
            def execute_model(self, scheduler_output, *args, **kwargs):
                pending = getattr(buffers._H2D_SYNC_BATCH, 'sources', None)
                if pending is not None:
                    pending.append(object())
                return self.prepare_inputs(scheduler_output, None)

            def prepare_inputs(self, scheduler_output, batch_desc):
                return 17

        module = SimpleNamespace(GPUModelRunner=GPUModelRunner)
        stable = SimpleNamespace(num_scheduled_tokens={'req': 1},
                                 scheduled_new_reqs=[], finished_req_ids=[])
        changed = SimpleNamespace(num_scheduled_tokens={'req': 1},
                                  scheduled_new_reqs=['req'], finished_req_ids=[])
        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_BATCH_H2D_SYNC': '1',
            'JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH': '1',
        }), patch.dict(sys.modules, {'jittor': fake_jittor}):
            with tx.runtime_hook('batch-h2d-' + uuid.uuid4().hex) as hook:
                self.assertTrue(buffers.patch_model_runner_h2d_batch(module))
                runner = GPUModelRunner()
                self.assertEqual(runner.execute_model(stable), 17)
                self.assertEqual(syncs, [True])
                self.assertIsNone(getattr(buffers._H2D_SYNC_BATCH,
                                          'sources', None))
                self.assertEqual(runner.execute_model(changed), 17)
                self.assertEqual(syncs, [True])
                self.assertEqual(runner.execute_model(stable, dummy_run=True), 17)
                self.assertEqual(syncs, [True])
            hook.rollback()

    def test_async_fetch_rejects_unsynchronized_h2d_source(self):
        class AsyncOutput:
            pass

        original_init = AsyncOutput.__init__
        module = SimpleNamespace(AsyncOutput=AsyncOutput,
                                 async_copy_to_np=lambda value: value.numpy())
        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH': '1',
            'JITTOR_VLLM_UVA_COPY_SYNC': '0',
        }):
            with self.assertRaisesRegex(RuntimeError, 'requires synchronized'):
                patch_async_copy_to_np(module)
        self.assertIs(AsyncOutput.__init__, original_init)

    def test_disabled_path_uses_synchronous_numpy(self):
        fake_jittor = ModuleType('jittor')
        fake_jittor.fetch = lambda *args: self.fail('fetch must stay disabled')

        class AsyncOutput:
            def __init__(self, *args):
                raise AssertionError('static-count decode should use its fast path')

        class Tensor:
            def numpy(self):
                return np.array([[23]], dtype=np.int64)

        module = SimpleNamespace(
            AsyncOutput=AsyncOutput,
            async_copy_to_np=lambda value: value.numpy(),
        )
        counts = SimpleNamespace(_jittor_static_numpy=np.ones(1, dtype=np.int32))
        sampler = SimpleNamespace(sampled_token_ids=Tensor(),
                                  logprobs_tensors=None, num_nans=None)
        model = SimpleNamespace(prompt_logprobs_dict={})
        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH': '0',
        }), patch.dict(sys.modules, {'jittor': fake_jittor}):
            with tx.runtime_hook('async-fetch-disabled-' + uuid.uuid4().hex) as hook:
                self.assertTrue(patch_async_copy_to_np(module))
                output = module.AsyncOutput(model, sampler, counts, None, None)
                self.assertEqual(output.sampled_token_ids.tolist(), [[23]])
            hook.rollback()

    def test_prompt_logprobs_keep_synchronous_readback(self):
        fake_jittor = ModuleType('jittor')
        fake_jittor.fetch = lambda *args: self.fail('fetch must not run')

        class AsyncOutput:
            def __init__(self, *args):
                raise AssertionError('static-count decode should use its fast path')

        class Tensor:
            def numpy(self):
                return np.array([[31]], dtype=np.int64)

        module = SimpleNamespace(AsyncOutput=AsyncOutput,
                                 async_copy_to_np=lambda value: value.numpy())
        counts = SimpleNamespace(_jittor_static_numpy=np.ones(1, dtype=np.int32))
        sampler = SimpleNamespace(sampled_token_ids=Tensor(),
                                  logprobs_tensors=None, num_nans=None)
        model = SimpleNamespace(prompt_logprobs_dict={'req': None})
        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH': '1',
        }), patch.dict(sys.modules, {'jittor': fake_jittor}):
            with tx.runtime_hook('async-fetch-logprob-' + uuid.uuid4().hex) as hook:
                self.assertTrue(patch_async_copy_to_np(module))
                output = module.AsyncOutput(model, sampler, counts, None, None)
                self.assertEqual(output.sampled_token_ids.tolist(), [[31]])
            hook.rollback()

    def test_callback_copies_bytes_before_fetch_storage_is_released(self):
        pending = []
        submitted = []
        fake_jittor = ModuleType('jittor')
        fake_jittor.fetch = lambda value, callback: pending.append((value, callback))
        fake_jittor.sync_all = lambda device_sync=False: submitted.append(device_sync)

        class AsyncOutput:
            def __init__(self, *args):
                raise AssertionError('the ordinary readback should not run')

        original_init = AsyncOutput.__init__

        module = SimpleNamespace(
            AsyncOutput=AsyncOutput,
            async_copy_to_np=lambda value: value.numpy(),
        )
        token = SimpleNamespace()
        counts = SimpleNamespace(_jittor_static_numpy=np.ones(1, dtype=np.int32))
        sampler = SimpleNamespace(sampled_token_ids=token,
                                  logprobs_tensors=None, num_nans=None)
        model = SimpleNamespace(prompt_logprobs_dict={})

        with patch.dict(os.environ, {
            'JITTOR_VLLM_EXPERIMENTAL_ASYNC_FETCH': '1',
        }), patch.dict(sys.modules, {'jittor': fake_jittor}):
            with tx.runtime_hook('async-fetch-output-' + uuid.uuid4().hex) as hook:
                self.assertTrue(patch_async_copy_to_np(module))
                output = module.AsyncOutput(model, sampler, counts, None, None)
                self.assertEqual(submitted, [False])
                self.assertEqual(len(pending), 1)
                self.assertIs(pending[0][0], token)
                staged = np.array([[17]], dtype=np.int64)
                pending[0][1](staged)
                staged[0, 0] = 99
                output.copy_event.synchronize()
                self.assertEqual(output.sampled_token_ids.tolist(), [[17]])
                self.assertEqual(output.num_sampled_tokens_np.tolist(), [1])

                class BadArray:
                    def __array__(self, *args, **kwargs):
                        raise ValueError('host snapshot failed')

                failed = module.AsyncOutput(model, sampler, counts, None, None)
                pending[1][1](BadArray())
                with self.assertRaisesRegex(RuntimeError, 'token fetch failed'):
                    failed.sampled_token_ids.tolist()
            hook.rollback()

        self.assertIs(module.AsyncOutput.__init__, original_init)


if __name__ == '__main__':
    unittest.main()
