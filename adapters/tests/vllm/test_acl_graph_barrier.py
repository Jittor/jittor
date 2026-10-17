"""The ACL paged-decode graph guard must fail closed only while tracing."""

import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from jittor_adapters.vllm import flash_attn


class _StopBeforeOperators(Exception):
    pass


class TestAclGraphBarrier(unittest.TestCase):
    def _barrier_calls(self, *, acl, tracing, tokens):
        graph_runtime = importlib.import_module("jittor._runtime.graph_replay")

        class FakeAttentionImpl:
            pass

        module = SimpleNamespace(FlashAttentionImpl=FakeAttentionImpl)
        fake_jt = SimpleNamespace(flags=SimpleNamespace(use_acl=acl))
        metadata = SimpleNamespace(num_actual_tokens=tokens)
        with patch.object(flash_attn, "jt", fake_jt), \
                patch.object(flash_attn, "set_attr", setattr), \
                patch.object(flash_attn, "_cuda_graph_capture_active",
                             side_effect=_StopBeforeOperators), \
                patch.object(graph_runtime, "_TRACING", [int(tracing)]), \
                patch.object(graph_runtime, "graph_replay_barrier") as barrier:
            self.assertTrue(flash_attn.patch_attention_impl(module))
            with self.assertRaises(_StopBeforeOperators):
                FakeAttentionImpl().forward(
                    None, None, None, None, None, metadata, None)
            return barrier.call_args_list

    def test_acl_decode_during_tracing_is_refused(self):
        calls = self._barrier_calls(acl=1, tracing=True, tokens=1)
        self.assertEqual(len(calls), 1)
        self.assertIn("dynamic host slot/sequence metadata", calls[0].args[0])

    def test_acl_eager_decode_is_not_refused(self):
        self.assertEqual(
            self._barrier_calls(acl=1, tracing=False, tokens=1), [])

    def test_cuda_tracing_is_not_refused(self):
        self.assertEqual(
            self._barrier_calls(acl=0, tracing=True, tokens=1), [])

    def test_acl_prefill_tracing_is_not_refused(self):
        self.assertEqual(
            self._barrier_calls(acl=1, tracing=True, tokens=8), [])


if __name__ == "__main__":
    unittest.main()
