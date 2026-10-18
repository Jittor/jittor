"""The NPU cache page policy must not change CUDA vLLM behavior."""

import os
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from jittor.compat import transaction as tx
from jittor_adapters.vllm.backend import (
    declare_acl_block_size, patch_npu_lazy_all_reduce)
from jittor_adapters.vllm.npu_platform import platform_plugin


class ACLBlockSizeTests(unittest.TestCase):
    def test_block_128_only_on_acl(self):
        class Backend:
            @staticmethod
            def get_supported_kernel_block_sizes():
                return [16, 32]

            @classmethod
            def get_preferred_block_size(cls, default):
                return default

        fake_jt = ModuleType('jittor')
        fake_jt.compiler = SimpleNamespace(has_acl=0)
        fake_jt.flags = SimpleNamespace(use_acl=0)
        module = SimpleNamespace(FlashAttentionBackend=Backend)
        with tx.runtime_hook('acl-block-test-' + uuid.uuid4().hex) as hook:
            self.assertTrue(declare_acl_block_size(module))
            with patch.dict(sys.modules, {'jittor': fake_jt}):
                self.assertEqual(Backend.get_supported_kernel_block_sizes(), [16, 32])
                self.assertEqual(Backend.get_preferred_block_size(16), 16)
                fake_jt.compiler.has_acl = 1
                fake_jt.flags.use_acl = 1
                self.assertEqual(Backend.get_supported_kernel_block_sizes(), [128])
                self.assertEqual(Backend.get_preferred_block_size(16), 128)
        hook.rollback()


class NPULazyAllReduceTests(unittest.TestCase):
    def test_opt_in_uses_native_hccl_result_only_on_acl(self):
        calls = []

        class Output:
            def out_hint(self):
                calls.append("hint")
                return self

        output = Output()

        class Group:
            _backend_kind = "hccl"

            def _all_reduce(self, value, operation):
                calls.append((value, operation))
                return output

        class Communicator:
            world_size = 2
            device_group = Group()

            def all_reduce(self, value):
                calls.append("fallback")
                return value

        fake_jt = ModuleType("jittor")
        fake_jt.flags = SimpleNamespace(use_acl=1)
        module = SimpleNamespace(DeviceCommunicatorBase=Communicator)
        with patch.dict(os.environ, {"JITTOR_VLLM_NPU_LAZY_ALL_REDUCE": "1"}), \
                tx.runtime_hook("npu-lazy-reduce-test-" + uuid.uuid4().hex) as hook:
            self.assertTrue(patch_npu_lazy_all_reduce(module))
            with patch.dict(sys.modules, {"jittor": fake_jt}):
                value = object()
                self.assertIs(Communicator().all_reduce(value), output)
                self.assertEqual(calls, [(value, "sum"), "hint"])
                fake_jt.flags.use_acl = 0
                self.assertIs(Communicator().all_reduce(value), value)
                self.assertEqual(calls[-1], "fallback")
        hook.rollback()


class NPUWorkerShutdownGraceTests(unittest.TestCase):
    def test_npu_default_and_explicit_override(self):
        fake_jt = ModuleType("jittor")
        fake_jt.flags = SimpleNamespace(auto_graph_replay=1)
        with patch.dict(os.environ, {"JITTOR_VLLM_NPU_PLATFORM": "1"}):
            os.environ.pop("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", None)
            with patch.dict(sys.modules, {"jittor": fake_jt}):
                self.assertIn("JittorNpuPlatform", platform_plugin())
                self.assertEqual(os.environ[
                    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"], "30")
                os.environ["VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"] = "45"
                platform_plugin()
                self.assertEqual(os.environ[
                    "VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS"], "45")

    def test_inactive_plugin_does_not_set_timeout(self):
        with patch.dict(os.environ, {"JITTOR_VLLM_NPU_PLATFORM": "0"}):
            os.environ.pop("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", None)
            self.assertIsNone(platform_plugin())
            self.assertNotIn("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", os.environ)


if __name__ == '__main__':
    unittest.main()
