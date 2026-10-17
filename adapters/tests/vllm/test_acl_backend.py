"""The NPU cache page policy must not change CUDA vLLM behavior."""

import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from jittor.compat import transaction as tx
from jittor_adapters.vllm.backend import declare_acl_block_size


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


if __name__ == '__main__':
    unittest.main()
