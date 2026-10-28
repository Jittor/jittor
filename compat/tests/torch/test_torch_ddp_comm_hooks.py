"""Default DDP hook imports work while unsupported hooks fail explicitly."""
import unittest

import torch


class TestDDPCommHooks(unittest.TestCase):
    def test_import_graph(self):
        from torch.distributed.algorithms import ddp_comm_hooks
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook

        self.assertIs(ddp_comm_hooks.default_hooks, default_hooks)
        self.assertIs(ddp_comm_hooks.powerSGD_hook, powerSGD_hook)
        self.assertIs(torch.distributed.algorithms.ddp_comm_hooks, ddp_comm_hooks)

    def test_unsupported_hooks_raise(self):
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks, powerSGD_hook

        for hook in (default_hooks.fp16_compress_hook,
                     default_hooks.bf16_compress_hook,
                     default_hooks.fp16_compress_wrapper,
                     default_hooks.bf16_compress_wrapper,
                     powerSGD_hook.powerSGD_hook,
                     powerSGD_hook.batched_powerSGD_hook,
                     powerSGD_hook.PowerSGDState):
            with self.subTest(hook=hook), self.assertRaises(NotImplementedError):
                hook(None, None)
