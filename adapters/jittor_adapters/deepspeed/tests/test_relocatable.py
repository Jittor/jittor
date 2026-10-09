"""Portable form of the shared public-API adapter boundary, with negative cases."""

from pathlib import Path
import unittest

from jittor_adapters.testing.relocatable import (
    adapter_sources,
    relocatable_violations,
)

PACKAGE = Path(__file__).resolve().parents[1]
PUBLIC_JITTOR_IMPORTS = {
    "jittor",
    "jittor.distributed",
    "jittor.compat.module_patcher",
    "jittor.compat.transaction",
}


def violations(text):
    return relocatable_violations(text, PUBLIC_JITTOR_IMPORTS)


class Relocatable(unittest.TestCase):
    def test_entire_package_uses_only_public_framework_contracts(self):
        paths = adapter_sources(PACKAGE)
        self.assertTrue(paths)
        self.assertTrue(any(path.parent.name == "builders" for path in paths))
        for path in paths:
            with self.subTest(path=path.name):
                self.assertEqual(violations(path.read_text(encoding="utf-8")), [])

    def test_private_and_mutation_rules_reject_negative_examples(self):
        bad = [
            "from jittor.compat.torch.context import get_install_context",
            "import jittor as framework; framework._secret()",
            "import torch as t; t.__version__ = '9'",
            "setattr(torch, 'Tensor', object)",
            "del jittor.nn",
            "getattr(torch, '_torch_compat_install_context')",
        ]
        for text in bad:
            with self.subTest(text=text):
                self.assertTrue(violations(text))
        self.assertEqual(violations("import torch; y = torch.tensor([1.])"), [])
        self.assertEqual(violations("from jittor.distributed import get_hccl_world_info"), [])


if __name__ == "__main__":
    unittest.main()
