"""Runtime-free negative tests for the shared adapter source boundary."""

from pathlib import Path
import tempfile
import unittest

from jittor_adapters.testing.relocatable import (
    adapter_sources, relocatable_violations,
)


ALLOWED = {"jittor", "jittor.distributed", "jittor.compat.module_patcher"}


class RelocatableContract(unittest.TestCase):
    def test_negative_cases_are_rejected_by_actual_scanner(self):
        cases = {
            "import jittor._runtime": "private import",
            "from jittor.compat.torch.context import get_install_context": "private import",
            "import jittor as framework; framework._secret()": "private attribute",
            "from jittor import _private": "private attribute",
            "from torch import nn as layers; layers._private()": "private attribute",
            "import torch as t; t.__version__ = '9'": "framework mutation",
            "torch.x += 1": "framework mutation",
            "torch.x: int = 1": "framework mutation",
            "torch.x, local = (1, 2)": "framework mutation",
            "setattr(torch, 'Tensor', object)": "framework mutation",
            "delattr(jittor, 'nn')": "framework mutation",
            "del jittor.nn": "framework mutation",
            "getattr(torch, '_torch_compat_install_context')": "private attribute",
            "hasattr(jittor, '_private')": "private attribute",
        }
        for text, kind in cases.items():
            with self.subTest(text=text):
                issues = relocatable_violations(text, ALLOWED)
                self.assertIn(kind, [issue.kind for issue in issues])
                self.assertTrue(all(issue.lineno >= 1 for issue in issues))

    def test_public_reads_and_local_writes_remain_legal(self):
        sources = [
            "import torch; result = torch.tensor([1.])",
            "from jittor.distributed import get_hccl_world_info",
            "from .layers import _local; _local()",
            "from jittor_adapters._common import require_version",
            "cfg = object(); cfg.__version__ = '1'; setattr(cfg, 'x', 1)",
            "import jittor as jt; getattr(jt.flags, 'use_acl', 0)",
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertEqual(relocatable_violations(source, ALLOWED), [])

    def test_allowlists_stay_per_adapter(self):
        source = "from jittor.distributed import get_hccl_world_info"
        self.assertEqual(relocatable_violations(source, ALLOWED), [])
        self.assertTrue(relocatable_violations(source, {"jittor"}))

    def test_nested_sources_are_scanned_and_tests_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            runtime = package / "nested" / "runtime.py"
            tests = package / "nested" / "tests" / "test_example.py"
            runtime.parent.mkdir(parents=True)
            tests.parent.mkdir(parents=True)
            runtime.write_text("torch.x = 1", encoding="utf-8")
            tests.write_text("torch._private()", encoding="utf-8")
            self.assertEqual(adapter_sources(package), [runtime])
            self.assertTrue(relocatable_violations(
                runtime.read_text(encoding="utf-8"), ALLOWED))


if __name__ == "__main__":
    unittest.main()
