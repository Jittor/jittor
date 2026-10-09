"""Run portable DeepSpeed contracts in fresh processes, without native imports."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ADAPTERS = Path(__file__).resolve().parents[1]
SUPPORT = ADAPTERS.parent / "tests"
if (SUPPORT / "_helpers/child_process.py").is_file():
    # The monorepo owns interpreter selection and checkout pinning.
    sys.path.insert(0, str(SUPPORT))
    from _helpers.child_process import PYTHON, child_env
else:
    # An independently distributed adapter has no monorepo test helpers.
    from sys import executable as PYTHON

    def child_env(extra, inherit=False, without_torch_mode=True):
        assert not inherit
        env = dict(extra)
        if without_torch_mode:
            for name in (
                "REAL_TORCH_SITE",
                "JITTOR_TORCH_SHIM",
                "JITTOR_TORCH_INDEPENDENT",
                "JITTOR_TORCH_PROJECT_ROOT",
                "JITTOR_TORCH_RUNTIME_ROOT",
            ):
                env.pop(name, None)
        return env


def run_contract(path):
    """Keep fixture modules and runtime caches out of the invoking process."""
    with tempfile.TemporaryDirectory(prefix="deepspeed-contract-") as directory:
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(ADAPTERS),
            JITTOR_TORCH_SHIM="0",
            JITTOR_HOME=directory,
            PYTHONDONTWRITEBYTECODE="1",
        )
        subprocess.run(
            [PYTHON, str(path), "-v"],
            env=child_env(env, inherit=False, without_torch_mode=True),
            check=True,
            timeout=60,
        )


class DeepSpeedOfflineGate(unittest.TestCase):
    def test_lifecycle_and_source_contracts(self):
        run_contract(ADAPTERS / "jittor_adapters/deepspeed/tests/test_contracts.py")

    def test_relocatable_contracts(self):
        run_contract(ADAPTERS / "jittor_adapters/deepspeed/tests/test_relocatable.py")

    def test_shared_relocatable_source_rules(self):
        run_contract(ADAPTERS / "tests/test_relocatable_contract.py")

    def test_child_failure_is_not_reported_as_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "failing.py"
            path.write_text("raise SystemExit(7)\n", encoding="utf-8")
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_contract(path)
            self.assertEqual(error.exception.returncode, 7)

    def test_child_module_mutation_cannot_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mutation.py"
            path.write_text(
                "import sys, types\n"
                "sys.modules['deepspeed_gate_fixture'] = types.ModuleType('fixture')\n",
                encoding="utf-8",
            )
            before = sys.modules.get("deepspeed_gate_fixture")
            run_contract(path)
            self.assertIs(sys.modules.get("deepspeed_gate_fixture"), before)


if __name__ == "__main__":
    unittest.main()
