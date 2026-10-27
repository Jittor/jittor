"""A non-Windows process must never certify the Windows installed-wheel gate."""

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tests'))
from _helpers.child_process import run_python_child  # noqa: E402

SCRIPT = ROOT / 'tools/release/check_windows_msvc.py'


class WindowsMSVCCheckTests(unittest.TestCase):
    def test_help_does_not_import_or_compile_jittor(self):
        result = run_python_child([str(SCRIPT), '--help'],
                                  without_torch_mode=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--wheel', result.stdout)
        self.assertNotIn('Compiling', result.stdout + result.stderr)

    @unittest.skipIf(os.name == 'nt', 'negative host guard requires a non-Windows host')
    def test_non_windows_cannot_emit_a_passed_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'result.json'
            result = run_python_child(
                [str(SCRIPT), '--wheel', str(root / 'missing.whl'),
                 '--checkout', str(root / 'checkout'), '--commit', 'a' * 40,
                 '--output', str(output)], without_torch_mode=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('requires a real Windows process', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__':
    unittest.main()
