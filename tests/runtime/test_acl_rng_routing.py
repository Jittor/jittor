"""Host routing regression: ACL state must never ask for cuRAND."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class TestAclRngRouting(unittest.TestCase):
    def test_acl_legacy_accelerator_rng_uses_registered_native_owner(self):
        calls = []
        core = types.SimpleNamespace(
            registered_backends=lambda: ['cpu', 'acl'],
            get_device_rng_state=lambda name, index: calls.append((name, index)) or 'ACL_STATE',
        )
        jt = types.ModuleType('jittor')
        jt.has_cuda = True
        jt.compiler = types.SimpleNamespace(has_acl=True)
        jt.core = core
        jt.get_device_count = lambda: 1
        jt.current_device = lambda: 0
        libraries = types.ModuleType('jittor._runtime.backend_libraries')
        def library(name, **kwargs):
            self.fail('ACL RNG attempted to load %s' % name)
        libraries.get_library = library
        root = types.ModuleType('jittor._runtime')
        root.__path__ = []
        path = Path(__file__).resolve().parents[2] / 'python/jittor/_runtime/cuda_rng.py'
        spec = importlib.util.spec_from_file_location('jittor._runtime.cuda_rng', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'jittor': jt, 'jittor._runtime': root,
                                     'jittor._runtime.backend_libraries': libraries}):
            spec.loader.exec_module(module)
            self.assertEqual(module.get_cuda_rng_state(0), 'ACL_STATE')
        self.assertEqual(calls, [('acl', 0)])


if __name__ == '__main__':
    unittest.main()
