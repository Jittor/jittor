"""vLLM adapter accelerator checks read the core capability owner."""

import types
import unittest

from jittor_adapters.vllm.buffers import _acl_active, _native_cuda_active


class _Accelerator:
    def __init__(self, enabled):
        self.enabled = enabled


class _Capability:
    def __init__(self, **enabled):
        self._enabled = enabled

    def accelerator(self, name):
        return _Accelerator(self._enabled.get(name, False))


def _namespace(capability=None, **flags):
    values = {'use_cuda': 0, 'use_acl': 0}
    values.update(flags)
    return types.SimpleNamespace(
        flags=types.SimpleNamespace(**values),
        # Deliberately contradict the capability to prove which owner wins.
        compiler=types.SimpleNamespace(has_cuda=1, has_acl=1),
        capability=capability,
    )


class AcceleratorCapabilityTests(unittest.TestCase):
    def test_acl_requires_both_the_capability_and_the_runtime_flag(self):
        ready = _Capability(acl=True)
        self.assertTrue(_acl_active(_namespace(ready, use_acl=1)))
        self.assertFalse(_acl_active(_namespace(ready, use_acl=0)))

    def test_capability_disabled_wins_over_a_stale_compiler_flag(self):
        disabled = _Capability(acl=False, cuda=False)
        namespace = _namespace(disabled, use_acl=1, use_cuda=1)
        self.assertFalse(_acl_active(namespace))
        self.assertFalse(_native_cuda_active(namespace))

    def test_cuda_is_rejected_when_acl_is_the_active_backend(self):
        both = _Capability(cuda=True, acl=True)
        self.assertFalse(_native_cuda_active(_namespace(both, use_cuda=1, use_acl=1)))
        self.assertTrue(_native_cuda_active(
            _namespace(_Capability(cuda=True, acl=False), use_cuda=1)))

    def test_minimal_test_doubles_keep_the_flag_fallback(self):
        namespace = _namespace(None, use_cuda=1, use_acl=1)
        self.assertTrue(_acl_active(namespace))
        self.assertFalse(_native_cuda_active(namespace))


if __name__ == '__main__':
    unittest.main()
