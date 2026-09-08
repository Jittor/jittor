"""Backend callbacks observed through real tensor execution, not registry metadata."""

import numpy as np
import pytest
import warnings


from _helpers.backend_probe import _backend_probe


def test_cpu_tensor_execution_uses_registered_allocator():
    import jittor as jt

    assert "cpu" in jt.core.registered_backends()
    assert jt.core.backend_device_count("cpu") == 1
    probe = _backend_probe(jt)
    data = np.arange(33, dtype=np.float32)
    with jt.flag_scope(use_cuda=0, use_cuda_host_allocator=0):
        jt.sync_all(True)
        probe.native_backend_probe_install(0)
        try:
            value = jt.array(data) * 3 + 7
            np.testing.assert_array_equal(value.numpy(), data * 3 + 7)
            assert value.location() == "cpu"
            jt.sync_all(True)
            assert probe.native_backend_probe_counts()[0] > 0
            assert probe.native_backend_probe_counts()[1] > 0
            assert probe.native_backend_probe_counts()[3] > 0
        finally:
            probe.native_backend_probe_restore()
        # Restoring the table must leave ordinary execution usable.
        np.testing.assert_array_equal((value + 1).numpy(), data * 3 + 8)


def test_unknown_backend_name_is_rejected():
    import jittor as jt

    with pytest.raises(RuntimeError, match="[Bb]ackend|registered"):
        jt.core.backend_device_count("missing_backend_for_registry_test")


@pytest.mark.parametrize("name", ["use_device", "use_acl", "use_rocm", "use_corex"])
def test_legacy_backend_aliases_warn_and_preserve_setter_semantics(name):
    import jittor as jt

    with jt.runtime.scope(use_cuda=0):
        with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
            assert getattr(jt.flags, name) == 0
        with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
            setattr(jt.flags, name, False)
        assert jt.runtime.use_cuda == 0


def test_warning_as_error_does_not_change_backend_mode():
    import jittor as jt

    with jt.runtime.scope(use_cuda=0), warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        with pytest.raises(DeprecationWarning, match="deprecated accelerator-mode alias"):
            jt.flags.use_device = True
        assert jt.runtime.use_cuda == 0
