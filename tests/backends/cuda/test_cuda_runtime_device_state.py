"""CUDA runtime-backed flag setters preserve rollback and alias behavior."""

import numpy as np
import pytest


def test_runtime_device_flag_rejects_invalid_index_and_remains_usable():
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")
    with jt.flag_scope(use_cuda=1):
        before = jt.flags.device_id
        # The flag holds the *requested* index (-1 until one is set); the
        # device actually in use is reported by ``current_device()``.
        before_dev = jt.current_device()
        try:
            with pytest.raises(RuntimeError, match="Invalid CUDA device index"):
                jt.flags.device_id = jt.get_device_count() + 1
            assert jt.flags.device_id == before
            assert jt.current_device() == before_dev
            result = jt.array([2.0, 3.0]).sqr()
            result.sync()
            assert result.device_id == before_dev
            assert result.location() == "device"
            np.testing.assert_array_equal(result.numpy(), [4.0, 9.0])
        finally:
            jt.flags.device_id = before


def test_runtime_device_aliases_share_the_native_flag():
    import jittor as jt

    if not jt.has_cuda:
        pytest.skip("CUDA runtime required")
    before = jt.runtime.use_cuda
    try:
        with jt.flag_scope(use_cuda=0):
            with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
                jt.flags.use_device = True
            with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
                assert all(getattr(jt.flags, name) == 1 for name in
                           ("use_cuda", "use_device", "use_acl", "use_rocm", "use_corex"))
            jt.flags.use_cuda = False
            with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
                assert jt.flags.use_device == 0
    finally:
        with pytest.warns(DeprecationWarning, match="deprecated accelerator-mode alias"):
            jt.flags.use_device = before
        jt.flags.use_cuda = before
