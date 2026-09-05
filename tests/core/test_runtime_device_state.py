"""Python flags, core state and dynamically compiled operators share storage."""

import jittor as jt
import numpy as np


def _native_device_flags():
    return jt.code(
        [4], "int32", cpu_header='#include "runtime/runtime.h"',
        cpu_src='''
        auto& state = jittor::runtime_device_state();
        @out(0) = state.use_cuda;
        @out(1) = state.device_id;
        @out(2) = state.sync_run;
        @out(3) = &state == &jittor::native_runtime().devices()
            && &state.use_cuda == &jittor::runtime_flag_use_cuda()
            && &state.device_id == &jittor::runtime_flag_device_id()
            && &state.sync_run == &jittor::runtime_flag_sync_run();
        ''',
    ).numpy()


def test_python_flag_writes_reach_core_and_jit_owner():
    saved = jt.flags.sync_run
    with jt.flag_scope(use_cuda=0, sync_run=0):
        np.testing.assert_array_equal(
            _native_device_flags(), [0, jt.flags.device_id, 0, 1])
        with jt.flag_scope(sync_run=True):
            np.testing.assert_array_equal(
                _native_device_flags(), [0, jt.flags.device_id, 1, 1])
        assert jt.flags.sync_run == 0
        assert all(getattr(jt.flags, name) == 0 for name in
                   ("use_cuda", "use_device", "use_acl", "use_rocm", "use_corex"))
    assert jt.flags.sync_run == saved
