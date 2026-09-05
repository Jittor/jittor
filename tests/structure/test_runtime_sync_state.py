"""Contract for the first read-only Runtime/Context state view."""

import pytest


def test_runtime_sync_run_is_a_live_read_only_view_of_the_native_flag():
    import jittor as jt

    original = jt.flags.sync_run
    try:
        assert jt.runtime.sync_run == original
        with jt.flag_scope(sync_run=0):
            assert jt.runtime.sync_run == 0
        assert jt.runtime.sync_run == original

        with pytest.raises(AttributeError):
            jt.runtime.sync_run = 0
    finally:
        jt.flags.sync_run = original


def test_runtime_state_does_not_duplicate_device_or_backend_flags():
    import jittor as jt

    assert tuple(jt.runtime.__slots__) == ("_context",)
    assert jt.runtime.context.__class__.__name__ == "RuntimeContext"
    assert jt.runtime.context._flags is jt.flags
    assert jt.runtime.context.snapshot() == {
        "sync_run": jt.flags.sync_run,
        "device_id": getattr(jt.flags, "device_id", -1),
        "use_cuda": jt.flags.use_cuda,
        "lazy_execution": jt.flags.lazy_execution,
        "auto_flush_ops": jt.flags.auto_flush_ops,
        "auto_convert_64_to_32": jt.flags.auto_convert_64_to_32,
        "reuse_array": jt.flags.reuse_array,
        "no_grad": jt.flags.no_grad,
        "amp_reg": jt.flags.amp_reg,
        "auto_mixed_precision_level": jt.flags.auto_mixed_precision_level,
        "try_use_32bit_index": jt.flags.try_use_32bit_index,
        "no_fuse": jt.flags.no_fuse,
        "gopt_disable": jt.flags.gopt_disable,
        "exec_called": jt.flags.exec_called,
        "use_threading": jt.flags.use_threading,
        "profile_memory_enable": jt.flags.profile_memory_enable,
        "profiler_warmup": jt.flags.profiler_warmup,
        "profiler_enable": jt.flags.profiler_enable,
        "profiler_rerun": jt.flags.profiler_rerun,
        "profiler_record_peek": jt.flags.profiler_record_peek,
        "profiler_record_shape": jt.flags.profiler_record_shape,
        "profiler_hide_relay": jt.flags.profiler_hide_relay,
        "check_graph": jt.flags.check_graph,
        "missing_grad_error": jt.flags.missing_grad_error,
        "disable_lock": jt.flags.disable_lock,
    }
    assert jt.runtime.device_id == getattr(jt.flags, "device_id", -1)
    assert jt.runtime.use_cuda == jt.flags.use_cuda


def test_runtime_context_is_the_single_sync_run_owner():
    import jittor as jt

    context = jt.runtime.context
    original = jt.flags.sync_run
    try:
        jt.flags.sync_run = 0
        assert context.sync_run == 0
        assert jt.runtime.sync_run == 0
        with pytest.raises(AttributeError):
            context.sync_run = 1
        with pytest.raises(AttributeError):
            jt.runtime.context = context
    finally:
        jt.flags.sync_run = original


def test_runtime_device_id_is_a_live_read_only_view():
    import jittor as jt

    assert jt.runtime.device_id == getattr(jt.flags, "device_id", -1)
    with pytest.raises(AttributeError):
        jt.runtime.device_id = 0


def test_runtime_use_cuda_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.use_cuda
    try:
        assert jt.runtime.use_cuda == original
        with jt.flag_scope(use_cuda=0):
            assert jt.runtime.use_cuda == 0
        assert jt.runtime.use_cuda == original
        with pytest.raises(AttributeError):
            jt.runtime.use_cuda = 0
    finally:
        jt.flags.use_cuda = original


def test_runtime_lazy_execution_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.lazy_execution
    try:
        assert jt.runtime.lazy_execution == original
        with jt.flag_scope(lazy_execution=0):
            assert jt.runtime.lazy_execution == 0
            assert jt.runtime.context.snapshot()["lazy_execution"] == 0
        assert jt.runtime.lazy_execution == original
        with pytest.raises(AttributeError):
            jt.runtime.lazy_execution = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.lazy_execution = 0
    finally:
        jt.flags.lazy_execution = original


def test_runtime_auto_flush_ops_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.auto_flush_ops
    try:
        assert jt.runtime.auto_flush_ops == original
        with jt.flag_scope(auto_flush_ops=7):
            assert jt.runtime.auto_flush_ops == 7
            assert jt.runtime.context.snapshot()["auto_flush_ops"] == 7
        assert jt.runtime.auto_flush_ops == original
        with pytest.raises(AttributeError):
            jt.runtime.auto_flush_ops = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.auto_flush_ops = 0
    finally:
        jt.flags.auto_flush_ops = original


def test_runtime_auto_convert_64_to_32_is_a_live_view_and_controls_cpu_array_dtype():
    import numpy as np
    import jittor as jt

    original = jt.flags.auto_convert_64_to_32
    try:
        assert jt.runtime.auto_convert_64_to_32 == original
        with jt.flag_scope(auto_convert_64_to_32=0):
            assert jt.runtime.auto_convert_64_to_32 == 0
            assert jt.runtime.context.snapshot()["auto_convert_64_to_32"] == 0
            assert jt.array(np.array([1.5], dtype=np.float64)).dtype == "float64"
        with jt.flag_scope(auto_convert_64_to_32=1):
            assert jt.runtime.auto_convert_64_to_32 == 1
            assert jt.array(np.array([1.5], dtype=np.float64)).dtype == "float32"
        assert jt.runtime.auto_convert_64_to_32 == original
        with pytest.raises(AttributeError):
            jt.runtime.auto_convert_64_to_32 = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.auto_convert_64_to_32 = 0
    finally:
        jt.flags.auto_convert_64_to_32 = original


def test_runtime_reuse_array_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.reuse_array
    try:
        assert jt.runtime.reuse_array == original
        with jt.flag_scope(reuse_array=1):
            assert jt.runtime.reuse_array == 1
            assert jt.runtime.context.snapshot()["reuse_array"] == 1
        assert jt.runtime.reuse_array == original
        with pytest.raises(AttributeError):
            jt.runtime.reuse_array = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.reuse_array = 0
    finally:
        jt.flags.reuse_array = original


def test_runtime_no_grad_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.no_grad
    try:
        assert jt.runtime.no_grad == original
        with jt.flag_scope(no_grad=1):
            assert jt.runtime.no_grad == 1
            assert jt.runtime.context.snapshot()["no_grad"] == 1
        assert jt.runtime.no_grad == original
        with pytest.raises(AttributeError):
            jt.runtime.no_grad = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.no_grad = 0
    finally:
        jt.flags.no_grad = original


@pytest.mark.parametrize("flag_name", ["amp_reg", "auto_mixed_precision_level"])
def test_runtime_amp_flags_are_live_read_only_views(flag_name):
    import jittor as jt

    original = getattr(jt.flags, flag_name)
    scoped_value = 2 if flag_name == "amp_reg" else 4
    try:
        assert getattr(jt.runtime, flag_name) == original
        with jt.flag_scope(**{flag_name: scoped_value}):
            assert getattr(jt.runtime, flag_name) == scoped_value
            assert jt.runtime.context.snapshot()[flag_name] == scoped_value
        assert getattr(jt.runtime, flag_name) == original
        with pytest.raises(AttributeError):
            setattr(jt.runtime, flag_name, 0)
        with pytest.raises(AttributeError):
            setattr(jt.runtime.context, flag_name, 0)
    finally:
        setattr(jt.flags, flag_name, original)


def test_runtime_try_use_32bit_index_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.try_use_32bit_index
    try:
        assert jt.runtime.try_use_32bit_index == original
        with jt.flag_scope(try_use_32bit_index=1):
            assert jt.runtime.try_use_32bit_index == 1
            assert jt.runtime.context.snapshot()["try_use_32bit_index"] == 1
        assert jt.runtime.try_use_32bit_index == original
        with pytest.raises(AttributeError):
            jt.runtime.try_use_32bit_index = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.try_use_32bit_index = 0
    finally:
        jt.flags.try_use_32bit_index = original


def test_runtime_gopt_disable_is_a_live_read_only_view_and_cpu_execution_survives():
    import numpy as np
    import jittor as jt

    original = jt.flags.gopt_disable
    try:
        assert jt.runtime.gopt_disable == original
        with jt.flag_scope(gopt_disable=1):
            assert jt.runtime.gopt_disable == 1
            assert jt.runtime.context.snapshot()["gopt_disable"] == 1
            value = (jt.array(np.arange(4, dtype="float32")) + 1).numpy()
            np.testing.assert_array_equal(value, np.arange(1, 5, dtype="float32"))
        assert jt.runtime.gopt_disable == original
        with pytest.raises(AttributeError):
            jt.runtime.gopt_disable = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.gopt_disable = 0
    finally:
        jt.flags.gopt_disable = original


def test_runtime_no_fuse_is_a_live_read_only_view_and_cpu_execution_survives():
    import numpy as np
    import jittor as jt

    original = jt.flags.no_fuse
    try:
        assert jt.runtime.no_fuse == original
        with jt.flag_scope(no_fuse=1):
            assert jt.runtime.no_fuse == 1
            assert jt.runtime.context.snapshot()["no_fuse"] == 1
            value = (jt.array(np.arange(4, dtype="float32")) + 1).numpy()
            np.testing.assert_array_equal(value, np.arange(1, 5, dtype="float32"))
        assert jt.runtime.no_fuse == original
        with pytest.raises(AttributeError):
            jt.runtime.no_fuse = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.no_fuse = 0
    finally:
        jt.flags.no_fuse = original


def test_runtime_exec_called_is_a_live_read_only_execution_counter():
    import numpy as np
    import jittor as jt

    before = jt.runtime.exec_called
    assert before == jt.flags.exec_called
    value = (jt.array(np.arange(4, dtype="float32")) + 1).numpy()
    np.testing.assert_array_equal(value, np.arange(1, 5, dtype="float32"))
    assert jt.runtime.exec_called == jt.flags.exec_called
    assert jt.runtime.exec_called >= before
    assert jt.runtime.context.snapshot()["exec_called"] == jt.runtime.exec_called
    with pytest.raises(AttributeError):
        jt.runtime.exec_called = before
    with pytest.raises(AttributeError):
        jt.runtime.context.exec_called = before


def test_runtime_use_threading_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.use_threading
    try:
        assert jt.runtime.use_threading == original
        with jt.flag_scope(use_threading=1):
            assert jt.runtime.use_threading == 1
            assert jt.runtime.context.snapshot()["use_threading"] == 1
        assert jt.runtime.use_threading == original
        with pytest.raises(AttributeError):
            jt.runtime.use_threading = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.use_threading = 0
    finally:
        jt.flags.use_threading = original


def test_runtime_profile_memory_enable_is_a_live_read_only_view_and_cpu_execution_survives():
    import numpy as np
    import jittor as jt

    original = jt.flags.profile_memory_enable
    try:
        assert jt.runtime.profile_memory_enable == original
        with jt.flag_scope(profile_memory_enable=1):
            assert jt.runtime.profile_memory_enable == 1
            assert jt.runtime.context.snapshot()["profile_memory_enable"] == 1
            value = (jt.array(np.arange(4, dtype="float32")) * 2).numpy()
            np.testing.assert_array_equal(value, np.arange(0, 8, 2, dtype="float32"))
        assert jt.runtime.profile_memory_enable == original
        with pytest.raises(AttributeError):
            jt.runtime.profile_memory_enable = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.profile_memory_enable = 0
    finally:
        jt.flags.profile_memory_enable = original


def test_runtime_profiler_warmup_is_a_live_read_only_view_and_cpu_execution_survives():
    import numpy as np
    import jittor as jt

    original = jt.flags.profiler_warmup
    try:
        assert jt.runtime.profiler_warmup == original
        with jt.flag_scope(profiler_warmup=2):
            assert jt.runtime.profiler_warmup == 2
            assert jt.runtime.context.snapshot()["profiler_warmup"] == 2
            value = (jt.array(np.arange(4, dtype="float32")) + 3).numpy()
            np.testing.assert_array_equal(value, np.arange(3, 7, dtype="float32"))
        assert jt.runtime.profiler_warmup == original
        with pytest.raises(AttributeError):
            jt.runtime.profiler_warmup = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.profiler_warmup = 0
    finally:
        jt.flags.profiler_warmup = original


def test_runtime_profiler_enable_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.profiler_enable
    try:
        assert jt.runtime.profiler_enable == original
        with jt.flag_scope(profiler_enable=1):
            assert jt.runtime.profiler_enable == 1
            assert jt.runtime.context.snapshot()["profiler_enable"] == 1
        assert jt.runtime.profiler_enable == original
        with pytest.raises(AttributeError):
            jt.runtime.profiler_enable = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.profiler_enable = 0
    finally:
        jt.flags.profiler_enable = original


def test_runtime_profiler_rerun_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.profiler_rerun
    try:
        assert jt.runtime.profiler_rerun == original
        with jt.flag_scope(profiler_rerun=3):
            assert jt.runtime.profiler_rerun == 3
            assert jt.runtime.context.snapshot()["profiler_rerun"] == 3
        assert jt.runtime.profiler_rerun == original
        with pytest.raises(AttributeError):
            jt.runtime.profiler_rerun = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.profiler_rerun = 0
    finally:
        jt.flags.profiler_rerun = original


def test_runtime_profiler_record_peek_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.profiler_record_peek
    try:
        assert jt.runtime.profiler_record_peek == original
        with jt.flag_scope(profiler_record_peek=1):
            assert jt.runtime.profiler_record_peek == 1
            assert jt.runtime.context.snapshot()["profiler_record_peek"] == 1
        assert jt.runtime.profiler_record_peek == original
        with pytest.raises(AttributeError):
            jt.runtime.profiler_record_peek = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.profiler_record_peek = 0
    finally:
        jt.flags.profiler_record_peek = original


@pytest.mark.parametrize("flag_name", ["profiler_record_shape", "profiler_hide_relay"])
def test_runtime_profiler_metadata_flags_are_live_read_only_views(flag_name):
    import jittor as jt

    original = getattr(jt.flags, flag_name)
    try:
        assert getattr(jt.runtime, flag_name) == original
        with jt.flag_scope(**{flag_name: 1}):
            assert getattr(jt.runtime, flag_name) == 1
            assert jt.runtime.context.snapshot()[flag_name] == 1
        assert getattr(jt.runtime, flag_name) == original
        with pytest.raises(AttributeError):
            setattr(jt.runtime, flag_name, 0)
        with pytest.raises(AttributeError):
            setattr(jt.runtime.context, flag_name, 0)
    finally:
        setattr(jt.flags, flag_name, original)


def test_runtime_check_graph_is_a_live_read_only_view_and_cpu_execution_survives():
    import numpy as np
    import jittor as jt

    original = jt.flags.check_graph
    try:
        assert jt.runtime.check_graph == original
        with jt.flag_scope(check_graph=1):
            assert jt.runtime.check_graph == 1
            assert jt.runtime.context.snapshot()["check_graph"] == 1
            value = (jt.array(np.arange(4, dtype="float32")) + 4).numpy()
            np.testing.assert_array_equal(value, np.arange(4, 8, dtype="float32"))
        assert jt.runtime.check_graph == original
        with pytest.raises(AttributeError):
            jt.runtime.check_graph = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.check_graph = 0
    finally:
        jt.flags.check_graph = original


def test_runtime_missing_grad_error_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.missing_grad_error
    try:
        assert jt.runtime.missing_grad_error == original
        with jt.flag_scope(missing_grad_error=1):
            assert jt.runtime.missing_grad_error == 1
            assert jt.runtime.context.snapshot()["missing_grad_error"] == 1
        assert jt.runtime.missing_grad_error == original
        with pytest.raises(AttributeError):
            jt.runtime.missing_grad_error = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.missing_grad_error = 0
    finally:
        jt.flags.missing_grad_error = original


def test_runtime_disable_lock_is_a_live_read_only_view():
    import jittor as jt

    original = jt.flags.disable_lock
    try:
        assert jt.runtime.disable_lock == original
        with jt.flag_scope(disable_lock=not bool(original)):
            expected = int(not bool(original))
            assert jt.runtime.disable_lock == expected
            assert jt.runtime.context.snapshot()["disable_lock"] == expected
        assert jt.runtime.disable_lock == original
        with pytest.raises(AttributeError):
            jt.runtime.disable_lock = 0
        with pytest.raises(AttributeError):
            jt.runtime.context.disable_lock = 0
    finally:
        jt.flags.disable_lock = original
