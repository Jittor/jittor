"""Registered callbacks must control execution of actual native graphs."""

import gc

import numpy as np
import pytest

from _helpers.native_op_dispatch import (
    _CPU_CODE, _CUDA_CODE, _dispatch_probe, _check_fused_callbacks,
)


def test_native_callback_replacement_pins_old_and_new_graph_definitions():
    import jittor as jt

    probe = _dispatch_probe(jt)
    data = np.arange(19, dtype=np.float32)
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0):
        jt.sync_all(True)
        source = jt.array(data).sync()
        original_id = probe.dispatch_probe_copy_id()
        old_graph = jt.ops.copy(source)
        probe.dispatch_probe_copy_install(False)
        try:
            assert probe.dispatch_probe_copy_id() == original_id
            new_graph = jt.ops.copy(source)
            np.testing.assert_array_equal(old_graph.numpy(), data)
            np.testing.assert_array_equal(new_graph.numpy(), data + 7)
            assert probe.dispatch_probe_counts()[0] == 1
        finally:
            try:
                jt.sync_all(True)
            finally:
                probe.dispatch_probe_copy_restore()
        np.testing.assert_array_equal(jt.ops.copy(source).numpy(), data)
        assert probe.dispatch_probe_copy_id() == original_id


def test_missing_registered_kernel_fails_instead_of_calling_old_virtual_run():
    import jittor as jt

    probe = _dispatch_probe(jt)
    data = np.arange(13, dtype=np.float32)
    with jt.flag_scope(use_cuda=0, lazy_execution=1, auto_flush_ops=0):
        jt.sync_all(True)
        source = jt.array(data).sync()
        failed_graph = None
        probe.dispatch_probe_copy_install(True)
        try:
            with pytest.raises(RuntimeError, match=r"No kernel registered for.*copy.*cpu"):
                failed_graph = jt.ops.copy(source)
                failed_graph.sync()
        finally:
            failed_graph = None
            gc.collect()
            try:
                jt.sync_all(True)
            finally:
                probe.dispatch_probe_copy_restore()
        np.testing.assert_array_equal(jt.ops.copy(source).numpy(), data)


def test_fused_cpu_execution_uses_registered_fragments_and_jit_kernel():
    import jittor as jt

    _check_fused_callbacks(jt, False)


def test_code_op_cpu_selects_cpu_source_from_dual_backend_definition():
    import jittor as jt

    data = np.arange(17, dtype=np.float32)
    with jt.flag_scope(use_cuda=0):
        source = jt.array(data)
        result = jt.code(source.shape, source.dtype, [source],
                         cpu_src=_CPU_CODE, cuda_src=_CUDA_CODE)
        np.testing.assert_array_equal(result.numpy(), data + 11)


def test_cpu_supported_ops_excludes_accelerator_only_kernels():
    import jittor as jt

    supported = set(jt.core.backend_supported_ops("cpu"))
    assert {"copy", "array", "binary", "random"} <= supported
    assert "fused_adamw" not in supported
    assert not any(name.startswith(("cublas_", "cudnn_", "cub_", "curand_", "nccl_"))
                   for name in supported)


def test_backend_is_selected_before_dual_source_codegen_cache_lookup():
    import jittor as jt

    data = np.arange(19, dtype=np.float32)
    with jt.runtime.scope(use_cuda=0, use_parallel_op_compiler=0, auto_flush_ops=0):
        source = jt.array(data).sync()
        first = jt.code(source.shape, source.dtype, [source],
                        cpu_src=_CPU_CODE, cuda_src=_CUDA_CODE)
        np.testing.assert_array_equal(first.numpy(), data + 11)
        second = jt.code(source.shape, source.dtype, [source],
                         cpu_src=_CPU_CODE.replace("+ 11", "+ 37"), cuda_src=_CUDA_CODE)
        np.testing.assert_array_equal(second.numpy(), data + 37)
