"""Preparing a key must not consume an operator's other backend capability."""

import numpy as np
import pytest
import jittor as jt
import jittor_utils

_CUDA_SOURCE = r"""
__global__ void write_value(@ARGS_DEF) {
    @PRECALC
    @out(0) = 29;
}
write_value<<<1, 1>>>(@ARGS);
"""


@pytest.fixture(scope="module")
def key_probe():
    return jittor_utils.compile_module(r'''
#include "core/var_holder.h"
#include "core/op.h"
namespace jittor {
// @pyjt(probe_keys)
vector<string> probe_keys(VarHolder* holder, bool accelerator) {
    auto* op = holder->var->input(0)->op();
    auto cpu = op->flag(OpFlags::_cpu);
    auto cuda = op->flag(OpFlags::_cuda);
    vector<string> keys;
    const auto outer = execution_target_backend();
    {
        ExecutionBackendScope scope(BackendId::Cpu);
        keys.push_back(op->get_jit_key(get_jk()));
        if (accelerator) {
            ExecutionBackendScope nested(BackendId::Cuda);
            keys.push_back(op->get_jit_key(get_jk()));
        }
        CHECK(execution_target_backend() == BackendId::Cpu);
        keys.push_back(op->get_jit_key(get_jk()));
    }
    CHECK(execution_target_backend() == outer);
    try {
        ExecutionBackendScope scope(BackendId::Cpu);
        throw 7;
    } catch (int value) { CHECK(value == 7); }
    CHECK(execution_target_backend() == outer);
    CHECK(op->flag(OpFlags::_cpu) == cpu);
    CHECK(op->flag(OpFlags::_cuda) == cuda);
    CHECK(keys.front() == keys.back());
    return keys;
}
}''', jt.compiler.cc_flags)


def test_cpu_prepare_preserves_dual_source_and_nested_target(key_probe):
    with jt.flag_scope(use_cuda=0):
        y = jt.code((1,), "float32", [], cpu_src="@out(0)=17;",
                    cuda_src=_CUDA_SOURCE)
        keys = key_probe.probe_keys(y, False)
        assert "JIT_cpu:1" in keys[0]
        np.testing.assert_array_equal(y.numpy(), [17])


@pytest.mark.skipif(not jt.has_cuda, reason="CUDA compiler and device required")
def test_cpu_prepared_operator_can_execute_cuda(key_probe):
    # The ordinary runtime switch flushes pending graphs. Prepare explicitly
    # for CPU inside the probe while keeping this graph pending for CUDA.
    with jt.flag_scope(use_cuda=1, backend_fallback="error"):
        y = jt.code((1,), "float32", [], cpu_src="@out(0)=17;",
                    cuda_src=_CUDA_SOURCE)
        keys = key_probe.probe_keys(y, True)
        assert "JIT_cpu:1" in keys[0]
        assert "JIT_cuda:1" in keys[1]
        assert keys[0] != keys[1]
        np.testing.assert_array_equal(y.numpy(), [29])
