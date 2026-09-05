"""Shared native registration probes for CPU and CUDA tests."""

import numpy as np


_PROBE_SOURCE = r"""
#pragma once
#include "op.h"
#include "var.h"
#include "ops/op_register.h"
#include <stdexcept>
namespace jittor {
namespace {
OpDef saved_copy, saved_binary, saved_fused;
bool copy_installed = false, fused_installed = false;
BackendId observed_backend = BackendId::Cpu;
int copy_runs = 0, binary_fragments = 0, fused_runs = 0;
void shifted_copy(Op* op) {
    ++copy_runs;
    auto* input = op->inputs().front();
    auto* output = op->outputs().front();
    if (input->dtype() != ns_float32 || output->dtype() != ns_float32)
        throw std::runtime_error("dispatch probe requires float32 copy");
    const auto* source = static_cast<const float*>(input->mem_ptr);
    auto* destination = static_cast<float*>(output->mem_ptr);
    for (int64 i = 0; i < input->num; ++i) destination[i] = source[i] + 7;
}
void observed_fragment(Op* op, JK& key) {
    ++binary_fragments;
    saved_binary.implementations.at(observed_backend).codegen.fragment(op, key);
}
void observed_fused_jit(Op* op, JK& key) {
    ++fused_runs;
    saved_fused.implementations.at(observed_backend).kernel.jit(op, key);
}
}
// @pyjt(dispatch_probe_copy_install)
void dispatch_probe_copy_install(bool missing) {
    if (copy_installed) throw std::runtime_error("copy probe already installed");
    saved_copy = get_op_info("copy");
    auto replacement = saved_copy;
    if (missing) replacement.implementations.erase(BackendId::Cpu);
    else replacement.implementations.at(BackendId::Cpu).kernel.native = shifted_copy;
    op_registe(replacement);
    copy_installed = true;
    copy_runs = 0;
}
// @pyjt(dispatch_probe_copy_restore)
void dispatch_probe_copy_restore() {
    if (!copy_installed) return;
    op_registe(saved_copy);
    copy_installed = false;
}
// @pyjt(dispatch_probe_fused_install)
void dispatch_probe_fused_install(bool accelerator) {
    if (fused_installed) throw std::runtime_error("fused probe already installed");
    observed_backend = accelerator ? accelerator_backend_id() : BackendId::Cpu;
    saved_binary = get_op_info("binary");
    saved_fused = get_op_info("fused");
    auto binary = saved_binary;
    auto fused = saved_fused;
    binary.implementations.at(observed_backend).codegen.fragment = observed_fragment;
    fused.implementations.at(observed_backend).kernel.jit = observed_fused_jit;
    if (accelerator) fused.implementations.erase(BackendId::Cpu);
    op_registe(binary);
    op_registe(fused);
    fused_installed = true;
    binary_fragments = fused_runs = 0;
}
// @pyjt(dispatch_probe_fused_restore)
void dispatch_probe_fused_restore() {
    if (!fused_installed) return;
    op_registe(saved_binary);
    op_registe(saved_fused);
    fused_installed = false;
}
// @pyjt(dispatch_probe_counts)
vector<int> dispatch_probe_counts() { return {copy_runs, binary_fragments, fused_runs}; }
// @pyjt(dispatch_probe_copy_id)
int64 dispatch_probe_copy_id() { return get_op_id("copy"); }
}
"""


_CPU_CODE = "for (int i = 0; i < in0_shape0; ++i) @out(i) = @in0(i) + 11;"
_CUDA_CODE = r"""
__global__ static void dispatch_probe_kernel(@ARGS_DEF) {
    @PRECALC
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < in0_shape0) @out(i) = @in0(i) + 23;
}
dispatch_probe_kernel<<<(in0_shape0 + 127) / 128, 128>>>(@ARGS);
"""


def _dispatch_probe(jt):
    import jittor_utils

    return jittor_utils.compile_module(_PROBE_SOURCE, jt.compiler.cc_flags)


def _check_fused_callbacks(jt, accelerator):
    probe = _dispatch_probe(jt)
    data = np.arange(257, dtype=np.float32) / 16
    with jt.flag_scope(use_cuda=int(accelerator), lazy_execution=1,
                       auto_flush_ops=0, use_parallel_op_compiler=0):
        jt.sync_all(True)
        source = jt.array(data).sync()
        probe.dispatch_probe_fused_install(accelerator)
        try:
            # Scalar array constants exercise ArrayOp's force-fused codegen.
            result = (source + jt.array(np.float32(2))) * jt.array(np.float32(3))
            result.sync()
            if accelerator:
                assert result.location() == "device"
            np.testing.assert_array_equal(result.numpy(), (data + 2) * 3)
            counts = probe.dispatch_probe_counts()
            assert counts[1] > 0
            assert counts[2] > 0
        finally:
            try:
                jt.sync_all(True)
            finally:
                probe.dispatch_probe_fused_restore()
