"""Runtime math policy is validated, snapshots into keys, and flushes old graphs."""

import jittor as jt
import pytest


def test_cuda_math_policy_rejects_invalid_values_and_restores_scope():
    before = jt.flags.cuda_kernel_math
    with jt.flag_scope(cuda_kernel_math="strict"):
        assert jt.flags.cuda_kernel_math == "strict"
        with pytest.raises(RuntimeError, match="cuda_kernel_math must be"):
            jt.flags.cuda_kernel_math = "not-a-policy"
        assert jt.flags.cuda_kernel_math == "strict"
    assert jt.flags.cuda_kernel_math == before


def test_cuda_math_compilation_uses_captured_key_not_current_policy():
    with jt.flag_scope(use_cuda=0, cuda_kernel_math="default"):
        result = jt.code([1], "int32", cpu_header=r'''
            #include "runtime/jit_policy.h"
            #include "runtime/runtime.h"
        ''', cpu_src=r'''
            using namespace jittor;
            const string startup = " -I\"a path\" --use_fast_math --fmad=false ";
            JK key;
            key << "probe";
            add_jit_define(key, "JIT_cuda_math", "strict");
            const auto strict = cuda_math_flags_for_key(startup, key.to_string());
            key.clear();
            key << "probe";
            add_jit_define(key, "JIT_cuda_math", "backend");
            const auto backend = cuda_math_flags_for_key(startup, key.to_string());
            key.clear();
            key << "probe";
            add_cuda_math_jit_define(key);
            const auto unchanged = cuda_math_flags_for_key(startup, key.to_string());
            @out(0) = runtime_flag_cuda_kernel_math() == "default"
                && &runtime_jit_policy() == &native_runtime().jit_policy()
                && strict.find("--use_fast_math") == string::npos
                && strict.find("--fmad=false") != string::npos
                && strict.find("--prec-div=true") != string::npos
                && strict.find("--prec-sqrt=true") != string::npos
                && strict.find("-I\"a path\"") != string::npos
                && backend.find("--fmad=false") == string::npos
                && backend.find("--use_fast_math") != string::npos
                && unchanged == startup;
        ''')
        assert result.item() == 1


def test_pending_graph_runs_under_old_math_policy():
    with jt.flag_scope(use_cuda=0, cuda_kernel_math="default"):
        result = jt.code([1], "int32", cpu_header=r'''
            #include "runtime/jit_policy.h"
        ''', cpu_src=r'''
            @out(0) = jittor::runtime_flag_cuda_kernel_math() == "default";
        ''')
        jt.flags.cuda_kernel_math = "strict"
        assert result.item() == 1
