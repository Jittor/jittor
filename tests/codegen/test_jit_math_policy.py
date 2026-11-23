"""Runtime math policy is validated, snapshots into keys, and flushes old graphs."""

import jittor as jt
import pytest


def test_cuda_math_policy_rejects_invalid_values_and_restores_scope():
    from contextlib import ExitStack as _TestPolicyStack
    with _TestPolicyStack() as _test_policy_stack:
        before = jt.introspection.policy.runtime.cuda_kernel_math
        with jt.flag_scope(cuda_kernel_math="strict"):
            assert jt.introspection.policy.runtime.cuda_kernel_math == "strict"
            with pytest.raises(RuntimeError, match="cuda_kernel_math must be"):
                _test_policy_stack.enter_context(jt.runtime.scope(cuda_kernel_math="not-a-policy"))
            assert jt.introspection.policy.runtime.cuda_kernel_math == "strict"
        assert jt.introspection.policy.runtime.cuda_kernel_math == before


def test_cuda_math_compilation_uses_captured_key_not_current_policy():
    with jt.flag_scope(use_cuda=0, cuda_kernel_math="fast"):
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
            @out(0) = runtime_flag_cuda_kernel_math() == "fast"
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
    from contextlib import ExitStack as _TestPolicyStack
    with _TestPolicyStack() as _test_policy_stack:
        with jt.flag_scope(use_cuda=0, cuda_kernel_math="fast"):
            result = jt.code([1], "int32", cpu_header=r'''
            #include "runtime/jit_policy.h"
        ''', cpu_src=r'''
            @out(0) = jittor::runtime_flag_cuda_kernel_math() == "fast";
        ''')
            _test_policy_stack.enter_context(jt.runtime.scope(cuda_kernel_math="strict"))
            assert result.item() == 1


def test_strict_is_the_default_and_default_names_fast():
    # The compiled-in default, read by a fresh process: an earlier test in
    # this one may have left a scope's value behind.
    import os
    import subprocess
    from _helpers.child_process import child_env, python_executable
    env = {k: v for k, v in os.environ.items()
           if k not in ("JT_CUDA_KERNEL_MATH", "cuda_kernel_math")}
    out = subprocess.run(
        [python_executable(), "-c", "import jittor as jt; print(jt.flags.cuda_kernel_math)"],
        env=child_env(env, inherit=False), capture_output=True, text=True,
        check=True).stdout.split()
    assert out[-1] == "strict"
    with jt.flag_scope(cuda_kernel_math="default"):
        assert jt.flags.cuda_kernel_math == "fast"
    with jt.flag_scope(cuda_kernel_math="fast"):
        with jt.flag_scope(cuda_kernel_math="strict"):
            assert jt.flags.cuda_kernel_math == "strict"
        assert jt.flags.cuda_kernel_math == "fast"
