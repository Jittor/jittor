"""Run the ACL bridge's actual failure-control code on a host without CANN."""

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "backends/acl/src/acl_op_exec.cc"


def _declaration(source, marker, suffix=""):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if not depth:
                return source[start:index + 1] + suffix
    raise AssertionError("Unterminated declaration: " + marker)


def _compile_and_run(tmp_path, stem, source):
    path = tmp_path / (stem + ".cc")
    path.write_text(source, encoding="utf-8")
    executable = tmp_path / stem
    result = subprocess.run([os.environ.get("CXX", "g++"), "-std=c++14",
                             str(path), "-o", str(executable)],
                            text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    return subprocess.run([str(executable)], capture_output=True, text=True,
                          timeout=10, cwd=tmp_path)


def test_execution_failure_is_not_a_fallback_decision(tmp_path):
    source = SOURCE.read_text(encoding="utf-8")
    dispatch = _declaration(source, "template<class Execute, class Fallback, class Cleanup>")
    harness = r'''
#include <cassert>
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
using std::string;
DISPATCH
struct Failure { int value; };
int main() {
    int executes = 0, fallbacks = 0, cleanups = 0;
    dispatch_acl_checked("unregistered variant", [&] { ++executes; },
        [&](const string& reason) { assert(reason == "unregistered variant"); ++fallbacks; },
        [&] { ++cleanups; });
    assert(executes == 0 && fallbacks == 1 && cleanups == 0);
    try {
        dispatch_acl_checked("unregistered variant", [&] { ++executes; },
            [&](const string&) { throw Failure{10}; }, [&] { ++cleanups; });
        assert(false);
    } catch (const Failure& failure) { assert(failure.value == 10); }
    assert(executes == 0 && cleanups == 0);
    dispatch_acl_checked("", [&] { ++executes; },
        [&](const string&) { ++fallbacks; }, [&] { ++cleanups; });
    assert(executes == 1 && fallbacks == 1 && cleanups == 0);
    for (bool cleanup_fails : {false, true}) {
        try {
            dispatch_acl_checked("", [&] { ++executes; throw Failure{42}; },
                [&](const string&) { ++fallbacks; },
                [&] { ++cleanups; if (cleanup_fails) throw std::runtime_error("cleanup"); });
            return 42;
        } catch (const Failure& failure) { assert(failure.value == 42); }
    }
    assert(executes == 3 && fallbacks == 1 && cleanups == 2);
}
'''
    result = _compile_and_run(tmp_path, "dispatch", harness.replace("DISPATCH", dispatch))
    assert result.returncode == 0, result.stderr
    poisoned = dispatch.replace("std::rethrow_exception(original);", 'fallback("kernel failure");')
    assert poisoned != dispatch
    result = _compile_and_run(tmp_path, "dispatch_poisoned", harness.replace("DISPATCH", poisoned))
    assert result.returncode != 0, "kernel-error fallback mutation escaped the regression"


def test_cpu_fallback_restores_original_mode_flags_and_fused_context(tmp_path):
    scope = _declaration(SOURCE.read_text(encoding="utf-8"), "class AclCpuFallbackScope", ";")
    harness = r'''
#include <cassert>
#include <map>
#include <string>
#include <vector>
using std::string;
using std::vector;
using loop_options_t = std::map<string, int>;
enum class BackendId { Cpu, Acl };
BackendId requested_backend = BackendId::Acl;
struct ExecutionBackendScope {
    BackendId previous;
    explicit ExecutionBackendScope(BackendId backend) : previous(requested_backend) {
        requested_backend = backend;
    }
    ~ExecutionBackendScope() { requested_backend = previous; }
};
struct OpFlags { enum Flags { _cpu, _cuda }; };
struct Op {
    string identity;
    int cpu = 0, cuda = 1;
    explicit Op(string name) : identity(name) {}
    string name() const { return identity; }
    int flag(OpFlags::Flags bit) const { return bit == OpFlags::_cpu ? cpu : cuda; }
    void set_flag(OpFlags::Flags bit, int value = 1) {
        (bit == OpFlags::_cpu ? cpu : cuda) = value;
    }
};
struct FusedOpContext {};
struct FusedOp : Op {
    vector<Op*> ops;
    FusedOpContext* context;
    loop_options_t loop_options_tuned;
    loop_options_t* loop_options;
    FusedOp() : Op("fused") {}
};
struct DeviceState { int use_cuda = 0; } state;
DeviceState& runtime_device_state() { return state; }
SCOPE
int main() {
    for (int mode : {0, 1}) {
        state.use_cuda = mode;
        Op child("unary"); child.cpu = 1;
        FusedOp fused;
        FusedOpContext original, temporary;
        loop_options_t original_options{{"original", 7}};
        fused.ops = {&child};
        fused.context = &original;
        fused.loop_options_tuned = {{"before", 3}};
        fused.loop_options = &original_options;
        try {
            AclCpuFallbackScope restore(&fused);
            assert(state.use_cuda == 0);
            assert(requested_backend == BackendId::Cpu);
            assert(fused.cpu == 0 && fused.cuda == 1 && child.cpu == 1 && child.cuda == 1);
            fused.context = &temporary;
            fused.loop_options_tuned = {{"cpu_tuned", 99}};
            fused.loop_options = &fused.loop_options_tuned;
            throw 13;
        } catch (int value) { assert(value == 13); }
        assert(state.use_cuda == mode);
        assert(requested_backend == BackendId::Acl);
        assert(fused.cpu == 0 && fused.cuda == 1 && child.cpu == 1 && child.cuda == 1);
        assert(fused.context == &original && fused.loop_options == &original_options);
        assert(fused.loop_options_tuned == loop_options_t({{"before", 3}}));
    }
}
'''
    result = _compile_and_run(tmp_path, "restore", harness.replace("SCOPE", scope))
    assert result.returncode == 0, result.stderr


def test_runner_adapter_keeps_descriptor_lifecycle_for_direct_and_registered_calls(tmp_path):
    runner = _declaration(SOURCE.read_text(encoding="utf-8"),
                          "template<class Runner, bool UsesRegistry = true>", ";")
    harness = r'''
#include <cassert>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
struct Check {
    bool ok;
    template<class T> Check& operator<<(const T&) { return *this; }
    ~Check() noexcept(false) { if (!ok) throw std::logic_error("registry"); }
};
#define INTERNAL_ASSERT(value) Check{bool(value)}
using aclTensor = int;
const std::map<std::string, int>& acl_op_registry() {
    static const std::map<std::string, int> entries{{"registered", 1}};
    return entries;
}
int drains = 0, destroys = 0, setups = 0, executes = 0, aclstream = 0;
int aclrtSynchronizeStream(int) { ++drains; return 0; }
int aclDestroyTensor(aclTensor*) { ++destroys; return 0; }
struct Runner {
    std::string name;
    int failure;
    aclTensor input = 0, output = 0;
    std::vector<aclTensor*> inputTensors, outputTensors;
    Runner(std::string name, int failure) : name(name), failure(failure) {}
    void setupInputDesc() {
        ++setups; inputTensors.push_back(&input);
        if (failure == 1) throw failure;
    }
    void setupOutputDesc() { ++setups; outputTensors.push_back(&output); }
    void executeOp(std::map<std::string, int>::const_iterator& entry) {
        ++executes;
        assert((entry == acl_op_registry().end()) == (name == "direct"));
        if (failure == 2) throw failure;
    }
    void cleanupDesc() {
        for (auto* value : inputTensors) aclDestroyTensor(value);
        for (auto* value : outputTensors) aclDestroyTensor(value);
    }
};
ADAPTER
int main() {
    AclExecutionRunner<Runner, false>("direct", 0).run();
    AclExecutionRunner<Runner>("registered", 0).run();
    assert(setups == 4 && executes == 2 && destroys == 4 && drains == 0);
    for (int failure : {1, 2}) {
        try { AclExecutionRunner<Runner, false>("direct", failure).run(); assert(false); }
        catch (int value) { assert(value == failure); }
    }
    assert(setups == 7 && executes == 3 && destroys == 7 && drains == 2);
}
'''
    result = _compile_and_run(tmp_path, "runner", harness.replace("ADAPTER", runner))
    assert result.returncode == 0, result.stderr


def test_preflight_and_execution_share_launcher_keys_before_any_fallback_migration():
    source = SOURCE.read_text(encoding="utf-8")
    fallback = _declaration(source, "void fallback_cpu(")
    assert fallback.index("check_backend_fallback(") < fallback.index("AclCpuFallbackScope restore")
    assert fallback.index("AclCpuFallbackScope restore") < fallback.index("migrate_to_cpu(")
    assert "try_exec_and_fallback_cpu" not in source
    assert 'LOGir << "fallback cpu"' not in source
    assert "dispatch_acl_checked(fused_acl_unsupported(ops)" in source
    assert source.count("op.name = fused_acl_name(current_op);") == 6
    assert "AclExecutionRunner<ArgReduceOpRunner, false>" in source
    assert "AclExecutionRunner<AdamWListOpRunner, false>" in source
    for variant in ("ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "Select", "Expand"):
        assert 'return "' + variant + '"' in source
    registration = (ROOT / "backends/acl/src/acl_jittor.cc").read_text()
    assert '{"ReduceProd", AclOpFunctions::direct(aclnnProd)}' in registration
