"""Native per-backend publication and pinned definitions, without JIT or a driver."""

import os
from pathlib import Path
import subprocess


SRC = Path(__file__).resolve().parents[3] / "src"


def test_publishing_one_backend_preserves_identity_other_backends_and_old_pins(tmp_path):
    source = tmp_path / "implementation_publication.cc"
    source.write_text(r'''
#include "core/op.h"
#include "ops/op_register.h"
#include "runtime/configuration.h"
#include <cassert>
using namespace jittor;
namespace jittor {
bool g_supports_color = false;
int log_v = 0;
string log_vprefix;
void print_prefix(std::ostream*) {}
void flush_log() {}
// Reached from the throwing log macros in log.h, which are header-inline, so
// every standalone snippet that includes a jittor header has to provide it.
string message_without_log_prefix(const string& message) { return message; }
void send_log(std::ostringstream&&, char, int) {}
bool check_vlog(const char*, int) { return false; }
string Op::op_name_to_file_name(const string& name) { return name.substr(0, name.find('.')); }
BackendRegistry& backend_registry() { static BackendRegistry value; return value; }
const BackendOps& BackendRegistry::get(const string&) const { static BackendOps cpu; return cpu; }
StartupConfigState& runtime_startup_config() { static StartupConfigState state; return state; }
// configuration.cc reports which flags the environment set (2.22); the table
// itself lives in log.cc, which this translation unit deliberately does not
// link. Same reason as the log symbols above.
vector<EnvFlagSource>& env_flag_sources() { static vector<EnvFlagSource> value; return value; }
}
namespace {
int result = 0;
void old_kernel(Op*) { result = 1; }
void new_kernel(Op*) { result = 2; }
void other_kernel(Op*) { result = 3; }
jit_op_entry_t compile_kernel(Op*) { return new_kernel; }
int constructor(int value) { return value; }
int composer_calls = 0, failing_calls = 0;
OpImplementation compose_backend(const OpDef&, const OpImplementation& implementation) {
    ++composer_calls;
    if (implementation.kernel.compile) return implementation;
    auto composed = implementation;
    composed.kernel.compile = compile_kernel;
    composed.kernel.fallback_only = true;
    return composed;
}
OpImplementation failing_composer(const OpDef& definition, const OpImplementation& implementation) {
    if (++failing_calls == 2) throw std::runtime_error("composer failed");
    return compose_backend(definition, implementation);
}
}
int main() {
    NativeOpRegistry registry;
    OpDef initial{"publication_probe", "core.cc", "-DCORE",
        {op_constructor_entry(&constructor)}, {{"value", 17}}};
    OpImplementation cpu;
    cpu.kernel.native = old_kernel;
    cpu.codegen.source_path = "cpu.cc";
    OpImplementation cuda;
    cuda.kernel.native = other_kernel;
    cuda.codegen.source_path = "cuda.cc";
    initial.implementations[BackendId::Cpu] = cpu;
    initial.implementations[BackendId::Cuda] = cuda;
    registry.register_op(initial);
    const auto old_graph = registry.definition("publication_probe");
    const auto id = old_graph->id;
    auto replacement = cpu;
    replacement.kernel.native = new_kernel;
    replacement.kernel.compile = compile_kernel;
    replacement.codegen.source_path = "replacement.cc";
    registry.register_op_implementation("publication_probe.member", BackendId::Cpu, replacement);
    const auto new_graph = registry.definition("publication_probe");
    assert(new_graph.get() != old_graph.get());
    assert(new_graph->id == id && registry.id("publication_probe") == id);
    assert(new_graph->name == old_graph->name);
    assert(new_graph->constructors == old_graph->constructors);
    assert(new_graph->codegen.source_path == old_graph->codegen.source_path);
    assert(new_graph->codegen.extra_flags == old_graph->codegen.extra_flags);
    assert(new_graph->codegen.var_members == old_graph->codegen.var_members);
    assert(!new_graph->compile_identity.empty());
    assert(new_graph->compile_identity != old_graph->compile_identity);
    old_graph->implementations.at(BackendId::Cpu).kernel.native(nullptr);
    assert(result == 1);
    new_graph->implementations.at(BackendId::Cpu).kernel.native(nullptr);
    assert(result == 2);
    assert(new_graph->implementations.at(BackendId::Cpu).kernel.compile == compile_kernel);
    assert(new_graph->implementations.at(BackendId::Cuda).kernel.native == other_kernel);
    replacement.codegen.source_path = "caller_mutated.cc";
    assert(new_graph->implementations.at(BackendId::Cpu).codegen.source_path == "replacement.cc");

    // A provider can add a previously absent backend without re-registering
    // constructors or changing the numeric operator identity.
    replacement.kernel.fallback_only = true;
    registry.register_op_implementation("publication_probe", BackendId::Acl, replacement);
    auto with_fallback = registry.definition("publication_probe");
    assert(with_fallback->id == id);
    assert(with_fallback->compile_identity != new_graph->compile_identity);
    assert(old_graph->implementations.count(BackendId::Acl) == 0);
    assert(registry.supported_ops(BackendId::Acl).empty());
    assert(registry.supported_ops(BackendId::Cpu) == vector<string>{"publication_probe"});
    replacement.kernel.fallback_only = false;
    registry.register_op_implementation("publication_probe", BackendId::Acl, replacement);
    assert(registry.supported_ops(BackendId::Acl) == vector<string>{"publication_probe"});
    auto current = registry.definition("publication_probe");
    bool missing = false;
    try { registry.register_op_implementation("missing", BackendId::Cpu, replacement); }
    catch (const UserError&) { missing = true; }
    assert(missing && registry.definition("publication_probe") == current);
    bool empty = false;
    try { registry.register_op_implementation("publication_probe", BackendId::Cpu, {}); }
    catch (const UserError&) { empty = true; }
    assert(empty && registry.definition("publication_probe") == current);

    op_registe(initial);
    auto global_old = get_op_definition("publication_probe");
    register_op_implementation("publication_probe", BackendId::Cpu, replacement);
    auto global_new = get_op_definition("publication_probe");
    assert(global_new->id == global_old->id && global_new != global_old);
    assert(global_old->implementations.at(BackendId::Cpu).kernel.native == old_kernel);
    assert(global_new->implementations.at(BackendId::Cpu).kernel.native == new_kernel);

    NativeOpRegistry first_boot, second_boot, changed_boot, runtime_then_boot;
    for (auto* owner : {&first_boot, &second_boot, &changed_boot, &runtime_then_boot})
        owner->register_op(initial);
    first_boot.register_op_implementation("publication_probe", BackendId::Acl, replacement, "acl-native-v1");
    second_boot.register_op_implementation("publication_probe", BackendId::Acl, replacement, "acl-native-v1");
    changed_boot.register_op_implementation("publication_probe", BackendId::Acl, replacement, "acl-native-v2");
    const auto first_identity = first_boot.definition("publication_probe")->compile_identity;
    assert(first_identity == second_boot.definition("publication_probe")->compile_identity);
    assert(first_identity != changed_boot.definition("publication_probe")->compile_identity);
    runtime_then_boot.register_op_implementation("publication_probe", BackendId::Cpu, replacement);
    runtime_then_boot.register_op_implementation("publication_probe", BackendId::Acl, replacement, "acl-native-v1");
    assert(first_identity != runtime_then_boot.definition("publication_probe")->compile_identity);
    auto pinned_boot = first_boot.definition("publication_probe");

    OpDef ordinary = initial;
    ordinary.name = "ordinary";
    ordinary.implementations.clear();
    ordinary.implementations[BackendId::Acl] = cpu;
    OpDef explicit_provider = ordinary;
    explicit_provider.name = "explicit";
    explicit_provider.implementations.at(BackendId::Acl).kernel.compile = compile_kernel;
    NativeOpRegistry before_install, after_install;
    before_install.register_op(ordinary);
    before_install.register_op(explicit_provider);
    before_install.register_op(initial); // no ACL row must remain absent
    auto old_ordinary = before_install.definition("ordinary");
    auto old_explicit = before_install.definition("explicit");
    before_install.register_backend_implementation_composer(BackendId::Acl, compose_backend, "acl-compose-v1");
    after_install.register_backend_implementation_composer(BackendId::Acl, compose_backend, "acl-compose-v1");
    after_install.register_op(ordinary);
    after_install.register_op(explicit_provider);
    after_install.register_op(initial);
    auto composed = before_install.definition("ordinary");
    auto late_composed = after_install.definition("ordinary");
    assert(old_ordinary->id == composed->id);
    assert(old_ordinary->implementations.at(BackendId::Acl).kernel.compile == nullptr);
    assert(composed->implementations.at(BackendId::Acl).kernel.compile == compile_kernel);
    assert(late_composed->implementations.at(BackendId::Acl).kernel.compile == compile_kernel);
    assert(composed->compile_identity == late_composed->compile_identity);
    assert(composed->implementations.at(BackendId::Acl).kernel.fallback_only);
    assert(before_install.definition("explicit") == old_explicit);
    assert(after_install.definition("explicit")->compile_identity.empty());
    assert(before_install.definition("publication_probe")->implementations.count(BackendId::Acl) == 0);
    assert(before_install.supported_ops(BackendId::Acl) == vector<string>{"explicit"});
    assert(before_install.supported_ops(BackendId::Acl) == after_install.supported_ops(BackendId::Acl));
    const int calls_before_repeat = composer_calls;
    before_install.register_backend_implementation_composer(BackendId::Acl, compose_backend, "acl-compose-v1");
    assert(composer_calls == calls_before_repeat && before_install.definition("ordinary") == composed);
    bool conflict = false;
    try { before_install.register_backend_implementation_composer(BackendId::Acl, compose_backend, "different"); }
    catch (const UserError&) { conflict = true; }
    assert(conflict && before_install.definition("ordinary") == composed);
    before_install.register_op(ordinary);
    auto runtime_composed = before_install.definition("ordinary");
    assert(runtime_composed->id == composed->id && runtime_composed->compile_identity != composed->compile_identity);
    assert(runtime_composed->implementations.at(BackendId::Acl).kernel.compile == compile_kernel);

    NativeOpRegistry failed_install;
    failed_install.register_op(ordinary);
    OpDef another = ordinary; another.name = "another";
    failed_install.register_op(another);
    auto pinned_first = failed_install.definition("ordinary");
    auto pinned_second = failed_install.definition("another");
    bool rejected = false;
    try { failed_install.register_backend_implementation_composer(BackendId::Acl, failing_composer, "bad"); }
    catch (const std::runtime_error&) { rejected = true; }
    assert(rejected && failed_install.definition("ordinary") == pinned_first);
    assert(failed_install.definition("another") == pinned_second);
    failed_install.register_backend_implementation_composer(BackendId::Acl, compose_backend, "good");
    register_backend_implementation_composer(BackendId::Acl, compose_backend, "global-compose-v1");
    op_registe(ordinary);
    assert(get_op_definition("ordinary")->implementations.at(BackendId::Acl).kernel.compile == compile_kernel);

    runtime_startup_config().seal();
    bool sealed = false;
    try {
        first_boot.register_op_implementation("publication_probe", BackendId::Acl, replacement, "late-bootstrap");
    } catch (const std::runtime_error& error) {
        sealed = string(error.what()).find("immutable startup configuration") != string::npos;
    }
    assert(sealed && first_boot.definition("publication_probe") == pinned_boot);
    first_boot.register_op_implementation("publication_probe", BackendId::Acl, replacement);
    assert(first_boot.definition("publication_probe")->compile_identity != first_identity);
    assert(pinned_boot->compile_identity == first_identity);
    bool composer_sealed = false;
    try { before_install.register_backend_implementation_composer(BackendId::Cuda, compose_backend, "too-late"); }
    catch (const std::runtime_error&) { composer_sealed = true; }
    assert(composer_sealed);
    // Identical installation is a no-op, even after sealing; loading a new
    // extension still composes its implementation under the installed policy.
    before_install.register_backend_implementation_composer(BackendId::Acl, compose_backend, "acl-compose-v1");
    another.name = "after_seal";
    before_install.register_op(another);
    assert(before_install.definition("after_seal")->implementations.at(BackendId::Acl).kernel.compile == compile_kernel);
    assert(before_install.definition("after_seal")->implementations.at(BackendId::Acl).kernel.fallback_only);
}
''', encoding="utf-8")
    executable = tmp_path / "implementation_publication"
    result = subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++14", "-pthread", "-I", str(SRC),
         str(source), str(SRC / "ops/op_register.cc"), str(SRC / "runtime/configuration.cc"),
         "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True,
                            timeout=10, cwd=tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
