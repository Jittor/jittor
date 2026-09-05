"""Exercise native capability selection without importing Jittor or compiling JIT kernels."""

import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "python/jittor/src"


def test_capability_registration_is_lazy_typed_and_observes_late_operators(tmp_path):
    source = tmp_path / "capability_contract.cc"
    source.write_text(r'''
#include "ops/op_capability.h"
#include <cassert>
#include <map>
using namespace jittor;
namespace jittor {
bool g_supports_color = false;
void print_prefix(std::ostream*) {}
void flush_log() {}
std::map<string, OpInfo>& definitions() {
    static std::map<string, OpInfo> value;
    return value;
}
bool has_op(const string& name) { return definitions().count(name); }
OpInfo get_op_info(const string& name) { return definitions().at(name); }
BackendRegistry& backend_registry() { static BackendRegistry value; return value; }
const BackendOps& BackendRegistry::get(const string& name) const {
    static BackendOps cpu;
    assert(name == "cpu");
    return cpu;
}
}
namespace {
int increment(int x) { return x + 1; }
int double_value(int x) { return x * 2; }
bool nonnegative(int x) { return x >= 0; }
// This executes before there is an OpInfo for the implementation.
RegisterOpCapability<int, int> registration(
    BackendId::Cpu, OpCapability::Matmul, "late_implementation", nonnegative);
}
int main() {
    auto find = [](int x) {
        return find_op_capability<int, int>(BackendId::Cpu, OpCapability::Matmul, x);
    };
    assert(!has_op_capability(BackendId::Cpu, OpCapability::Matmul));
    assert(!find(1));
    assert(backend_supported_capabilities("cpu").empty());
    OpInfo definition;
    definition.name = "late_implementation";
    definition.constructors = {op_constructor_entry(&increment)};
    definitions()[definition.name] = definition;
    assert(has_op_capability(BackendId::Cpu, OpCapability::Matmul));
    assert(find(4)(4) == 5);
    assert(!find(-1));
    assert(!has_op_capability(BackendId::Cuda, OpCapability::Matmul));
    assert((!find_op_capability<int, int>(BackendId::Cuda, OpCapability::Matmul, 1)));
    assert(backend_supported_capabilities("cpu") == vector<string>{"matmul"});
    // Replacement is observed rather than retaining the first function pointer.
    definitions()[definition.name].constructors = {op_constructor_entry(&double_value)};
    assert(find(4)(4) == 8);
    bool wrong_signature = false;
    try { find_op_capability<double, double>(BackendId::Cpu, OpCapability::Matmul, 1.0); }
    catch (const UserError& error) {
        wrong_signature = string(error.what()).find("signature mismatch") != string::npos;
    }
    assert(wrong_signature);
    bool duplicate = false;
    try {
        RegisterOpCapability<int, int> another(
            BackendId::Cpu, OpCapability::Matmul, "another_implementation");
    } catch (const UserError& error) {
        duplicate = string(error.what()).find("Duplicate") != string::npos;
    }
    assert(duplicate);
    definitions().erase(definition.name);
    assert(!find(1));
    assert(backend_supported_capabilities("cpu").empty());
    definitions()[definition.name] = definition;
    assert(find(4)(4) == 5);
}
''', encoding="utf-8")
    executable = tmp_path / "capability_contract"
    result = subprocess.run(
        [os.environ.get("CXX", "g++"), "-std=c++14", "-I", str(SRC),
         str(source), str(SRC / "ops/op_capability.cc"), "-o", str(executable)],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    subprocess.run([str(executable)], check=True, timeout=10)


def test_core_optional_replacements_do_not_name_implementing_libraries():
    paths = [SRC / "ops" / (name + "_op.cc") for name in
             ("arg_reduce", "argsort", "where", "random", "transpose")]
    paths += [SRC / "opt/tuner" / (name + "_tuner.cc") for name in ("conv", "matmul")]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert not re.search(r"\b(?:cub|cublas|cudnn|curand|cutt|mkl)_", source), path
        assert "find_op_capability<" in source, path


def test_every_optional_capability_is_owned_by_a_library():
    declarations = []
    for directory in (ROOT / "python/jittor/extern/cuda", ROOT / "python/jittor/extern/mkl"):
        for path in directory.rglob("*_capabilities.cc"):
            source = path.read_text(encoding="utf-8")
            assert "get_op_info(" not in source, path
            declarations += re.findall(r"OpCapability::(\w+)", source)
    assert set(declarations) == {
        "SegmentedArgReduce", "SegmentedArgsort", "Where", "Random", "Transpose",
        "Matmul", "Conv2d", "Conv2dBackwardInput", "Conv2dBackwardWeight",
    }
    assert len(declarations) == 13
