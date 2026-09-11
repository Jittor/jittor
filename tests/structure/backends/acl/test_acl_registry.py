"""Run the real erased ACL registry with typed host query/launcher stand-ins."""
import importlib.util
import os
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[4]


def test_registry_has_one_tu_owner_and_preserves_typed_query_arguments(tmp_path):
    script = ROOT / "agent/skills/acl-host-syntax-check/make_cann_stub.py"
    spec = importlib.util.spec_from_file_location("registry_sdk_stub", script)
    stubber = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stubber)
    stub = tmp_path / "sdk"
    stubber.build(ROOT / "backends/acl", stub)
    declarations = (stub / "acl/aclnn_entry_points.h").read_text()
    definitions = re.findall(r"^aclnnStatus (aclnn\w*)\(([^\n]*)\);$", declarations, re.M)
    sdk = tmp_path / "sdk.cc"
    sdk.write_text('#include "acl/aclnn_entry_points.h"\n' + '\n'.join(
        'aclnnStatus %s(%s) { return 0; }' % pair for pair in definitions))
    for suffix in ("a", "b"):
        (tmp_path / (suffix + ".cc")).write_text(
            '#include "acl_op_registry.h"\n'
            'const jittor::AclOpRegistry* from_%s() { return &jittor::acl_op_registry(); }\n' % suffix)
    main = tmp_path / "main.cc"
    main.write_text(r'''
#include "acl_op_registry.h"
#include <cassert>
struct aclTensor {};
struct aclScalar {};
struct aclOpExecutor {};
using namespace jittor;
const AclOpRegistry* from_a();
const AclOpRegistry* from_b();
aclTensor x, y, output;
aclScalar alpha;
aclOpExecutor executor;
int seen = 0;
aclnnStatus unary(const aclTensor* a, aclTensor* out, uint64_t* size, aclOpExecutor** exec) {
    assert(a == &x && out == &output); *size = 17; *exec = &executor; seen |= 1; return 0;
}
aclnnStatus cast(const aclTensor* a, aclDataType dtype, aclTensor* out,
                 uint64_t* size, aclOpExecutor** exec) {
    assert(dtype == ACL_INT32); seen |= 2; return unary(a, out, size, exec);
}
aclnnStatus binary(const aclTensor* a, const aclTensor* b, aclTensor* out,
                   uint64_t* size, aclOpExecutor** exec) {
    assert(b == &y); seen |= 4; return unary(a, out, size, exec);
}
aclnnStatus add(const aclTensor* a, const aclTensor* b, const aclScalar* scale,
                aclTensor* out, uint64_t* size, aclOpExecutor** exec) {
    assert(scale == &alpha); seen |= 8; return binary(a, b, out, size, exec);
}
aclnnStatus execute(void* workspace, uint64_t size, aclOpExecutor* exec, aclrtStream stream) {
    assert(!workspace && size == 17 && exec == &executor && !stream); return 29;
}
int main() {
    static_assert(sizeof(AclOpFunctions) <= 64, "registry entries must not regain per-op callable slots");
    assert(from_a() == from_b() && from_a() == &acl_op_registry());
    assert(acl_op_registry().size() == 121); // 123 rows, two identical duplicate keys (Floor, Sigmoid).
    assert(acl_op_registry().at("Abs").launcher() == &aclnnAbs);
    assert(acl_op_registry().at("Cast").supports(AclOpFunctions::QueryKind::Cast));
    assert(acl_op_registry().at("Add").supports(AclOpFunctions::QueryKind::Add));
    assert(acl_op_registry().at("Mul").supports(AclOpFunctions::QueryKind::Binary));
    assert(acl_op_registry().at("ReduceProd").launcher() == &aclnnProd);
    assert(!acl_op_registry().at("ReduceProd").has_grouped_query());
    AclWorkspaceArguments args;
    args.x = &x; args.y = &y; args.output = &output; args.alpha = &alpha; args.dtype = ACL_INT32;
    const AclOpFunctions entries[] = {
        AclOpFunctions::unary(unary, execute), AclOpFunctions::cast(cast, execute),
        AclOpFunctions::binary(binary, execute), AclOpFunctions::add(add, execute)};
    for (const auto& entry : entries) {
        uint64_t size = 0; aclOpExecutor* exec = nullptr;
        assert(entry.workspace(args, &size, &exec) == 0);
        assert(entry.launcher()(nullptr, size, exec, nullptr) == 29);
    }
    assert(seen == 15);
    bool rejected = false;
    try { uint64_t size; aclOpExecutor* exec;
        AclOpFunctions::direct(execute).workspace(args, &size, &exec);
    } catch (const std::logic_error&) { rejected = true; }
    assert(rejected);
    rejected = false;
    try { AclOpFunctions::direct(nullptr); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
''')
    executable = tmp_path / "registry"
    result = subprocess.run([
        os.environ.get("CXX", "g++"), "-std=c++14", "-pthread",
        "-I" + str(stub), "-I" + str(stub / "acl"), "-I" + str(ROOT / "src"),
        "-I" + str(ROOT / "backends/acl/include"), str(main), str(sdk),
        str(tmp_path / "a.cc"), str(tmp_path / "b.cc"),
        str(ROOT / "backends/acl/src/acl_jittor.cc"), "-o", str(executable),
    ], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
