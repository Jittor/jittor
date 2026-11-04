"""Topk's actual native query keeps k, tensor roles and shared launch intact."""
import os
import sys
from pathlib import Path
import subprocess

from _helpers.acl_recording_pipeline import build_pipeline


ROOT = Path(__file__).resolve().parents[4]


def test_topk_inference_assembles_native_owner_and_preserves_both_output_types(monkeypatch):
    load, Tensor, calls = build_pipeline(monkeypatch)
    module = load("topk_op")
    monkeypatch.setattr(sys.modules["jittor"].flags, "no_grad", 1)
    values, indices = module.TopkACL()(Tensor((2, 128), "float16"), 17)
    assert values.shape == indices.shape == (2, 17)
    assert values.dtype == "float16" and indices.dtype == "int64"
    assert calls[-1]["backend"] == "acl"
    assert "TopkOpRunner op;" in calls[-1]["cuda_src"]
    assert "aclnnTopk" not in calls[-1]["cuda_src"]
    assert not calls[-1].get("cuda_grad_src")


def test_real_topk_query_derives_k_from_shape_and_keeps_last_axis_sorted_largest(tmp_path):
    (tmp_path / "aclnnop").mkdir(); (tmp_path / "core").mkdir()
    (tmp_path / "core/var.h").write_text("#pragma once\n")
    (tmp_path / "aclnnop/aclnn_topk.h").write_text(r'''
#pragma once
#include <cstdint>
struct aclTensor {};
struct aclOpExecutor {};
using aclnnStatus = int;
using aclrtStream = void*;
aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor*, int64_t, int64_t, bool, bool,
    aclTensor*, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnTopk(void*, uint64_t, aclOpExecutor*, aclrtStream);
''')
    (tmp_path / "topk_op_acl.h").write_text(r'''
#pragma once
#include "aclnnop/aclnn_topk.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace jittor {
using std::string; using std::vector;
enum Dtype { ns_float16, ns_float32, ns_int64, ns_int32 };
struct Var { Dtype type; Dtype dtype() const { return type; } };
struct Check { bool ok; template<class T> Check& operator<<(const T&) { return *this; }
    ~Check() noexcept(false) { if (!ok) throw std::runtime_error("guard"); } };
#define USER_CHECK(value) Check{bool(value)}
extern string soc;
inline const char* acl_soc_name() { return soc.c_str(); }
using AclOpRegistry = std::unordered_map<string, int>;
extern int launches;
struct BaseOpRunner {
    string name;
    vector<vector<int64_t>> inputShapes, outputShapes;
    vector<aclTensor*> inputTensors, outputTensors;
    vector<Var*> in_, out_;
    int ret = -1; uint64_t workspaceSize = 0; aclOpExecutor* executor = nullptr;
    explicit BaseOpRunner(string value) : name(value) {}
    virtual ~BaseOpRunner() = default;
    virtual void executeOp(AclOpRegistry::const_iterator&) = 0;
    void launch(int status, decltype(&aclnnTopk) function, bool synchronize) {
        assert(status == 0 && synchronize); launches++;
        assert(function(nullptr, workspaceSize, executor, nullptr) == 0);
    }
};
struct TopkOpRunner : BaseOpRunner {
    TopkOpRunner();
    void executeOp(AclOpRegistry::const_iterator&) override;
};
}
''')
    main = tmp_path / "main.cc"
    main.write_text(r'''
#include "topk_op_acl.h"
namespace jittor { string soc = "Ascend310P3"; int launches = 0; }
aclTensor input, values, indices;
aclOpExecutor execution;
int queries = 0;
int64_t requested = 0;
aclnnStatus aclnnTopkGetWorkspaceSize(const aclTensor* x, int64_t k, int64_t dim,
        bool largest, bool sorted, aclTensor* val, aclTensor* ids,
        uint64_t* size, aclOpExecutor** executor) {
    assert(x == &input && val == &values && ids == &indices);
    assert(k == requested && dim == -1 && largest && sorted); queries++;
    *size = 17; *executor = &execution; return 0;
}
aclnnStatus aclnnTopk(void*, uint64_t size, aclOpExecutor* executor, aclrtStream) {
    assert(size == 17 && executor == &execution); return 0;
}
int main() {
    using namespace jittor;
    TopkOpRunner runner;
    Var x{ns_float16}, val{ns_float16}, ids{ns_int64};
    runner.inputShapes = {{2,128}};
    runner.inputTensors = {&input}; runner.outputTensors = {&values, &indices};
    runner.in_ = {&x}; runner.out_ = {&val, &ids};
    AclOpRegistry registry; auto it = registry.cbegin();
    for (int64_t k : {1,17,128}) {
        requested = k; runner.outputShapes = {{2,k}, {2,k}};
        runner.executeOp(it);
    }
    assert(queries == 3 && launches == 3);
    for (int failure = 0; failure < 5; failure++) {
        runner.outputShapes = {{2,17}, {2,17}}; ids.type = ns_int64;
        if (failure == 0) soc = "Ascend910B3"; else soc = "Ascend310P3";
        if (failure == 1) ids.type = ns_int32;
        if (failure == 2) runner.outputShapes[1][1] = 16;
        if (failure == 3) runner.outputShapes = {{2,0}, {2,0}};
        if (failure == 4) runner.outputShapes = {{2,129}, {2,129}};
        bool rejected = false;
        try { runner.executeOp(it); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && queries == 3 && launches == 3);
    }
}
''')
    executable = tmp_path / "canary"
    source = ROOT / "backends/acl/kernels/native/topk_op_acl.cc"
    build = subprocess.run([os.environ.get("CXX", "g++"), "-std=c++14", "-I" + str(tmp_path),
        str(source), str(main), "-o", str(executable)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    execution = subprocess.run([str(executable)], capture_output=True, text=True)
    assert execution.returncode == 0, execution.stdout + execution.stderr
