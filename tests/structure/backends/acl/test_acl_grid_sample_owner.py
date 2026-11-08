"""Real grid-sampler query body uses typed flags and shared launch handling."""
import os
import sys
from pathlib import Path
import subprocess

from _helpers.acl_recording_pipeline import build_pipeline


ROOT = Path(__file__).resolve().parents[4]


def test_grid_sampling_inference_uses_native_owner_without_generated_cpp(monkeypatch):
    load, Tensor, calls = build_pipeline(monkeypatch)
    module = load("grid_sample_op")
    monkeypatch.setattr(sys.modules["jittor"].flags, "no_grad", 1)
    output = module.GridSampler2DACL()(Tensor((8, 32, 80, 80), "float16"), Tensor((8, 300, 4, 2), "float16"))
    assert output.shape == (8, 32, 300, 4) and output.dtype == "float16"
    assert calls[-1]["backend"] == "acl"
    assert "GridSampler2DOpRunner op;" in calls[-1]["cuda_src"]
    assert "aclnnGridSampler2D" not in calls[-1]["cuda_src"]
    assert not calls[-1].get("cuda_grad_src")


def test_real_native_grid_query_preserves_mode_padding_align_and_tensor_roles(tmp_path):
    (tmp_path / "aclnnop").mkdir()
    (tmp_path / "core").mkdir()
    (tmp_path / "core/var.h").write_text("#pragma once\n")
    (tmp_path / "aclnnop/aclnn_grid_sampler2d.h").write_text(r'''
#pragma once
#include <cstdint>
struct aclTensor {};
struct aclOpExecutor {};
using aclnnStatus = int;
using aclrtStream = void*;
aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor*, const aclTensor*, int64_t, int64_t,
    bool, aclTensor*, uint64_t*, aclOpExecutor**);
aclnnStatus aclnnGridSampler2D(void*, uint64_t, aclOpExecutor*, aclrtStream);
''')
    (tmp_path / "grid_sample_op_acl.h").write_text(r'''
#pragma once
#include "aclnnop/aclnn_grid_sampler2d.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace jittor {
using std::string;
using std::vector;
enum Dtype { ns_float16, ns_float32, ns_int32 };
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
    int ret = -1;
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;
    explicit BaseOpRunner(string value) : name(value) {}
    virtual ~BaseOpRunner() = default;
    virtual void executeOp(AclOpRegistry::const_iterator&) = 0;
    void launch(int status, decltype(&aclnnGridSampler2D) function, bool synchronize) {
        assert(status == 0 && synchronize); launches++;
        assert(function(nullptr, workspaceSize, executor, nullptr) == 0);
    }
};
struct GridSampler2DOpRunner : BaseOpRunner {
    GridSampler2DOpRunner();
    void executeOp(AclOpRegistry::const_iterator&) override;
};
}
''')
    main = tmp_path / "main.cc"
    main.write_text(r'''
#include "grid_sample_op_acl.h"
namespace jittor { string soc = "Ascend310P3"; int launches = 0; }
aclTensor x, grid, output;
aclOpExecutor execution;
int queries = 0;
aclnnStatus aclnnGridSampler2DGetWorkspaceSize(const aclTensor* a, const aclTensor* b,
        int64_t mode, int64_t padding, bool align, aclTensor* out,
        uint64_t* size, aclOpExecutor** executor) {
    assert(a == &x && b == &grid && out == &output);
    assert(mode == 0 && padding == 0 && !align); queries++;
    *size = 17; *executor = &execution; return 0;
}
aclnnStatus aclnnGridSampler2D(void*, uint64_t size, aclOpExecutor* executor, aclrtStream) {
    assert(size == 17 && executor == &execution); return 0;
}
int main() {
    using namespace jittor;
    GridSampler2DOpRunner runner;
    Var a{ns_float16}, b{ns_float16}, out{ns_float16};
    runner.inputShapes = {{8,32,80,80}, {8,300,4,2}};
    runner.outputShapes = {{8,32,300,4}};
    runner.inputTensors = {&x, &grid}; runner.outputTensors = {&output};
    runner.in_ = {&a, &b}; runner.out_ = {&out};
    AclOpRegistry registry; auto it = registry.cbegin();
    runner.executeOp(it); assert(queries == 1 && launches == 1);
    for (int failure = 0; failure < 4; failure++) {
        if (failure == 0) soc = "Ascend910B3";
        if (failure == 1) { soc = "Ascend310P3"; b.type = ns_float32; }
        if (failure == 2) { b.type = ns_float16; runner.inputShapes[1][3] = 3; }
        if (failure == 3) { runner.inputShapes[1][3] = 2; runner.outputShapes[0][1] = 31; }
        bool rejected = false;
        try { runner.executeOp(it); } catch (const std::runtime_error&) { rejected = true; }
        assert(rejected && queries == 1 && launches == 1);
    }
}
''')
    executable = tmp_path / "canary"
    source = ROOT / "backends/acl/kernels/native/grid_sample_op_acl.cc"
    build = subprocess.run([os.environ.get("CXX", "g++"), "-std=c++14", "-I" + str(tmp_path),
        str(source), str(main), "-o", str(executable)], capture_output=True, text=True)
    assert build.returncode == 0, build.stdout + build.stderr
    execution = subprocess.run([str(executable)], capture_output=True, text=True)
    assert execution.returncode == 0, execution.stdout + execution.stderr
