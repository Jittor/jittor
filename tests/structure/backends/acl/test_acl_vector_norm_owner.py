"""Vector-norm SDK query preserves order, axis, accumulation dtype and roles."""
import os
from pathlib import Path
import subprocess
import sys

from _helpers.acl_recording_pipeline import build_pipeline

ROOT = Path(__file__).resolve().parents[4]


def test_vector_norm_uses_standard_native_owner_without_model_or_generated_math(monkeypatch):
    load, Tensor, calls = build_pipeline(monkeypatch)
    module = load("vector_norm_op")
    monkeypatch.setattr(sys.modules["jittor"].flags, "no_grad", 1)
    for keepdim, shape in ((False, (2, 5)), (True, (2, 5, 1))):
        output = module.VectorNormACL()(Tensor((2, 5, 17), "float16"), keepdim)
        assert output.shape == shape and output.dtype == "float16"
        assert calls[-1]["backend"] == "acl"
        assert "LinalgVectorNormOpRunner op;" in calls[-1]["cuda_src"]
        assert "aclnnLinalgVectorNorm" not in calls[-1]["cuda_src"]
        assert not calls[-1].get("cuda_grad_src")


def test_real_query_keeps_order_axis_keepdim_dtype_and_tensor_roles(tmp_path):
    (tmp_path / "aclnnop").mkdir(); (tmp_path / "core").mkdir()
    (tmp_path / "core/var.h").write_text("#pragma once\n")
    (tmp_path / "aclnnop/aclnn_linalg_vector_norm.h").write_text(r'''
#pragma once
#include <cstdint>
struct aclTensor {}; struct aclOpExecutor {};
struct aclScalar { float value; }; struct aclIntArray { int64_t value; };
using aclnnStatus=int; using aclrtStream=void*; using aclDataType=int;
constexpr int ACL_FLOAT=0;
aclScalar* aclCreateScalar(const void*, aclDataType);
aclIntArray* aclCreateIntArray(const int64_t*, uint64_t);
void aclDestroyScalar(aclScalar*); void aclDestroyIntArray(aclIntArray*);
aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor*,const aclScalar*,const aclIntArray*,bool,aclDataType,aclTensor*,uint64_t*,aclOpExecutor**);
aclnnStatus aclnnLinalgVectorNorm(void*,uint64_t,aclOpExecutor*,aclrtStream);
''')
    (tmp_path / "vector_norm_op_acl.h").write_text(r'''
#pragma once
#include "aclnnop/aclnn_linalg_vector_norm.h"
#include <cassert>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
namespace jittor {
using std::string; using std::vector;
enum Dtype {ns_float16=1,ns_float32=2,ns_int32=3};
inline int get_dtype(Dtype value) {return int(value);}
struct Var { Dtype type; bool dense=true; Dtype dtype() const{return type;}
    bool is_contiguous() const{return dense;} };
struct Check {bool ok; template<class T> Check& operator<<(const T&){return *this;}
    ~Check() noexcept(false){if(!ok)throw std::runtime_error("guard");}};
#define USER_CHECK(value) Check{bool(value)}
extern string soc; inline const char* acl_soc_name(){return soc.c_str();}
using AclOpRegistry=std::unordered_map<string,int>;
struct BaseOpRunner {
    string name; vector<vector<int64_t>> inputShapes,outputShapes;
    vector<aclTensor*> inputTensors,outputTensors; vector<Var*> in_,out_;
    int ret=-1;uint64_t workspaceSize=0;aclOpExecutor* executor=nullptr;
    explicit BaseOpRunner(string name):name(name){}
    virtual ~BaseOpRunner()=default;
    virtual void executeOp(AclOpRegistry::const_iterator&)=0;
    void launch(int status,decltype(&aclnnLinalgVectorNorm) function,bool sync){
        assert(status==0 && sync);assert(function(nullptr,workspaceSize,executor,nullptr)==0);}
};
struct LinalgVectorNormOpRunner:BaseOpRunner {
    LinalgVectorNormOpRunner();void executeOp(AclOpRegistry::const_iterator&) override;};
}
''')
    main = tmp_path / "main.cc"
    main.write_text(r'''
#include "vector_norm_op_acl.h"
namespace jittor {string soc="Ascend310P3";}
aclTensor input,output;aclOpExecutor executor;
int dtype=0,queries=0,launches=0,destroyed=0;bool keep=false;
aclScalar* aclCreateScalar(const void* value,int type){assert(type==ACL_FLOAT);return new aclScalar{*static_cast<const float*>(value)};}
aclIntArray* aclCreateIntArray(const int64_t* value,uint64_t size){assert(size==1);return new aclIntArray{value[0]};}
void aclDestroyScalar(aclScalar* p){destroyed++;delete p;}
void aclDestroyIntArray(aclIntArray* p){destroyed++;delete p;}
aclnnStatus aclnnLinalgVectorNormGetWorkspaceSize(const aclTensor* x,const aclScalar* p,const aclIntArray* dims,bool keeping,int type,aclTensor* y,uint64_t* size,aclOpExecutor** exec){
    assert(x==&input && y==&output);assert(p->value==2 && dims->value==-1);
    assert(keeping==keep && type==dtype);queries++;*size=17;*exec=&executor;return 0;}
aclnnStatus aclnnLinalgVectorNorm(void*,uint64_t size,aclOpExecutor* exec,aclrtStream){assert(size==17 && exec==&executor);launches++;return 0;}
int main(){
    using namespace jittor;LinalgVectorNormOpRunner runner;
    Var x{ns_float16},y{ns_float16};runner.in_={&x};runner.out_={&y};
    runner.inputShapes={{2,5,17}};runner.inputTensors={&input};runner.outputTensors={&output};
    AclOpRegistry registry;auto it=registry.cbegin();
    for(auto type:{ns_float16,ns_float32})for(bool keeping:{false,true}){
        x.type=y.type=type;dtype=get_dtype(type);keep=keeping;
        runner.outputShapes={keeping?vector<int64_t>{2,5,1}:vector<int64_t>{2,5}};
        runner.executeOp(it);}
    assert(queries==4 && launches==4 && destroyed==8);
    soc="Ascend910B3";bool rejected=false;
    try{runner.executeOp(it);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && queries==4);
    soc="Ascend310P3";runner.outputShapes={{2,17}};rejected=false;
    try{runner.executeOp(it);}catch(const std::runtime_error&){rejected=true;}
    assert(rejected && queries==4);
}
''')
    executable = tmp_path / "query"
    source = ROOT / "backends/acl/kernels/native/vector_norm_op_acl.cc"
    result = subprocess.run([os.environ.get("CXX", "g++"), "-std=c++14", "-I"+str(tmp_path),
                             str(source), str(main), "-o", str(executable)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    result = subprocess.run([str(executable)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
