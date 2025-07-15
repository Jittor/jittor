// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "op.h"
#include "acl_jittor.h"
#include "utils/str_utils.h"
#include <chrono>
#include <thread>

namespace jittor {

uint64_t acl_jittor_tid;
int acl_jittor_thread_running=0;
aclrtContext acl_jittor_context;
aclrtStream aclstream;

#define CHECK_ACL(x) ASSERTop(x,==,0)

static void* acl_jittor_process_callback(void*) {
    acl_jittor_thread_running = 1;
    int deviceId = 0;
    CHECK_ACL(aclrtSetCurrentContext(acl_jittor_context));
    
    while (acl_jittor_thread_running) {
        // LOGir << "acl_jittor_process_callback";
        auto ret = aclrtProcessReport(1000);
        if (ret) {
            if (acl_jittor_thread_running && ret != ACL_ERROR_RT_REPORT_TIMEOUT && ret != ACL_ERROR_RT_THREAD_SUBSCRIBE)
                LOGir << "aclrtProcessReport:" << ret << acl_error_to_string(ret);
            break;
        }
    }
    acl_jittor_thread_running = 0;
    return (void*)0;
}

// void aaa(void*) {
//     LOGir << "haha";
// }

struct acl_jittor_initer {

acl_jittor_initer() {
    CHECK_ACL(aclInit(nullptr));
    uint device_count = 0;
    // 获取可用的Device数量
    CHECK_ACL(aclrtGetDeviceCount(&device_count));
    LOGi << "Found ACL device number:" << device_count;
    CHECK_ACL(aclrtSetDevice(0));
    CHECK_ACL(aclrtCreateContext(&acl_jittor_context, 0));
    CHECK_ACL(aclrtSetCurrentContext(acl_jittor_context));
    
    pthread_create(&acl_jittor_tid, nullptr, acl_jittor_process_callback, 0);

    // subscribe for default stream
    CHECK_ACL(aclrtSubscribeReport(acl_jittor_tid,0));

    // simple callback test
    CHECK_ACL(aclrtCreateStream(&aclstream));
    // CHECK_ACL(aclrtSubscribeReport(acl_jittor_tid,aclstream));
    // CHECK_ACL(aclrtLaunchCallback((aclrtCallback)&aaa, 0, ACL_CALLBACK_NO_BLOCK, aclstream));
    // CHECK_ACL(aclrtLaunchCallback((aclrtCallback)&aaa, 0, ACL_CALLBACK_NO_BLOCK, 0));
}

~acl_jittor_initer() {
    acl_jittor_thread_running = 0;
    CHECK_ACL(aclrtUnSubscribeReport(acl_jittor_tid,0));
    CHECK_ACL(aclrtDestroyContext(acl_jittor_context));
    CHECK_ACL(aclFinalize());
}

} _acl_jittor_initer;

string process_acl(const string& src, const string& name, const map<string,string>& kargs) {
    if (endswith(name, "_jittor.cc"))
        return src;
    // static vector<string> dont_compile = {"fp16_emu.cc"};
    // for (auto& s : dont_compile)
    //     if (endswith(name, s))
    //         return " ";
    static unordered_set<string> cuda_headers = {
        "cuda_runtime", "cudnn", "driver_types",
        "cuda_fp16", "cuda_runtime_api", "fp16_emu",
        "cudnn_rnn_descriptor", "cublas_v2", "cublas_wrapper",
        "curand", "curand_wrapper", "cufft", "cufftXt",
        "CudaUtils", "cutt", "cudnn_wrapper", "cuda_bf16"
    };
    static unordered_set<string> fake_class = {
        "cudnnHandle_t", "cudnnConvolutionBwdFilterAlgo_t",
        "cudnnConvolutionBwdDataAlgo_t", "cudnnConvolutionFwdAlgo_t",
        "cufftHandle"
    };
    try {
    auto tokens = token_split(src);
    int edit = 0;
    for (int i=0; i<tokens.size(); i++) {
        auto& token = tokens[i];
        if (cuda_headers.count(token)) token = "acl_jittor", edit ++; else
        if (fake_class.count(token)) token = "int", edit ++; else
        if (token == "CUDA") token = "ACL", edit ++; else
        if (startswith(token, "cuda")) {
            if (token.size()>=5 && token[4] >= 'A' && token[4] <= 'Z') {
                if (token == "cudaGetDeviceCount") {
                    token_replace(tokens, i, "($1);", "((uint*)$1);");
                } else if (token == "cudaLaunchHostFunc") {
                    // ACL_CALLBACK_BLOCK for 310
                    token_replace(tokens, i, "LaunchHostFunc($1,$2,$3)",
                        "LaunchCallback($2,$3,ACL_CALLBACK_NO_BLOCK,$1)");
                } else if (token == "cudaMemcpy")
                    token_replace(tokens, i, "cudaMemcpy($1,$2,$3,",
                        "aclrtMemcpy($1,$3,$2,$3,");
                else if (token == "cudaMemcpyAsync")
                    token_replace(tokens, i, "cudaMemcpyAsync($1,$2,$3,",
                        "aclrtMemcpyAsync($1,$3,$2,$3,");
                else if (token == "cudaMemcpyDeviceToHost") token = "ACL_MEMCPY_DEVICE_TO_HOST";
                else if (token == "cudaMemcpyDefault") token = "ACL_MEMCPY_HOST_TO_DEVICE";
                else if (token == "cudaMemcpyHostToDevice") token = "ACL_MEMCPY_HOST_TO_DEVICE";
                else if (token == "cudaMemcpyDeviceToDevice") token = "ACL_MEMCPY_DEVICE_TO_DEVICE";
                else if (token == "cudaMallocManaged" || token == "cudaMalloc") {
                    // unified address not supported
                    token = "aclrtMalloc";
                    token_replace(tokens, i, "($1,$2)",
                        "($1,$2,ACL_MEM_MALLOC_HUGE_FIRST)");
                } else if (token == "cudaMemGetInfo")
                    token_replace(tokens, i, "cudaMemGetInfo($1,$2)",
                        "aclrtGetMemInfo(ACL_DDR_MEM,$1,$2)");
                else if (token == "cudaGetLastError")
                    token_replace(tokens, i, "cudaGetLastError()", "0");
                else if (token == "cudaStreamCreateWithFlags")
                    token_replace(tokens, i-1, 
                        "(cudaStreamCreateWithFlags($1,$2));",
                        "(aclrtCreateStream($1)); checkAclErrors(aclrtSubscribeReport(acl_jittor_tid,*$1));");
                else if (token == "cudaEventCreate")
                    token_replace(tokens, i, 
                        "cudaEventCreate($1,$2)",
                        "aclrtCreateEvent($1)");
                else if (token == "cudaDeviceSynchronize")
                    token = "aclrtSynchronizeDevice";
                else if (token == "cudaStreamDestroy")
                    token_replace(tokens, i, "cudaStreamDestroy($1)",
                        "(aclrtUnSubscribeReport(acl_jittor_tid,$1), aclrtDestroyStream($1))");
                else if (token == "cudaEventDestroy")
                    token = "aclrtDestroyEvent";
                else if (token == "cudaEventRecord")
                    token = "aclrtRecordEvent";
                else if (token == "cudaStreamWaitEvent")
                    token_replace(tokens, i, 
                        "cudaStreamWaitEvent($1,$2,$3)",
                        "aclrtStreamWaitEvent($1,$2)");
                
                if (token.size() && token[0] == 'c')
                    token = "aclrt" + token.substr(4);
                if (endswith(token, "_t"))
                    token = token.substr(0, token.size()-2);
                edit ++;
            }
        } else
        if (token == "_cudaGetErrorEnum") {
            token_replace(tokens, i, "_cudaGetErrorEnum($1)", "(acl_error_to_string($1))");
            edit ++;
        } else
        if (token == "checkCudaErrors")
            token = "checkAclErrors";
        else if (token == "JPU") {
            edit ++;
            string new_code;
            if (tokens[i+2] == "op_compiler")
                token_replace(tokens, i, 
                    "JPU(op_compiler($1,$2,$3))",
                    "acl_jittor_op_compiler($1,$2,$3)");
            else if (tokens[i+2] == "header")
                new_code = "#include \"acl_jittor.h\"";
            if (new_code.size())
                token_replace(tokens, i,  "JPU($1)", new_code);
        } else if (token == "use_cuda_managed_allocator" && tokens[i+1][0]==',') {
            tokens[i+2] = "0"; // disable unified address
        }
    }
    if (!edit) return src;
    string new_src = join(tokens, "");
    // if (name == "executor.cc") {
    //     new_src = string("#include <Python.h>\n#include <pystate.h>\n#include <common.h>\n")+
    //     "namespace jittor { void acl_op_exec(Op*); }\n" + 
    //     replace(new_src, "op->do_run_after_prepare(jkl);", 
    //     R"({
    //         acl_op_exec(op);
    //     })");
    // }
    if (name == "profiler.cc") {
        new_src = token_replace_all(new_src, ".cc", ".tikcc");
    }
    // LOGir << name << (name == "pass_manager.cc");
    if (name == "pass_manager.cc") {
        LOGir << "replace" << name;
        new_src = token_replace_all(new_src, "run_pass<FloatAtomicFixPass>();", "WTF");
    }
    // ????????
    return new_src;
    } catch (const std::exception& e) {
        LOGe << "process acl error:" << e.what();
        LOGe << "name:" << name;
        throw;
    }
}

void acl_jittor_op_compiler(string& filename, string& src, bool is_acl, string& extra_flags) {
    if (!is_acl) return;
    // extra_flags += " --tik-soc-version=Ascend910 ";
    // filename = replace(filename, ".cc", ".tikcc");
    // LOGir << filename;
    string new_src = process_acl(src, "", {});
    new_src = replace(new_src, R"(#include "misc/cuda_atomic.h")", "");
    new_src = replace(new_src, R"(#include "misc/cuda_limits.h")", "");
    new_src = replace(new_src, "__global__", "__ai_device_entry__");
    new_src = token_replace_all(new_src, "__launch_bounds__($1)", "");
    new_src = token_replace_all(new_src, "int thread_num = $1;", "int thread_num = 1;");
    new_src = token_replace_all(new_src, "tn0=std::max(tn0, $1);", "");
    new_src = token_replace_all(new_src, "<<<$1>>>", "<<<1,0>>>");
    new_src = token_replace_all(new_src, "int thread_id = $1;", "int thread_id = 1;");
    // for inc error
    new_src = token_replace_all(new_src, "for ($1+=$2)", "for ($1++)");
    // bit op error
    new_src = token_replace_all(new_src, "int tnum$1;", "");
    new_src = token_replace_all(new_src, "int p1$1;", "");
    new_src = token_replace_all(new_src, "int p2$1;", "");
    new_src = token_replace_all(new_src, "int tn$1=$2;", "int tn$1=0;");
    new_src = token_replace_all(new_src, "int tid$1=$2;", "int tid$1=0;");
    src = new_src;

    new_src = token_replace_all(new_src, "atomicAdd(&$1,$2);", "$1=$1+$2;");
    // new_src = token_replace_all(new_src, "bool", "int8");
    new_src = token_replace_all(new_src, "::numeric_min<float32>()", "-1e30");
    new_src = token_replace_all(new_src, "::numeric_max<float32>()", "1e30");
    // TODO: support max
    unordered_map<string,string> opmap = {
        // {"::max","tikcc::scalar_max"}, 
        {"::sqrtf", "tikcc::scalar_sqrt"}
    };
    auto ss = split(new_src, ";");
    for (auto &s : ss) {
        if (s.find("?") != string::npos) {
            s = token_replace_all(s+";", "auto $1=$2?$3:$4;", "auto $1=$3;if (!($2)) $1=$4;");
        }
        if (s.find("::max") != string::npos) {
            if (s.find("auto") == string::npos) {
                s = token_replace_all(s+";", " $1=$4::max($2,$3);", " $1=$2;if ($2 < $3) $1=$3;");
            } else {
                s = token_replace_all(s+";", "auto $1=$4::max($2,$3);", "auto $1=$2;if ($2 < $3) $1=$3;");
            }
        }
        for (auto& kv : opmap) {
            if (s.find(kv.first) != string::npos) {
                if (s.find("auto") == string::npos) {
                    // $1 = op($2) --> op($1, $2)
                    s = token_replace_all(s+";", " $1= "+kv.first+"($2);", kv.second+"($1, $2);");
                } else {
                    // auto $1 = op($2) --> float32 $1; op($1, $2);
                    s = token_replace_all(s+";", "auto $1= "+kv.first+"($2);", "float32 $1; " + kv.second+"($1, $2);");
                }
            }
        }
        // s = token_replace_all(s+";", "auto $1=$2?$3:$4;", "auto $1=$3;if (!($2)) $1=$4;");
        // s = token_replace_all(s+";", "auto $1=$2?$3:$4;", "auto $1=$3;if (!($2)) $1=$4;");
        // if (s.find("::max") != string::npos) {
        //     s = token_replace_all(s+";", " $1= ::max($2);", "tikcc::scalar_max($1, $2);");
        // }
    }
    new_src = join(ss, ";");
    src = new_src;
}

}
