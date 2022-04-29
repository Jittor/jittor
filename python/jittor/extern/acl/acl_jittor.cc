// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "acl_jittor.h"
#include "utils/str_utils.h"
#include <chrono>
#include <thread>

namespace jittor {

uint64_t acl_jittor_tid;
int acl_jittor_thread_running=0;
aclrtContext acl_jittor_context;

#define CHECK_ACL(x) ASSERTop(x,==,0)

static void* acl_jittor_process_callback(void*) {
    acl_jittor_thread_running = 1;
    int deviceId = 0;
    CHECK_ACL(aclrtSetCurrentContext(acl_jittor_context));
    
    while (acl_jittor_thread_running) {
        // LOGir << "acl_jittor_process_callback";
        auto ret = aclrtProcessReport(1000);
        if (ret) {
            if (acl_jittor_thread_running && ret != ACL_ERROR_RT_REPORT_TIMEOUT)
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
    // aclrtStream stream;
    // CHECK_ACL(aclrtCreateStream(&stream));
    // CHECK_ACL(aclrtSubscribeReport(acl_jittor_tid,stream));
    // CHECK_ACL(aclrtLaunchCallback((aclrtCallback)&aaa, 0, ACL_CALLBACK_NO_BLOCK, stream));
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
    auto tokens = token_split(src);
    int edit = 0;
    for (int i=0; i<tokens.size(); i++) {
        auto& token = tokens[i];
        if (token == "cuda_runtime") token = "acl_jittor", edit ++; else
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
    return join(tokens, "");
}

void acl_jittor_op_compiler(string& filename, string& src, bool is_acl, string& extra_flags) {
    if (!is_acl) return;
    extra_flags += " --tik-soc-version=Ascend910 ";
    filename = replace(filename, ".cc", ".tikcc");
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
    new_src = token_replace_all(new_src, "::max($1,$2);", "($1)>($2)?($1):($2);");
    // new_src = replace(new_src, "::max", "fmax");
    src = new_src;
    // auto tokens = token_split(new_src);
}

}
