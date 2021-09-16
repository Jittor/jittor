// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <fstream>
#include <streambuf>
#include <stdlib.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <mutex>

#include "jit_compiler.h"
#include "op.h"
#include "utils/cache_compile.h"
#include "utils/flags.h"
#include "fused_op.h"

namespace jittor {
    
DEFINE_FLAG(string, jittor_path, "", "Source path of jittor");
DEFINE_FLAG(string, cc_path, "", "Path of C++ compiler");
DEFINE_FLAG(string, cc_type, "", "Type of C++ compiler(clang, icc, g++)");
DEFINE_FLAG(string, cc_flags, "", "Flags of C++ compiler");
DEFINE_FLAG(string, nvcc_path, "", "Path of CUDA C++ compiler");
DEFINE_FLAG(string, nvcc_flags, "", "Flags of CUDA C++ compiler");
DEFINE_FLAG(string, python_path, "", "Path of python interpreter");
DEFINE_FLAG(string, cache_path, "", "Cache path of jittor");
DEFINE_FLAG(int, rewrite_op, 1, "Rewrite source file of jit operator or not");

namespace jit_compiler {

std::mutex dl_open_mutex;

jit_op_entry_t load_jit_lib(string name, string symbol_name="jit_entry") {
    std::lock_guard<std::mutex> lock(dl_open_mutex);
    const char* msg = "";
    LOGvv << "Opening jit lib:" << name;
    #ifdef _WIN32
    void* handle = (void*)LoadLibraryExA(name.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
        LOAD_LIBRARY_SEARCH_USER_DIRS);
    #elif defined(__linux__) && !defined(mobile)
    void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    msg = dlerror();
    #else
    void *handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
    msg = dlerror();
    #endif
  
    CHECK(handle) << "Cannot open library" << name << ":" << msg;
  
    #ifdef _WIN32
    auto jit_entry = (jit_op_entry_t)GetProcAddress((HINSTANCE)handle, symbol_name.c_str());
    #else
    //dlerror();
    auto jit_entry = (jit_op_entry_t)dlsym(handle, symbol_name.c_str());
    msg = dlerror();
    #endif
    CHECK(jit_entry) << "Loading symbol" << symbol_name << "from" << name << "failed:" << msg;
    
    return jit_entry;
}

void run_cmd(string cmd, string cwd="") {
    if (cwd.size()) cmd = "cd '"+cwd + "' && " + cmd;
    LOGvvv << "Run cmd:" << cmd;
    system_with_check(cmd.c_str());
}

static string get_symbol_name(const string& jit_key) {
    int i=0;
    while (i<jit_key.size() && jit_key[i]!='[') i++;
    string op_name = i ? jit_key.substr(0, i) : "fused";
    op_name = Op::file_name_to_class_name(op_name);
    // _ZN7jittorXyyyyyy7jit_runEv
    // jittor::yyyyyy::jit_run
    #ifdef _MSC_VER
    op_name = "?jit_run@"+op_name+"Op@jittor@@QEAAXXZ";
    #else
    op_name = "_ZN6jittor"+S(op_name.size()+2)+op_name+"Op7jit_runEv";
    #endif
    return op_name;
}

jit_op_entry_t compile(const string& jit_key, const string& src, const bool is_cuda_op, const string& extra_flags) {
    LOGvv << "Compile op" << jit_key;
    // compiler do not allowed filename too long
    CHECK(cc_path.size());
    string jit_src_path = Op::get_filename_from_jit_key(jit_key, ".cc");
    #ifdef _WIN32
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".dll");
    #else
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".so");
    #endif
    string other_src;
    LOGvvv << "Generate" << jit_src_path >> "\n" >> src;
    if (rewrite_op || !file_exist(jit_src_path))
        write(jit_src_path, src);
    string cmd;
    
#ifndef _MSC_VER
    if (is_cuda_op) {
        cmd = "\"" + nvcc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + nvcc_flags + extra_flags
            + " -o \"" + jit_lib_path + "\"";
    } else {
#ifdef mobile
        cmd = cc_path
            + " '" + jit_src_path + "'" + other_src
            + cc_flags + extra_flags
            + " -Dmobile -L/data/data/com.example.mjittor/.cache/jittor/default/clang -L/data/data/com.example.mjittor/termux/lib -lpython3.9 -lomp -ljit_utils_core -ljittor_core -Wl,-rpath=/data/data/com.example.mjittor/.cache/jittor/default/clang/ -o '" + jit_lib_path + "'";
#else
        cmd = cc_path
            + " '" + jit_src_path + "'" + other_src
            + cc_flags + extra_flags
            + " -o '" + jit_lib_path + "'";
#endif
#if defined(__linux__) && !defined(mobile)
        cmd = python_path+" "+jittor_path+"/utils/asm_tuner.py "
            "--cc_path=" + cmd;
#endif
    }
#else // Windows _MSC_VER
    if (is_cuda_op) {
        cmd = "\"" + nvcc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + nvcc_flags + extra_flags
            + " -o \"" + jit_lib_path + "\"";
    } else {
        auto symbol_name = get_symbol_name(jit_key);
        auto pos = cc_flags.find("-link");
        auto cc_flags1 = cc_flags.substr(0, pos);
        auto cc_flags2 = cc_flags.substr(pos);
        cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + cc_flags1 + extra_flags
            + " -Fe: \"" + jit_lib_path + "\" " + cc_flags2 + " -EXPORT:\""
            + symbol_name + "\"";
    }
#endif
    cache_compile(cmd, cache_path, jittor_path);
    auto symbol_name = get_symbol_name(jit_key);
    auto jit_entry = load_jit_lib(jit_lib_path, symbol_name);
    return jit_entry;
}

} // jit_compiler
} // jittor