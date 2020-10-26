// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <fstream>
#include <streambuf>
#include <stdlib.h>
#include <dlfcn.h>

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

jit_op_entry_t load_jit_lib(string name, string symbol_name="jit_entry") {
    LOGvv << "Opening jit lib:" << name;
    // void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_DEEPBIND | RTLD_LOCAL);
    // RTLD_DEEPBIND and openmp cause segfault
    void* handle = dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL | RTLD_DEEPBIND);
    CHECK(handle) << "Cannot open library" << name << ":" << dlerror();
    
    //dlerror();
    auto jit_entry = (jit_op_entry_t)dlsym(handle, symbol_name.c_str());
    const char* dlsym_error = dlerror();
    CHECK(!dlsym_error) << "Loading symbol jit_entry from" << name << "failed:" << dlsym_error;
    
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
    op_name = "_ZN6jittor"+S(op_name.size()+2)+op_name+"Op7jit_runEv";
    return op_name;
}

jit_op_entry_t compile(const string& jit_key, const string& src, const bool is_cuda_op, const string& extra_flags) {
    LOGvv << "Compile op" << jit_key;
    // compiler do not allowed filename too long
    CHECK(cc_path.size());
    string jit_src_path = Op::get_filename_from_jit_key(jit_key, ".cc");
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".so");
    string other_src = " "+join(jittor_path, "src/op.cc")+" "+
        join(jittor_path, "src/var.cc")+" ";
    other_src = "";
    LOGvvv << "Generate" << jit_src_path >> "\n" >> src;
    if (rewrite_op || !file_exist(jit_src_path))
        write(jit_src_path, src);
    string cmd;
    if (is_cuda_op) {
        cmd = nvcc_path 
            + " '" + jit_src_path + "'" + other_src
            + nvcc_flags + extra_flags
            + " -o '" + jit_lib_path + "'";
    } else {
        cmd = cc_path
            + " '" + jit_src_path + "'" + other_src
            + cc_flags + extra_flags
            + " -o '" + jit_lib_path + "'";
        cmd = python_path+" "+jittor_path+"/utils/asm_tuner.py "
            "--cc_path=" + cmd;
    }
    cache_compile(cmd, cache_path, jittor_path);
    auto symbol_name = get_symbol_name(jit_key);
    auto jit_entry = load_jit_lib(jit_lib_path, symbol_name);
    return jit_entry;
}

} // jit_compiler
} // jittor