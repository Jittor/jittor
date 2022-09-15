// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
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
#include "utils/str_utils.h"
JPU(header)

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

vector<string> shsplit(const string& s) {
    auto s1 = split(s, " ");
    vector<string> s2;
    int count = 0;
    for (auto& s : s1) {
        int nc = 0;
        for (auto& c : s)
            nc += c=='"' || c=='\'';
        if (count&1) {
            count += nc;
            s2.back() += " ";
            s2.back() += s;
        } else {
            count = nc;
            s2.push_back(s);
        }
    }
    return s2;
}

string fix_cl_flags(const string& cmd, bool is_cuda) {
#ifdef _MSC_VER
    auto flags = shsplit(cmd);
    vector<string> output, output2;
    
    for (auto& f : flags) {
        if (startswith(f, "-link"))
            continue;
        else if (startswith(f, "-l"))
            output2.push_back(f.substr(2)+".lib");
        else if (startswith(f, "-LIB"))
            output2.push_back(f);
        else if (startswith(f, "-LD"))
            output.push_back(f);
        else if (startswith(f, "-L"))
            output2.push_back("-LIBPATH:"+f.substr(2));
        else if (f.find(".lib") != string::npos)
            output2.push_back(f);
        else if (startswith(f, "-DEF:"))
            output2.push_back(f);
        else if (startswith(f, "-W") || startswith(f,"-f"))
            continue;
        else if (startswith(f,"-std="))
            output.push_back("-std:"+f.substr(5));
        else if (startswith(f,"-include"))
            output.push_back("-FI");
        else if (startswith(f,"-shared"))
            output.push_back("-LD");
        else
            output.push_back(f);
    }
    string cmdx = "";
    for (auto& s : output) {
        cmdx += s;
        cmdx += " ";
    }
    cmdx += "-link ";
    for (auto& s : output2) {
        cmdx += s;
        cmdx += " ";
    }
    return cmdx;
#else
    auto flags = shsplit(cmd);
    vector<string> output;
    #ifdef __APPLE__
    vector<string> libpaths;
    #endif
    
    for (auto& f : flags) {
        if (startswith(f, "-l") && 
            (f.find("cpython") != string::npos ||
             f.find("lib") != string::npos)) {
            #ifdef __APPLE__
            auto fname = f.substr(2) + ".so";
            int i;
            for (i=libpaths.size()-1; i>=0; i--) {
                auto full = libpaths[i] + '/' + fname;
                string full2;
                for (auto c : full)
                    if (c != '\"') full2 += c;
                if (jit_compiler::file_exist(full2)) {
                    output.push_back(full2);
                    break;
                }
            }
            if (i<0) output.push_back(f);
            #else
            output.push_back("-l:"+f.substr(2)+".so");
            #endif
        }
        else if (startswith(f, "-L")) {
            if (is_cuda)
                output.push_back(f+" -Xlinker -rpath="+f.substr(2));
            else
                output.push_back(f+" -Wl,-rpath,"+f.substr(2));
            #ifdef __APPLE__
            libpaths.push_back(f.substr(2));
            #endif
        } else
            output.push_back(f);
    }
    string cmdx = "";
    for (auto& s : output) {
        cmdx += s;
        cmdx += " ";
    }
    return cmdx;
#endif
}

namespace jit_compiler {

std::mutex dl_open_mutex;

jit_op_entry_t load_jit_lib(
    string name, string symbol_name="jit_entry", const string& extra_flags="") {
    std::lock_guard<std::mutex> lock(dl_open_mutex);
    const char* msg = "";
    LOGvv << "Opening jit lib:" << name;
    #ifdef _WIN32
    void* handle = (void*)LoadLibraryExA(_to_winstr(name).c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DEFAULT_DIRS |
        LOAD_LIBRARY_SEARCH_USER_DIRS);
    #elif defined(__linux__)
    auto flags = RTLD_LAZY | RTLD_DEEPBIND | RTLD_LOCAL;
    if (extra_flags.find("GLOBAL_VAR") != string::npos)
        flags = RTLD_LAZY | RTLD_DEEPBIND | RTLD_GLOBAL;
    void* handle = dlopen(name.c_str(), flags);
    msg = dlerror();
    #else
    auto flags = RTLD_LAZY | RTLD_LOCAL;
    if (extra_flags.find("GLOBAL_VAR") != string::npos)
        flags = RTLD_LAZY | RTLD_GLOBAL;
    void *handle = dlopen(name.c_str(), flags);
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
    while (i<jit_key.size() && jit_key[i]>=0 && jit_key[i]<=127) i++;
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
    string jit_src_path;
    if(is_cuda_op && extra_flags.find("-dc") != string::npos)
        jit_src_path = Op::get_filename_from_jit_key(jit_key, ".cu");
    else
        jit_src_path = Op::get_filename_from_jit_key(jit_key, ".cc");
    string* src2 = (string*)&src;
    string* extra_flags2 = (string*)&extra_flags;
    JPU(op_compiler(jit_src_path, *src2, is_cuda_op, *extra_flags2));
    #ifdef _WIN32
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".dll");
    string jit_src_path2 = _to_winstr(jit_src_path);
    #else
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".so");
    string& jit_src_path2 = jit_src_path;
    #endif
    string other_src;
    LOGvvv << "Generate" << jit_src_path >> "\n" >> src;
    if (rewrite_op || !file_exist(jit_src_path2))
        write(jit_src_path2, src);
    string cmd;
    
    auto symbol_name = get_symbol_name(jit_key);
#ifndef _MSC_VER
    if (is_cuda_op) {
        cmd = "\"" + nvcc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + fix_cl_flags(nvcc_flags + extra_flags, is_cuda_op)
            + " -o \"" + jit_lib_path + "\"";
        if (cmd.find("-dc") != string::npos) {
            cmd = python_path+" "+jittor_path+"/utils/dlink_compiler.py " + cmd;
        }
    } else {
        cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + fix_cl_flags(cc_flags + extra_flags, is_cuda_op)
            + " -o \"" + jit_lib_path + "\"";
#ifdef __linux__
        cmd = python_path+" "+jittor_path+"/utils/asm_tuner.py "
            "--cc_path=" + cmd;
#endif
    }
#else // Windows _MSC_VER
    if (is_cuda_op) {
        cmd = "\"" + nvcc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + nvcc_flags + extra_flags
            + " -o \"" + jit_lib_path + "\""
            +  " -Xlinker -EXPORT:\""
            + symbol_name + "\"";
    } else {
        cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\"" + other_src
            + " -Fe: \"" + jit_lib_path + "\" "
            + fix_cl_flags(cc_flags + extra_flags, is_cuda_op) + " -EXPORT:\""
            + symbol_name + "\"";
    }
#endif
    cache_compile(cmd, cache_path, jittor_path);
    auto jit_entry = load_jit_lib(jit_lib_path, symbol_name, extra_flags);
    return jit_entry;
}

} // jit_compiler
} // jittor