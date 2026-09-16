// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
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
#include <algorithm>

#include "codegen/jit_compiler.h"
#include "runtime/jit_policy.h"
#include "runtime/configuration.h"
#include "core/op.h"
#include "utils/cache_compile.h"
#include "runtime/lock.h"
#include "utils/flags.h"
#include "core/fused_op.h"
#include "utils/str_utils.h"

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

struct AcceleratorCompilerSpec {
    string path, flags, language, source_suffix;
    vector<string> remove_flags;
    bool device_link = false;
    bool configured = false;
};

static AcceleratorCompilerSpec accelerator_compiler;

void configure_accelerator_compiler(const string& path, const string& flags,
        const string& language, const string& source_suffix,
        const vector<string>& remove_flags, bool device_link) {
    check_startup_config_write("accelerator compiler");
    USER_CHECK(!path.empty()) << "Accelerator compiler path must not be empty";
    USER_CHECK(language == "cuda" || language == "hip" || language == "cxx")
        << "Accelerator compiler language must be cuda, hip, or cxx";
    USER_CHECK(source_suffix == ".cc" || source_suffix == ".cpp"
        || source_suffix == ".cxx" || source_suffix == ".cu" || source_suffix == ".hip")
        << "Unsupported accelerator source suffix" << source_suffix;
    USER_CHECK(!device_link || language == "cuda")
        << "The device-link wrapper requires the CUDA compiler interface";
    for (const auto& flag : remove_flags)
        USER_CHECK(!flag.empty() && flag.front() == '-' && flag.find_first_of(" \t\r\n") == string::npos)
            << "Compiler flag filters must name individual option tokens";
    AcceleratorCompilerSpec next{path, flags, language, source_suffix, remove_flags, device_link, true};
    accelerator_compiler = move(next);
}

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

static string accelerator_flags_for_key(const string& key, const string& extra_flags) {
    USER_CHECK(accelerator_compiler.configured) << "Accelerator compiler is not configured";
    const auto& spec = accelerator_compiler;
    const auto base = spec.language == "cxx" ? spec.flags : cuda_math_flags_for_key(spec.flags, key);
    string result = " ";
    for (auto token : shsplit(base + " " + extra_flags)) {
        if (std::find(spec.remove_flags.begin(), spec.remove_flags.end(), token) != spec.remove_flags.end())
            continue;
        if (token == "--device-c") token = "-dc";
        // The captured strict-math policy uses the same driver vocabulary as
        // the CUDA frontend. HIP's Clang driver consumes equivalent options.
        if (spec.language == "hip") {
            if (token == "--fmad=false") token = "-ffp-contract=off";
            else if (token == "--prec-div=true" || token == "--prec-sqrt=true") token = "-fno-fast-math";
            else if (token == "--use_fast_math") token = "-ffast-math";
        }
        result += token + " ";
    }
    return result;
}

static bool requests_device_link(const string& flags) {
    for (const auto& token : shsplit(flags))
        if (token == "-dc" || token == "--device-c") return true;
    return false;
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
        if (startswith(f, "-l:")) {
            output.push_back(f);
        }
        else if (startswith(f, "-l") &&
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
    // The sanitizer runtime refuses to load a library opened with
    // RTLD_DEEPBIND ("incompatible with sanitizer runtime", sanitizers#611), so
    // an instrumented run needs it off: JITTOR_NO_DEEPBIND=1. Off by default,
    // because the flag also decides which definition of a symbol the operator
    // library binds to.
    int deepbind = getenv("JITTOR_NO_DEEPBIND") ? 0 : RTLD_DEEPBIND;
    auto flags = RTLD_LAZY | deepbind | RTLD_LOCAL;
    if (extra_flags.find("GLOBAL_VAR") != string::npos)
        flags = RTLD_LAZY | deepbind | RTLD_GLOBAL;
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
    const string kernel_flags = is_cuda_op ? accelerator_flags_for_key(jit_key, extra_flags) : string();
    const bool device_link = is_cuda_op && requests_device_link(kernel_flags);
    USER_CHECK(!device_link || accelerator_compiler.device_link)
        << "Selected accelerator compiler does not support device-link requests";
    const string suffix = is_cuda_op
        ? (device_link ? ".cu" : accelerator_compiler.source_suffix) : ".cc";
    string jit_src_path = Op::get_filename_from_jit_key(jit_key, suffix);
    #ifdef _WIN32
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".dll");
    string jit_src_path2 = _to_winstr(jit_src_path);
    #else
    string jit_lib_path = Op::get_filename_from_jit_key(jit_key, ".so");
    string& jit_src_path2 = jit_src_path;
    #endif
    LOGvvv << "Generate" << jit_src_path >> "\n" >> src;
    const bool would_write_source = rewrite_op || !file_exist(jit_src_path2);
    string cmd;
    // The preparation key captures policy before asynchronous compilation.
    // Extension-local flags remain explicit per-op overrides.
    
    auto symbol_name = get_symbol_name(jit_key);
#ifndef _MSC_VER
    if (is_cuda_op) {
        cmd = "\"" + accelerator_compiler.path + "\""
            + " \"" + jit_src_path + "\""
            + fix_cl_flags(kernel_flags, accelerator_compiler.language == "cuda")
            + " -o \"" + jit_lib_path + "\"";
        if (device_link) {
            cmd = python_path+" "+jittor_path+"/build/dlink_compiler.py " + cmd;
        }
    } else {
        cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\""
            + fix_cl_flags(cc_flags + extra_flags, is_cuda_op)
            + " -o \"" + jit_lib_path + "\"";
    }
#else // Windows _MSC_VER
    if (is_cuda_op) {
        cmd = "\"" + accelerator_compiler.path + "\""
            + " \"" + jit_src_path + "\""
            + kernel_flags
            + " -o \"" + jit_lib_path + "\""
            +  " -Xlinker -EXPORT:\""
            + symbol_name + "\"";
    } else {
        cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\""
            + " -Fe: \"" + jit_lib_path + "\" "
            + fix_cl_flags(cc_flags + extra_flags, is_cuda_op) + " -EXPORT:\""
            + symbol_name + "\"";
    }
#endif
    // A warm cache is a read, and a read does not need the build lock.
    //
    // Everything a hit does is compare the key this command would produce
    // against the key recorded next to the product, then dlopen the product.
    // That is safe to do unlocked because of the order in which a build
    // commits: cache_compile() renames the finished product into place first
    // and the .key second, and rename() is atomic within a directory. So a
    // reader that sees the matching key is looking at a directory entry that
    // already points at the complete product of that very build -- there is no
    // state in which the key has been published and the product has not.
    //
    // The generated source is the one input that only exists in memory here.
    // Writing it is a build step, not a read, so the fast path does not do it;
    // it hashes the string instead, which is exactly what hashing the file
    // that string would become gives. If the recorded key agrees with that,
    // the product on disk was built from this very source.
    //
    // Two things can still change under a reader: another process can publish
    // a *newer* build between the key check and the dlopen, and a product can
    // be unreadable for reasons the key cannot see. The first is closed by
    // re-reading the key after the library is mapped -- an unchanged key means
    // no build committed in between, because a build always republishes it.
    // The second is why the whole attempt is inside a try: any failure just
    // falls through to the locked path, which rebuilds and reports properly.
    try {
        auto probe = jit_compiler::cache_compile_probe(
            cmd, would_write_source ? jit_src_path : string(), src);
        if (probe.up_to_date) {
            auto jit_entry = load_jit_lib(jit_lib_path, symbol_name, extra_flags);
            if (read_all(probe.key_name) == probe.cache_key) {
                LOGvv << "Cached op, no build lock taken:" << jit_key;
                return jit_entry;
            }
            LOGvv << "Cache key of" << jit_lib_path
                << "changed while loading it, rebuilding under the lock";
        }
    } catch (const std::exception& e) {
        LOGvv << "Unlocked cache hit for" << jit_lib_path
            << "did not hold, falling back to the locked path:" << e.what();
    }

    // Something has to be built. Take the lock, and let cache_compile() decide
    // again now that it holds it: another process may have published this very
    // product while we waited, in which case it does nothing but the key
    // comparison.
    jittor::lock_guard lg;
    if (would_write_source)
        write(jit_src_path2, src);
    cache_compile(cmd, cache_path, jittor_path);
    auto jit_entry = load_jit_lib(jit_lib_path, symbol_name, extra_flags);
    return jit_entry;
}

} // jit_compiler
} // jittor
