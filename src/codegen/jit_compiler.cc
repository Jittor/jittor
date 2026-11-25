// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <fstream>
#include <streambuf>
#include <stdlib.h>
#include <atomic>
#include <cerrno>
#include <cstring>
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <mutex>
#include <algorithm>

#include "codegen/jit_compiler.h"
#include "runtime/jit_policy.h"
#include "runtime/configuration.h"
#include "core/op.h"
#include "utils/cache_compile.h"
#include "utils/hash.h"
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
DEFINE_FLAG(int, use_jit_pch, 1, "Enable precompiled header for CPU JIT kernels");

static string g_core_abi_fingerprint;

void set_core_abi_fingerprint(const string& fingerprint) {
    check_startup_config_write("core ABI fingerprint");
    g_core_abi_fingerprint = fingerprint;
}

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

static string strip_quotes_and_trailing_slash(string s) {
    s = strip(s);
    if (s.size() >= 2 && ((s.front() == '"' && s.back() == '"') ||
                          (s.front() == '\'' && s.back() == '\''))) {
        s = s.substr(1, s.size() - 2);
    }
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

static string parent_dir_path(const string& p) {
    if (p.empty()) return "";
    auto pos = p.rfind('/');
    if (pos == string::npos || pos == 0) return "";
    return p.substr(0, pos);
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
            string raw_dir = f.substr(2);
            string rpath_arg = raw_dir;
            #ifdef __linux__
            string dir = strip_quotes_and_trailing_slash(raw_dir);
            string norm_cache = strip_quotes_and_trailing_slash(cache_path);
            string parent_cache = parent_dir_path(norm_cache);
            string custom_ops_cache = norm_cache.empty() ? string() : norm_cache + "/custom_ops";
            // JIT shared libraries live in <cache_path>/jit/*.so. Using $ORIGIN
            // relative rpaths makes every compiled JIT .so relocatable across
            // JITTOR_HOME directories and git worktrees without embedding
            // machine- or checkout-specific cache paths.
            if (!norm_cache.empty() && dir == norm_cache)
                rpath_arg = "'$ORIGIN/..'";
            else if (!parent_cache.empty() && dir == parent_cache)
                rpath_arg = "'$ORIGIN/../..'";
            else if (!custom_ops_cache.empty() && dir == custom_ops_cache)
                rpath_arg = "'$ORIGIN/../custom_ops'";
            #endif
            if (is_cuda)
                output.push_back(f+" -Xlinker -rpath="+rpath_arg);
            else
                output.push_back(f+" -Wl,-rpath,"+rpath_arg);
            #ifdef __APPLE__
            libpaths.push_back(raw_dir);
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

#if defined(__linux__) && !defined(_MSC_VER)
static std::atomic<uint64_t> g_tmp_seq{0};

static string unique_tmp_path(const string& target) {
    uint64_t seq = g_tmp_seq.fetch_add(1, std::memory_order_relaxed);
    return target + ".tmp." + std::to_string(getpid()) + "." + std::to_string(seq);
}

static bool mkdir_p(const string& dir) {
    if (dir.empty()) return false;
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
    size_t pos = 1;
    while (pos <= dir.size()) {
        auto slash = dir.find('/', pos);
        string sub = (slash == string::npos) ? dir : dir.substr(0, slash);
        if (!sub.empty()) {
            if (::mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
                if (stat(sub.c_str(), &st) != 0 || !S_ISDIR(st.st_mode))
                    return false;
            }
        }
        if (slash == string::npos) break;
        pos = slash + 1;
    }
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

static bool is_valid_elf_file(const string& path) {
    struct stat st;
    if (stat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 64)
        return false;
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) return false;
    unsigned char magic[4] = {0, 0, 0, 0};
    ifs.read(reinterpret_cast<char*>(magic), 4);
    return ifs.gcount() == 4 &&
           magic[0] == 0x7f && magic[1] == 'E' && magic[2] == 'L' && magic[3] == 'F';
}

static bool copy_file_bytes(const string& src_path, const string& dst_path, mode_t mode) {
    std::ifstream ifs(src_path, std::ios::binary);
    if (!ifs) return false;
    std::ofstream ofs(dst_path, std::ios::binary | std::ios::trunc);
    if (!ofs) return false;
    char buf[65536];
    while (ifs) {
        ifs.read(buf, sizeof(buf));
        std::streamsize n = ifs.gcount();
        if (n > 0) {
            ofs.write(buf, n);
            if (!ofs) {
                ofs.close();
                remove(dst_path.c_str());
                return false;
            }
        }
    }
    ofs.close();
    if (!ofs || ifs.bad()) {
        remove(dst_path.c_str());
        return false;
    }
    ::chmod(dst_path.c_str(), mode);
    return true;
}

static bool copy_file_atomically(const string& src_path, const string& dst_path) {
    string tmp = unique_tmp_path(dst_path);
    if (!copy_file_bytes(src_path, tmp, 0755)) {
        remove(tmp.c_str());
        return false;
    }
    if (::rename(tmp.c_str(), dst_path.c_str()) != 0) {
        remove(tmp.c_str());
        return false;
    }
    return true;
}

static string canonical_path_or_self(const string& path) {
    string cleaned = strip_quotes_and_trailing_slash(path);
    if (cleaned.empty()) return cleaned;
    char* resolved = ::realpath(cleaned.c_str(), nullptr);
    if (!resolved) return cleaned;
    string out(resolved);
    ::free(resolved);
    while (out.size() > 1 && out.back() == '/') out.pop_back();
    return out;
}

static bool path_is_within_dir(const string& path, const string& dir) {
    if (dir.empty() || path.empty()) return false;
    if (dir == "/") return path.front() == '/';
    if (path == dir) return true;
    return path.size() > dir.size() &&
           path.compare(0, dir.size(), dir) == 0 &&
           path[dir.size()] == '/';
}

static vector<string> extract_include_dirs(const string& cmd) {
    vector<string> dirs;
    auto tokens = shsplit(cmd);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const string& t = tokens[i];
        if (t == "-I" || t == "-isystem" || t == "-iquote") {
            if (i + 1 < tokens.size()) {
                string dir = canonical_path_or_self(tokens[++i]);
                if (!dir.empty()) dirs.push_back(dir);
            }
        } else if (startswith(t, "-I") && t.size() > 2) {
            string dir = canonical_path_or_self(t.substr(2));
            if (!dir.empty()) dirs.push_back(dir);
        } else if (startswith(t, "-isystem") && t.size() > 8) {
            string dir = canonical_path_or_self(t.substr(8));
            if (!dir.empty()) dirs.push_back(dir);
        } else if (startswith(t, "-iquote") && t.size() > 7) {
            string dir = canonical_path_or_self(t.substr(7));
            if (!dir.empty()) dirs.push_back(dir);
        }
    }
    return dirs;
}

static string replace_all_substr(string text, const string& from, const string& to) {
    if (from.empty() || from == to) return text;
    size_t pos = 0;
    while ((pos = text.find(from, pos)) != string::npos) {
        text.replace(pos, from.size(), to);
        pos += to.size();
    }
    return text;
}

static string repo_root_from_jittor_path(const string& jpath) {
    string norm = strip_quotes_and_trailing_slash(jpath);
    static const string suffix = "/python/jittor";
    if (norm.size() > suffix.size() &&
        norm.compare(norm.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return norm.substr(0, norm.size() - suffix.size());
    }
    return norm;
}

// ---------------------------------------------------------------------------
// CPU JIT Precompiled Header (.gch)
// ---------------------------------------------------------------------------

static const string& cpu_jit_pch_content() {
    static const string content =
        "#ifndef JITTOR_JIT_CPU_PCH_H\n"
        "#define JITTOR_JIT_CPU_PCH_H\n"
        "#include <assert.h>\n"
        "#include <cmath>\n"
        "#include <algorithm>\n"
        "#include <limits>\n"
        "#include \"core/common.h\"\n"
        "#include \"core/var.h\"\n"
        "#include \"core/op.h\"\n"
        "#include \"core/fused_op.h\"\n"
        "#include \"core/executor.h\"\n"
        "#include \"runtime/backend.h\"\n"
        "#include \"runtime/device.h\"\n"
        "#include \"runtime/dispatch_context.h\"\n"
        "#include \"type/cpu_math.h\"\n"
        "#include \"ops/op_register.h\"\n"
        "#include \"ops/layout_propagation.h\"\n"
        "#include \"ops/binary_op.h\"\n"
        "#include \"ops/unary_op.h\"\n"
        "#include \"ops/broadcast_to_op.h\"\n"
        "#include \"ops/reduce_op.h\"\n"
        "#include \"ops/reindex_op.h\"\n"
        "#include \"ops/reindex_reduce_op.h\"\n"
        "#include \"ops/ternary_op.h\"\n"
        "#include \"ops/composite/array_op.h\"\n"
        "#endif\n";
    return content;
}

static string strip_pch_link_flags(const string& flags, bool& has_werror, bool& incompatible) {
    has_werror = false;
    incompatible = false;
    string out;
    auto tokens = shsplit(flags);
    for (size_t i = 0; i < tokens.size(); ++i) {
        const string& t = tokens[i];
        if (t.empty()) continue;
        if (t == "-Werror" || startswith(t, "-Werror="))
            has_werror = true;
        if (t == "-include" || startswith(t, "-include") ||
            t == "-imacros" || startswith(t, "-imacros") ||
            t == "-x" || t == "-undef" ||
            t == "-nostdinc" || t == "-nostdinc++" ||
            t == "-E" || t == "-fsyntax-only") {
            incompatible = true;
            return "";
        }
        if (t == "-shared" || t == "-c" || t == "-rdynamic") continue;
        if (t == "-Xlinker" || t == "-L" || t == "-l") {
            if (i + 1 < tokens.size()) ++i;
            continue;
        }
        if (startswith(t, "-Wl,") || startswith(t, "-L") || startswith(t, "-l"))
            continue;
        if (!startswith(t, "-") &&
            (endswith(t, ".so") || endswith(t, ".a") || endswith(t, ".o") || endswith(t, ".lib")))
            continue;
        out += t;
        out += " ";
    }
    return out;
}

static bool src_is_pch_compatible(const string& src) {
    bool saw_jit = false;
    bool saw_jit_cpu = false;
    size_t pos = 0;
    while (pos < src.size()) {
        size_t end = src.find('\n', pos);
        if (end == string::npos) end = src.size();
        string line = strip(src.substr(pos, end - pos));
        pos = end + 1;
        if (line.empty() || startswith(line, "//")) continue;
        if (startswith(line, "#include")) break;
        if (startswith(line, "#define ") || startswith(line, "#undef ")) {
            // Collapse internal whitespace to verify exact macro definitions.
            string norm;
            bool in_ws = false;
            for (char c : line) {
                if (c == ' ' || c == '\t') {
                    if (!in_ws) { norm += ' '; in_ws = true; }
                } else {
                    norm += c;
                    in_ws = false;
                }
            }
            if (norm == "#define JIT 1") {
                saw_jit = true;
                continue;
            }
            if (norm == "#define JIT_cpu 1") {
                saw_jit_cpu = true;
                continue;
            }
            // Any other macro defined/undefined before the first #include
            // might alter header behavior relative to the precompiled header.
            return false;
        }
        // Reject any other preprocessor directive or non-comment preamble
        // before the first #include so PCH never reorders declarations.
        return false;
    }
    if (!saw_jit || !saw_jit_cpu) return false;
    if (src.find("#define JIT_cuda") != string::npos ||
        src.find("#undef JIT") != string::npos)
        return false;
    return true;
}

struct CpuPchState {
    std::mutex state_mutex;
    std::mutex build_mutex;
    string ready_sig;
    string failed_sig;
};

static CpuPchState& cpu_pch_state() {
    static CpuPchState* s = new CpuPchState();
    return *s;
}

static bool compute_cpu_pch_spec(string& pch_h, string& pch_gch, string& pch_cmd,
                                 string& sig, bool& has_werror) {
    if (!use_jit_pch || cache_path.empty() || cc_path.empty()) return false;
    // GCC .gch auto-discovery via -include is GCC-specific.
    if (cc_type != "g++") return false;
    bool incompatible = false;
    string pch_flags = strip_pch_link_flags(cc_flags, has_werror, incompatible);
    if (incompatible) return false;
    pch_h = join(join(cache_path, "gen"), "jit_cpu_pch.h");
    pch_gch = pch_h + ".gch";
    pch_cmd = "\"" + cc_path + "\" -x c++-header \"" + pch_h + "\" "
        + pch_flags + " -DJIT=1 -DJIT_cpu=1 -o \"" + pch_gch + "\"";
    sig = pch_cmd + "\n" + cpu_jit_pch_content();
    return true;
}

// Fast non-building check: returns true iff a matching, up-to-date .gch is
// already on disk. Never invokes the compiler or takes the build lock.
static bool try_get_existing_cpu_jit_pch(string& pch_h, bool& has_werror) {
    string pch_gch, pch_cmd, sig;
    if (!compute_cpu_pch_spec(pch_h, pch_gch, pch_cmd, sig, has_werror))
        return false;
    auto& st = cpu_pch_state();
    {
        std::lock_guard<std::mutex> lock(st.state_mutex);
        if (st.ready_sig == sig && file_exist(pch_h) && file_exist(pch_gch) && file_exist(pch_gch + ".key"))
            return true;
        if (st.failed_sig == sig)
            return false;
    }
    if (!file_exist(pch_gch) || !file_exist(pch_gch + ".key") || !file_exist(pch_h))
        return false;
    try {
        const string& content = cpu_jit_pch_content();
        if (read_all(pch_h) != content) return false;
        auto probe = cache_compile_probe(pch_cmd, pch_h, content);
        if (probe.up_to_date) {
            std::lock_guard<std::mutex> lock(st.state_mutex);
            st.ready_sig = sig;
            return true;
        }
    } catch (const std::exception&) {
    }
    return false;
}

// Build (or verify) the CPU JIT PCH lazily, thread-safely, and atomically.
// Falls back silently on any failure so compilation never breaks.
static bool ensure_cpu_jit_pch(string& pch_h, bool& has_werror) {
    if (try_get_existing_cpu_jit_pch(pch_h, has_werror))
        return true;
    string pch_gch, pch_cmd, sig;
    if (!compute_cpu_pch_spec(pch_h, pch_gch, pch_cmd, sig, has_werror))
        return false;
    auto& st = cpu_pch_state();
    std::lock_guard<std::mutex> build_lock(st.build_mutex);
    {
        std::lock_guard<std::mutex> lock(st.state_mutex);
        if (st.ready_sig == sig && file_exist(pch_h) && file_exist(pch_gch) && file_exist(pch_gch + ".key"))
            return true;
        if (st.failed_sig == sig)
            return false;
    }
    try {
        jittor::lock_guard lg;
        const string& content = cpu_jit_pch_content();
        mkdir_p(parent_dir_path(pch_h));
        if (!file_exist(pch_h) || read_all(pch_h) != content)
            write_atomically(pch_h, content);
        cache_compile(pch_cmd, cache_path, jittor_path);
        if (file_exist(pch_gch) && file_exist(pch_gch + ".key")) {
            std::lock_guard<std::mutex> lock(st.state_mutex);
            st.ready_sig = sig;
            return true;
        }
    } catch (const std::exception& e) {
        LOGv << "CPU JIT PCH build failed, falling back to non-PCH:" << e.what();
    }
    {
        std::lock_guard<std::mutex> lock(st.state_mutex);
        st.failed_sig = sig;
    }
    return false;
}

static string build_cpu_pch_cmd(const string& pch_h, bool has_werror,
                                const string& jit_src_path,
                                const string& jit_lib_path,
                                const string& extra_flags) {
    return "\"" + cc_path + "\""
        + " -DJIT=1 -DJIT_cpu=1"
        + (has_werror ? "" : " -Winvalid-pch")
        + " -include \"" + pch_h + "\""
        + " \"" + jit_src_path + "\""
        + fix_cl_flags(cc_flags + extra_flags, false)
        + " -o \"" + jit_lib_path + "\"";
}

// ---------------------------------------------------------------------------
// Cross-Worktree Content-Addressed Shared JIT Cache (JITTOR_SHARED_CACHE)
// ---------------------------------------------------------------------------

static string get_shared_cache_root() {
    const char* env = getenv("JITTOR_SHARED_CACHE");
    if (!env) return "";
    return strip_quotes_and_trailing_slash(string(env));
}

static bool extra_flags_safe_for_shared_cache(const string& extra_flags) {
    if (strip(extra_flags).empty()) return true;
    for (const auto& t : shsplit(extra_flags)) {
        if (t.empty()) continue;
        if (startswith(t, "-I") || startswith(t, "-L") || startswith(t, "-l") ||
            startswith(t, "-isystem") || startswith(t, "-iquote") ||
            startswith(t, "-include") || startswith(t, "-imacros") ||
            startswith(t, "-Wl,") || startswith(t, "-Xlinker") ||
            startswith(t, "-B"))
            return false;
        if (!startswith(t, "-") &&
            (endswith(t, ".o") || endswith(t, ".a") || endswith(t, ".so") || endswith(t, ".lib")))
            return false;
    }
    return true;
}

static string compiler_binary_fingerprint(const string& compiler_path) {
    struct stat st;
    if (stat(compiler_path.c_str(), &st) != 0)
        return compiler_path + ":missing";
    return compiler_path + ":" + std::to_string((int64)st.st_size) + ":"
        + std::to_string((int64)st.st_mtim.tv_sec) + "."
        + std::to_string((int64)st.st_mtim.tv_nsec);
}

static string normalize_src_for_cas(const string& src, const string& repo_root) {
    if (repo_root.empty() || src.find(repo_root) == string::npos)
        return src;
    string out;
    out.reserve(src.size());
    size_t pos = 0;
    while (pos < src.size()) {
        size_t end = src.find('\n', pos);
        bool has_nl = (end != string::npos);
        if (!has_nl) end = src.size();
        string line = src.substr(pos, end - pos);
        size_t first = line.find_first_not_of(" \t");
        if (first != string::npos && startswith(line, "#line ", first)) {
            line = replace_all_substr(line, repo_root, "<REPO_ROOT>");
        }
        out += line;
        if (has_nl) out += '\n';
        pos = has_nl ? end + 1 : src.size();
    }
    return out;
}

static string normalize_cmd_for_cas(string cmd,
                                    const string& jit_src_path,
                                    const string& jit_lib_path,
                                    const string& norm_cache,
                                    const string& parent_cache,
                                    const string& norm_jittor,
                                    const string& repo_root) {
    vector<pair<string, string>> subs = {
        {jit_src_path, "<JIT_SRC>"},
        {jit_lib_path, "<JIT_OUT>"},
        {norm_cache, "<CACHE_PATH>"},
        {parent_cache, "<PARENT_CACHE_PATH>"},
        {norm_jittor, "<JITTOR_PATH>"},
        {repo_root, "<REPO_ROOT>"},
    };
    std::sort(subs.begin(), subs.end(),
              [](const pair<string, string>& a, const pair<string, string>& b) {
                  return a.first.size() > b.first.size();
              });
    for (const auto& kv : subs) {
        if (!kv.first.empty())
            cmd = replace_all_substr(cmd, kv.first, kv.second);
    }
    return cmd;
}

static string escape_make_dep_path(const string& p) {
    string out;
    out.reserve(p.size());
    for (char c : p) {
        if (c == ' ' || c == '\\') out += '\\';
        out += c;
    }
    return out;
}

struct SharedCacheEntry {
    bool enabled = false;
    string so_path;
    string deps_path;
    string norm_cache;
    string repo_root;
    string compiler_exe;
    vector<string> allowed_include_dirs;
};

static SharedCacheEntry prepare_shared_cache_entry(
    const string& base_cmd, const string& src, bool is_cuda_op,
    bool device_link, const string& extra_flags, const string& symbol_name,
    const string& jit_src_path, const string& jit_lib_path) {
    SharedCacheEntry entry;
    string root = get_shared_cache_root();
    if (root.empty() || g_core_abi_fingerprint.empty() || device_link)
        return entry;
    if (!extra_flags_safe_for_shared_cache(extra_flags))
        return entry;
    entry.norm_cache = strip_quotes_and_trailing_slash(cache_path);
    string parent_cache = parent_dir_path(entry.norm_cache);
    string norm_jittor = strip_quotes_and_trailing_slash(jittor_path);
    entry.repo_root = repo_root_from_jittor_path(norm_jittor);
    const string& compiler_exe = is_cuda_op ? accelerator_compiler.path : cc_path;
    entry.compiler_exe = compiler_exe;
    entry.allowed_include_dirs = extract_include_dirs(base_cmd);
    string norm_cmd = normalize_cmd_for_cas(
        base_cmd, jit_src_path, jit_lib_path,
        entry.norm_cache, parent_cache, norm_jittor, entry.repo_root);
    string norm_src = normalize_src_for_cas(src, entry.repo_root);
    string cas_input = "jittor-shared-jit-v1\n";
    cas_input += "abi:" + g_core_abi_fingerprint + "\n";
    cas_input += "cuda:" + string(is_cuda_op ? "1" : "0") + "\n";
    cas_input += "symbol:" + symbol_name + "\n";
    cas_input += "compiler:" + compiler_binary_fingerprint(compiler_exe) + "\n";
    cas_input += "cmd:" + norm_cmd + "\n";
    cas_input += "src:" + content_hash(norm_src) + "\n";
    string h = content_hash(cas_input);
    string abi_dir = g_core_abi_fingerprint.substr(0, std::min<size_t>(16, g_core_abi_fingerprint.size()));
    string bucket = join(join(join(root, "v1"), abi_dir), h.substr(0, 2));
    entry.so_path = join(bucket, h + ".so");
    entry.deps_path = join(bucket, h + ".deps");
    entry.enabled = true;
    return entry;
}

static void try_publish_to_shared_cache(
    const SharedCacheEntry& entry, const string& jit_lib_path,
    const string& jit_src_path, bool overwrite_existing) {
    if (!entry.enabled || !is_valid_elf_file(jit_lib_path)) return;
    if (!overwrite_existing && is_valid_elf_file(entry.so_path)) return;
    try {
        string depfile = jit_lib_path + ".d";
        // Require a depfile so we can prove no external mutable user header
        // was pulled in (e.g. via #include "/tmp/my_header.h").
        if (!file_exist(depfile)) return;
        auto deps = read_depfile_paths(depfile);
        string canon_src = canonical_path_or_self(jit_src_path);
        string canon_repo = canonical_path_or_self(entry.repo_root);
        string canon_cache = canonical_path_or_self(entry.norm_cache);
        string canon_cache_gen = canon_cache.empty() ? string() : canon_cache + "/gen";
        string canon_compiler = canonical_path_or_self(entry.compiler_exe);
        string compiler_prefix = parent_dir_path(parent_dir_path(canon_compiler));
        if (compiler_prefix == "/") compiler_prefix.clear();

        string normalized_deps;
        for (const auto& d : deps) {
            if (d == jit_src_path || d.find("jit_cpu_pch") != string::npos)
                continue;
            // Ignore pseudo-paths emitted from synthetic #line directives.
            if (!file_exist(d))
                continue;
            string canon_d = canonical_path_or_self(d);
            if (canon_d == canon_src)
                continue;
            if (!canon_repo.empty() && path_is_within_dir(canon_d, canon_repo)) {
                normalized_deps += "<REPO_ROOT>" + canon_d.substr(canon_repo.size()) + "\n";
                continue;
            }
            if (!canon_cache_gen.empty() && path_is_within_dir(canon_d, canon_cache_gen)) {
                normalized_deps += "<CACHE_PATH>" + canon_d.substr(canon_cache.size()) + "\n";
                continue;
            }
            if (path_is_within_dir(canon_d, "/usr") ||
                path_is_within_dir(canon_d, "/lib") ||
                path_is_within_dir(canon_d, "/lib64")) {
                continue;
            }
            if (!compiler_prefix.empty() && path_is_within_dir(canon_d, compiler_prefix)) {
                continue;
            }
            bool in_allowed_inc = false;
            for (const auto& inc_dir : entry.allowed_include_dirs) {
                if (inc_dir == canon_cache) continue;
                if (path_is_within_dir(canon_d, inc_dir)) {
                    in_allowed_inc = true;
                    break;
                }
            }
            if (in_allowed_inc) continue;
            // External user header outside repo, cache/gen, system, compiler,
            // and standard -I dirs: skip publishing so edits cannot serve stale .so.
            LOGvv << "Shared JIT cache publish skipped for external header:" << canon_d;
            return;
        }
        if (!mkdir_p(parent_dir_path(entry.so_path))) return;
        // Publish normalized repo-relative deps first so readers observing
        // the .so always find a complete .deps companion.
        string tmp_deps = unique_tmp_path(entry.deps_path);
        {
            std::ofstream ofs(tmp_deps, std::ios::binary | std::ios::trunc);
            if (!ofs) return;
            ofs.write(normalized_deps.data(), normalized_deps.size());
            ofs.close();
            if (!ofs) {
                remove(tmp_deps.c_str());
                return;
            }
            if (::rename(tmp_deps.c_str(), entry.deps_path.c_str()) != 0) {
                remove(tmp_deps.c_str());
                return;
            }
        }
        copy_file_atomically(jit_lib_path, entry.so_path);
    } catch (const std::exception& e) {
        LOGv << "Shared JIT cache publish skipped:" << e.what();
    }
}

static bool try_stage_from_shared_cache(
    const SharedCacheEntry& entry, const string& jit_src_path,
    const string& jit_lib_path, const string& effective_src,
    bool would_write_source) {
    if (!entry.enabled || !is_valid_elf_file(entry.so_path))
        return false;
    try {
        if (!copy_file_atomically(entry.so_path, jit_lib_path))
            return false;
        if (would_write_source)
            write_atomically(jit_src_path, effective_src);
        // Materialize local .d naming the generated source, core_abi.stamp
        // (which tracks all core/backend headers), and any mapped local deps.
        // Local .key is intentionally committed only after load_jit_lib() succeeds.
        vector<string> local_deps;
        local_deps.push_back(jit_src_path);
        string abi_stamp = join(entry.norm_cache, "core_abi.stamp");
        if (file_exist(abi_stamp))
            local_deps.push_back(abi_stamp);
        if (file_exist(entry.deps_path)) {
            string raw = read_all(entry.deps_path);
            size_t pos = 0;
            while (pos < raw.size()) {
                size_t end = raw.find('\n', pos);
                if (end == string::npos) end = raw.size();
                string line = raw.substr(pos, end - pos);
                pos = end + 1;
                if (startswith(line, "<REPO_ROOT>/") && !entry.repo_root.empty()) {
                    string mapped = entry.repo_root + line.substr(11);
                    if (file_exist(mapped)) local_deps.push_back(mapped);
                } else if (startswith(line, "<CACHE_PATH>/") && !entry.norm_cache.empty()) {
                    string mapped = entry.norm_cache + line.substr(12);
                    if (file_exist(mapped)) local_deps.push_back(mapped);
                }
            }
        }
        string dep_text = escape_make_dep_path(jit_lib_path) + ":";
        for (const auto& d : local_deps) {
            dep_text += " ";
            dep_text += escape_make_dep_path(d);
        }
        dep_text += "\n";
        write_atomically(jit_lib_path + ".d", dep_text);
        return true;
    } catch (const std::exception& e) {
        LOGv << "Shared JIT cache stage failed, falling back to local compile:" << e.what();
        remove(jit_lib_path.c_str());
        remove((jit_lib_path + ".d").c_str());
        return false;
    }
}
#endif

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
    string base_cmd;
    // The preparation key captures policy before asynchronous compilation.
    // Extension-local flags remain explicit per-op overrides.
    
    auto symbol_name = get_symbol_name(jit_key);
#ifndef _MSC_VER
    if (is_cuda_op) {
        base_cmd = "\"" + accelerator_compiler.path + "\""
            + " \"" + jit_src_path + "\""
            + fix_cl_flags(kernel_flags, accelerator_compiler.language == "cuda")
            + " -o \"" + jit_lib_path + "\"";
        if (device_link) {
            base_cmd = python_path+" "+jittor_path+"/build/dlink_compiler.py " + base_cmd;
        }
    } else {
        base_cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\""
            + fix_cl_flags(cc_flags + extra_flags, is_cuda_op)
            + " -o \"" + jit_lib_path + "\"";
    }
#else // Windows _MSC_VER
    if (is_cuda_op) {
        base_cmd = "\"" + accelerator_compiler.path + "\""
            + " \"" + jit_src_path + "\""
            + kernel_flags
            + " -o \"" + jit_lib_path + "\""
            +  " -Xlinker -EXPORT:\""
            + symbol_name + "\"";
    } else {
        base_cmd = "\"" + cc_path + "\""
            + " \"" + jit_src_path + "\""
            + " -Fe: \"" + jit_lib_path + "\" "
            + fix_cl_flags(cc_flags + extra_flags, is_cuda_op) + " -EXPORT:\""
            + symbol_name + "\"";
    }
#endif

    auto try_unlocked_load = [&](const string& probe_cmd) -> jit_op_entry_t {
        try {
            auto probe = jit_compiler::cache_compile_probe(
                probe_cmd, would_write_source ? jit_src_path : string(), src);
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
        return nullptr;
    };

    // A warm cache is a read, and a read does not need the build lock.
    // Check an already-built PCH command first (without building PCH yet) and
    // also the plain command (which covers non-PCH and shared-cache hits).
#if defined(__linux__) && !defined(_MSC_VER)
    // When rewrite_op=0 and the source file already exists on disk, respect
    // the on-disk content for both PCH eligibility and shared-cache CAS keys.
    string disk_src_buf;
    if (!would_write_source)
        disk_src_buf = read_all(jit_src_path2);
    const string& effective_src = would_write_source ? src : disk_src_buf;
    const bool cpu_pch_eligible = !is_cuda_op && use_jit_pch &&
        strip(extra_flags).empty() && src_is_pch_compatible(effective_src);
    if (cpu_pch_eligible) {
        string existing_pch_h;
        bool has_werror = false;
        if (try_get_existing_cpu_jit_pch(existing_pch_h, has_werror)) {
            string pch_cmd = build_cpu_pch_cmd(
                existing_pch_h, has_werror, jit_src_path, jit_lib_path, extra_flags);
            if (auto entry = try_unlocked_load(pch_cmd))
                return entry;
        }
    }
#endif
    if (auto entry = try_unlocked_load(base_cmd))
        return entry;

    // Something has to be built (or fetched from JITTOR_SHARED_CACHE). Take the
    // lock, re-check locally, then check JITTOR_SHARED_CACHE before compiling.
    jittor::lock_guard lg;
#if defined(__linux__) && !defined(_MSC_VER)
    auto shared_entry = prepare_shared_cache_entry(
        base_cmd, effective_src, is_cuda_op, device_link, extra_flags,
        symbol_name, jit_src_path, jit_lib_path);
    bool shared_fetch_failed = false;
    if (shared_entry.enabled) {
        // Re-check local cache under lock (both PCH and plain commands) in case
        // another process just built this op while we waited for the lock.
        if (cpu_pch_eligible) {
            string existing_pch_h;
            bool has_werror = false;
            if (try_get_existing_cpu_jit_pch(existing_pch_h, has_werror)) {
                string pch_cmd = build_cpu_pch_cmd(
                    existing_pch_h, has_werror, jit_src_path, jit_lib_path, extra_flags);
                try {
                    if (cache_compile_probe(pch_cmd, would_write_source ? jit_src_path : string(), src).up_to_date)
                        return load_jit_lib(jit_lib_path, symbol_name, extra_flags);
                } catch (const std::exception&) {}
            }
        }
        try {
            if (cache_compile_probe(base_cmd, would_write_source ? jit_src_path : string(), src).up_to_date)
                return load_jit_lib(jit_lib_path, symbol_name, extra_flags);
        } catch (const std::exception&) {}
        if (try_stage_from_shared_cache(
                shared_entry, jit_src_path, jit_lib_path, effective_src, would_write_source)) {
            try {
                auto jit_entry = load_jit_lib(jit_lib_path, symbol_name, extra_flags);
                // Commit .key only after dlopen+dlsym succeeded so concurrent
                // unlocked readers never observe an unverified .so and local
                // fallback recompilation is never skipped.
                auto post_probe = cache_compile_probe(base_cmd, "", "");
                write_atomically(post_probe.key_name, post_probe.cache_key);
                LOGvv << "Shared JIT cache hit:" << jit_key;
                return jit_entry;
            } catch (const std::exception& e) {
                LOGv << "Shared JIT cache library failed to load, recompiling locally:" << e.what();
                shared_fetch_failed = true;
                remove(jit_lib_path.c_str());
                remove((jit_lib_path + ".d").c_str());
                remove((jit_lib_path + ".key").c_str());
            }
        }
    }
    string cmd = base_cmd;
    if (cpu_pch_eligible) {
        string pch_h;
        bool has_werror = false;
        if (ensure_cpu_jit_pch(pch_h, has_werror)) {
            cmd = build_cpu_pch_cmd(pch_h, has_werror, jit_src_path, jit_lib_path, extra_flags);
        }
    }
#else
    string cmd = base_cmd;
#endif
    if (would_write_source)
        write(jit_src_path2, src);
    cache_compile(cmd, cache_path, jittor_path);
    auto jit_entry = load_jit_lib(jit_lib_path, symbol_name, extra_flags);
#if defined(__linux__) && !defined(_MSC_VER)
    if (shared_entry.enabled)
        try_publish_to_shared_cache(shared_entry, jit_lib_path, jit_src_path, shared_fetch_failed);
#endif
    return jit_entry;
}

} // jit_compiler
} // jittor
