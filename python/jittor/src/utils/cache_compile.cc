// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <fstream>
#include <streambuf>
#ifdef _WIN32
#include <filesystem>
#include <process.h>
#include <windows.h>
#endif
#include <stdio.h>
#include <string.h>
#include <errno.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "misc/hash.h"
#include "utils/cache_compile.h"
#include "utils/str_utils.h"

namespace jittor {
namespace jit_compiler {

#ifndef TEST
string read_all(const string& fname) {
    std::ifstream ifs(fname);
    if (ifs && ifs.good())
        return string((std::istreambuf_iterator<char>(ifs)),
                      (std::istreambuf_iterator<char>()));
    return "";
}

void write(const string& fname, const string& src) {
    std::ofstream(fname) << src;
}

bool file_exist(const string& fname) {
    std::ifstream f(fname);
    return f && f.good();
}
#endif

string join(string a, string b) {
    const char sep = '/';
    if (!b.empty() && b.front() == sep) return b;
    a.reserve(a.size() + b.size() + 1);
    if (!a.empty() && a.back() != sep) a += sep;
    a += b;
    return a;
}

void find_names(string cmd, vector<string>& input_names, string& output_name, map<string,vector<string>>& extra) {
    // find space not in str
    #define is_quate(x) ((x)=='\'' || (x)=='\"')
    auto pass = [&](size_t& j) {
        while (j<cmd.size()) {
            if (is_quate(cmd[j])) {
                j++;
                while (j<cmd.size() && !is_quate(cmd[j])) j++;
                ASSERT(j<cmd.size());
                j++;
                continue;
            }
            while (j<cmd.size() && cmd[j]!=' ' && !is_quate(cmd[j])) j++;
            if (j<cmd.size()) {
                if (cmd[j]==' ') break;
                if (is_quate(cmd[j])) continue;
            }
        }
    };
    // remove "'"
    auto substr = [&](size_t i, size_t j) -> string {
        string s;
        for (size_t k=i; k<j; k++)
            if (!is_quate(cmd[k])) s += cmd[k];
        return s;
    };
    size_t i=0;
    pass(i);
    while (i<cmd.size()) {
        if (cmd[i] == ' ') {
            i++;
            continue;
        }
        if (cmd[i] == '-') {
            #ifdef _MSC_VER
            if (i+4<cmd.size() && cmd[i+1]=='F' && cmd[i+4]==' ') {
                // -Fo: -Fe:
                auto j=i+5;
                while (j<cmd.size() && cmd[j] == ' ') j++;
                CHECK(j<cmd.size());
                auto k=j;
                pass(k);
                CHECK(j<k && output_name.size()==0);
                // -Fo: xxx
                // i    j  k
                output_name = substr(j, k);
                i = k;
                continue;
            } else
            #endif
            if (i+2<cmd.size() && cmd[i+1]=='o' && cmd[i+2]==' ') {
                auto j=i+3;
                while (j<cmd.size() && cmd[j] == ' ') j++;
                CHECK(j<cmd.size());
                auto k=j;
                pass(k);
                CHECK(j<k && output_name.size()==0);
                // -o xxx
                // i  j  k
                output_name = substr(j, k);
                i = k;
                continue;
            } else if (i+2<cmd.size() && cmd[i+1]=='I') {
                // -Ixxx -I'xxx' -I xxx
                size_t j=i+2;
                while (j<cmd.size() && cmd[j]==' ') j++;
                size_t k=j;
                pass(k);
                CHECK(j<k);
                auto inc = substr(j, k);
                // find include
                i = k;
                extra["I"].push_back(inc);
                continue;
            } else if (i+2<cmd.size() && cmd[i+1]=='x') {
                // option with space and arg
                size_t j=i+2;
                while (j<cmd.size() && cmd[j]==' ') j++;
                size_t k=j;
                pass(k);
                i = k;
                continue;
            } else {
                pass(i);
                continue;
            }
        }
        auto j=i;
        pass(j);
        input_names.push_back(substr(i, j));
        i = j;
    }
    CHECK(output_name.size() && input_names.size())
        << "output_name: " << output_name
        << " input_names: " << input_names << "\n" << cmd;
}

static vector<string> parse_make_dependencies(const string& source) {
    size_t colon = string::npos;
    bool escaped = false;
    for (size_t i=0; i<source.size(); i++) {
        if (!escaped && source[i] == ':') {
            colon = i;
            break;
        }
        if (!escaped && source[i] == '\\') escaped = true;
        else escaped = false;
    }
    if (colon == string::npos) return {};

    vector<string> dependencies;
    string token;
    for (size_t i=colon+1; i<source.size(); i++) {
        char c = source[i];
        if (c == '\\' && i+1 < source.size()) {
            if (source[i+1] == '\n') {
                i++;
                continue;
            }
            if (source[i+1] == '\r' && i+2 < source.size()
                    && source[i+2] == '\n') {
                i += 2;
                continue;
            }
            token += source[++i];
            continue;
        }
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!token.empty()) {
                dependencies.push_back(token);
                token.clear();
            }
            continue;
        }
        token += c;
    }
    if (!token.empty()) dependencies.push_back(token);
    return dependencies;
}

static vector<string> parse_show_includes(const string& source) {
    vector<string> dependencies;
    size_t line_start = 0;
    while (line_start < source.size()) {
        size_t line_end = source.find('\n', line_start);
        if (line_end == string::npos) line_end = source.size();
        string line = source.substr(line_start, line_end-line_start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        size_t path_start = string::npos;
        for (size_t i=0; i+2<line.size(); i++) {
            bool drive = ((line[i]>='A' && line[i]<='Z') ||
                          (line[i]>='a' && line[i]<='z')) &&
                         line[i+1] == ':' &&
                         (line[i+2] == '\\' || line[i+2] == '/');
            if (drive) {
                path_start = i;
                break;
            }
        }
        if (path_start == string::npos) {
            auto unc = line.find("\\\\");
            if (unc != string::npos) path_start = unc;
        }
        if (path_start != string::npos)
            dependencies.push_back(strip(line.substr(path_start)));
        line_start = line_end + 1;
    }
    return dependencies;
}

static inline void check_win_file(const string& name) {
#ifdef _WIN32
    // win32 not allowed so file change when load
    // but we can rename it
    if (!file_exist(name)) return;
    if (!(endswith(name, ".pyd") || endswith(name, ".dll")))
        return;
    string new_name = name+".bk";
    LOGv << "move file" << name << "-> " << new_name;
    if (file_exist(new_name))
        std::filesystem::remove(new_name);
    std::filesystem::rename(name, new_name);
#endif
}

static string temporary_name(const string& name) {
#ifdef _WIN32
    auto pid = _getpid();
#else
    auto pid = getpid();
#endif
    return name + ".tmp." + std::to_string(pid);
}

static string add_dependency_flags(const string& cmd, const string& output_name,
                                   const string& dependency_name, bool msvc) {
    if (msvc) {
        // /showIncludes is stdout, not a file option. Keep compiler failures
        // visible to system_with_check while retaining successful output for
        // the dependency parser. VSLANG makes the diagnostic text stable, but
        // parsing below keys on the absolute path rather than the prefix.
        return "cmd.exe /D /S /C \"set VSLANG=1033&& " + cmd +
            " /showIncludes > \"" +
            dependency_name + "\" || (type \"" + dependency_name +
            "\" & exit /b 1)\"";
    }
    auto output_pos = cmd.rfind(output_name);
    CHECK(output_pos != string::npos) << "Output path not found in command:"
        << output_name << cmd;
    auto flag_pos = cmd.rfind(" -o ", output_pos);
    CHECK(flag_pos != string::npos) << "Output flag not found in command:" << cmd;
    // Insert before -o. run_and_install deliberately replaces the final
    // occurrence of output_name; a depfile derived from that name must not
    // become the occurrence it redirects.
    return cmd.substr(0, flag_pos) + " -MD -MF \"" + dependency_name +
        "\"" + cmd.substr(flag_pos);
}

static vector<string> read_dependencies(const string& dependency_name) {
    auto source = read_all(dependency_name);
    if (source.empty()) return {};
#ifdef _MSC_VER
    return parse_show_includes(source);
#else
    auto dependencies = parse_make_dependencies(source);
    CHECK(dependencies.size()) << "Could not parse dependency file:"
        << dependency_name;
    return dependencies;
#endif
}

static string build_cache_key(const string& cmd, vector<string> input_names,
                              const string& dependency_name) {
    auto dependencies = read_dependencies(dependency_name);
    input_names.insert(input_names.end(), dependencies.begin(), dependencies.end());
    unordered_set<string> processed;
    string cache_key = cmd + "\n";
    for (const auto& name : input_names) {
        if (name.empty() || name == "dynamic_lookup" || processed.count(name))
            continue;
        processed.insert(name);
        // Preserve the historical exclusion for import libraries. Object
        // files and compiler/wrapper executables remain command inputs and are
        // hashed even when the compiler does not put them in a depfile.
        if (name.back() == 'b') continue;
        cache_key += "# " + name + ": ";
        if (!file_exist(name)) {
            cache_key += "missing\n";
            continue;
        }
        cache_key += content_hash(read_all(name)) + "\n";
    }
    return cache_key;
}

static bool expects_dependency_file(const vector<string>& input_names) {
    for (const auto& name : input_names) {
        if (endswith(name, ".c") || endswith(name, ".cc") ||
            endswith(name, ".cpp") || endswith(name, ".cxx") ||
            endswith(name, ".cu"))
            return true;
    }
    return false;
}

static void install_file(const string& temporary, const string& destination) {
#ifdef _WIN32
    if (!MoveFileExA(temporary.c_str(), destination.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        auto reason = GetLastError();
        remove(temporary.c_str());
        LOGf << "could not install" << temporary << "as" << destination
             << ": win32 error" << reason;
    }
#else
    if (rename(temporary.c_str(), destination.c_str()) != 0) {
        string reason = strerror(errno);
        remove(temporary.c_str());
        LOGf << "could not install" << temporary << "as" << destination
             << ":" << reason;
    }
#endif
}

static void write_atomically(const string& name, const string& content) {
#ifdef TEST
    write(name, content);
#else
    string temporary = temporary_name(name);
    try {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        CHECK(output) << "Could not open temporary file:" << temporary;
        output.write(content.data(), content.size());
        output.close();
        CHECK(output) << "Could not write temporary file:" << temporary;
        install_file(temporary, name);
    } catch (...) {
        remove(temporary.c_str());
        throw;
    }
#endif
}

// Build the product under a private name and rename it into place, rather
// than letting the compiler write the final path directly.
//
// A linker writes its output by truncating the existing file. Rebuilding a
// shared library in place therefore corrupts the copy another process already
// has mapped -- which is exactly what happens when sources change while a test
// run is in flight -- and a reader that arrives mid-write sees a half-written
// file rather than either version. rename() within a directory is atomic, so
// neither can happen: the old inode stays alive for whoever already opened it,
// and the path flips from one complete product to the next.
static void run_and_install(const string& cmd, const string& output_name,
                            const string& tmp_dir,
                            const string& dependency_temporary,
                            const string& dependency_name,
                            bool dependency_required) {
#ifdef _WIN32
    check_win_file(output_name);
    try {
        system_with_check(cmd.c_str(), tmp_dir.c_str());
    } catch (...) {
        remove(dependency_temporary.c_str());
        throw;
    }
#else
    auto pos = cmd.rfind(output_name);
    if (pos == string::npos) {
        system_with_check(cmd.c_str(), tmp_dir.c_str());
        return;
    }
    string tmp_name = temporary_name(output_name);
    string tmp_cmd = cmd.substr(0, pos) + tmp_name
        + cmd.substr(pos + output_name.size());
    try {
        system_with_check(tmp_cmd.c_str(), tmp_dir.c_str());
    } catch (...) {
        remove(tmp_name.c_str());
        remove(dependency_temporary.c_str());
        throw;
    }
    if (!file_exist(tmp_name)) {
        // The command succeeded but wrote nothing where we redirected it: a
        // wrapper that ignores its -o argument, or the TEST build, where
        // system_with_check is a stub and cache_compile writes the output
        // itself. Leave the path exactly as the command left it.
        LOGvv << "no product at" << tmp_name >> ", installed nothing";
    } else {
        install_file(tmp_name, output_name);
    }
#endif
    if (file_exist(dependency_temporary)) {
        install_file(dependency_temporary, dependency_name);
#ifndef TEST
    } else {
        CHECK(!dependency_required) << "Compiler produced no dependency file:"
            << dependency_temporary << "\nCommand:" << cmd;
        // A pure link has no preprocessor input, so GCC legitimately emits no
        // depfile. Do not retain an older source-compile dependency list.
        remove(dependency_name.c_str());
#endif
    }
}

bool cache_compile(string cmd, const string& cache_path_, const string& jittor_path_) {
    #ifdef _WIN32
    cmd = _to_winstr(cmd);
    string cache_path = _to_winstr(cache_path_);
    string jittor_path = _to_winstr(jittor_path_);
    #else
    const string& cache_path = cache_path_;
    const string& jittor_path = jittor_path_;
    #endif
    vector<string> input_names;
    map<string,vector<string>> extra;
    string output_name;
    find_names(cmd, input_names, output_name, extra);
    string output_cache_key;
    bool ran = false;
    if (file_exist(output_name))
        output_cache_key = read_all(output_name+".key");
    string dependency_name = output_name + ".d";
    string dependency_temporary = temporary_name(dependency_name);
    string cache_key = build_cache_key(cmd, input_names, dependency_name);
    string tmp_dir = join(cache_path, "obj_files");
    if (output_cache_key != cache_key) {
        if (output_cache_key.empty())
            LOGvv << "Cache key of" << output_name << "not found.";
        else
            LOGvv << "Cache key of" << output_name << "changed.";
        remove(dependency_temporary.c_str());
#ifdef _MSC_VER
        const bool msvc = true;
#else
        const bool msvc = false;
#endif
        auto compile_cmd = add_dependency_flags(
            cmd, output_name, dependency_temporary, msvc);
        LOGvvv << "Run cmd:" << compile_cmd;
        run_and_install(compile_cmd, output_name, tmp_dir,
                        dependency_temporary, dependency_name,
                        expects_dependency_file(input_names));
        ran = true;
        cache_key = build_cache_key(cmd, input_names, dependency_name);
    }
    if (output_cache_key != cache_key) {
        LOGvvvv << "Prev cache key" << output_cache_key;
        LOGvvvv << "Write cache key" << output_name+".key:\n" >> cache_key;
        write_atomically(output_name+".key", cache_key);
    }
    if (!ran)
        LOGvvvv << "Command cached:" << cmd;
    #ifdef TEST
    if (ran)
        write(output_name, "...");
    #endif
    return ran;
}

} // jit_compiler
} // jittor

#ifdef TEST

#include "test.h"

static unordered_map<string,string> files;

namespace jittor {
namespace jit_compiler {

string read_all(const string& fname) {
    if (files.count(fname)) return files[fname];
    return "";
}

void write(const string& fname, const string& src) {
    files[fname] = src;
}

bool file_exist(const string& fname) {
    return files.count(fname);
}

}
}

void test_find_names(string cmd, vector<string> input_names, string output_name, map<string,vector<string>> extra={}) {
    LOGvv << cmd;
    vector<string> inames;
    string oname;
    map<string,vector<string>> ename;
    jittor::jit_compiler::find_names(cmd, inames, oname, ename);
    CHECKop(oname,==,output_name);
    CHECKop(inames.size(),==,input_names.size());
    for (size_t i=0; i<inames.size(); i++)
        CHECKop(inames[i],==,input_names[i]);
    CHECKop(ename.size(),==,extra.size());
    for (auto& kv : extra) {
        auto& x = ename[kv.first];
        CHECKop(x.size(),==,kv.second.size());
        for (size_t i=0; i<x.size(); i++)
            CHECKop(x[i],==,kv.second[i]);
    }
}

void test_find_nams_error(string cmd) {
    expect_error([&]() {
        vector<string> inames;
        string oname;
        map<string, vector<string>> ename;
        jittor::jit_compiler::find_names(cmd, inames, oname, ename);
    });
}

void test_main() {
    using jittor::jit_compiler::cache_compile;
    test_find_names("g++ a.cc b.cc -afdsf -xvs c.o -o asd",
        {"a.cc", "b.cc", "c.o"}, "asd");
    test_find_names("g++ -o asd a.cc b.cc -afdsf -xvs c.o",
        {"a.cc", "b.cc", "c.o"}, "asd");
    test_find_names("g++ -o asd 'a  ().cc' b.cc -afdsf -xvs c.o",
        {"a  ().cc", "b.cc", "c.o"}, "asd");
    test_find_nams_error("g++ -o");
    test_find_nams_error("g++ -o ");
    test_find_nams_error("g++ -o asd");
    
    // test include -I option
    test_find_names("g++ a.cc b.cc -I/a/b -I'/a a/b' -I  'a/ a/' -afdsf -xvs c.o -o asd",
        {"a.cc", "b.cc", "c.o"}, "asd", {{"I",{"/a/b","/a a/b","a/ a/"}}});
    
    auto deps = jittor::jit_compiler::parse_make_dependencies(
        "a.o: src/a.cc ex/a\\ b.h \\\n ex/c.h\n");
    CHECKop(deps.size(),==,3);
    CHECKop(deps[0],==,"src/a.cc");
    CHECKop(deps[1],==,"ex/a b.h");
    CHECKop(deps[2],==,"ex/c.h");

    auto shown = jittor::jit_compiler::parse_show_includes(
        "Note: including file: C:\\sdk\\a.h\r\n"
        "prefix \\\\server\\include\\b.h\r\n");
    CHECKop(shown.size(),==,2);
    CHECKop(shown[0],==,"C:\\sdk\\a.h");
    CHECKop(shown[1],==,"\\\\server\\include\\b.h");

    auto dep_cmd = jittor::jit_compiler::add_dependency_flags(
        "g++ a.cc -o a.o", "a.o", "a.o.d.tmp.1", false);
    CHECKop(dep_cmd,==,
        "g++ a.cc -MD -MF \"a.o.d.tmp.1\" -o a.o");
    auto msvc_cmd = jittor::jit_compiler::add_dependency_flags(
        "cl a.cc -Fo: a.obj", "a.obj", "a.obj.d.tmp.1", true);
    CHECK(msvc_cmd.find("cmd.exe /D /S /C \"set VSLANG=1033&& ") == 0);
    CHECK(msvc_cmd.find(" /showIncludes > \"a.obj.d.tmp.1\"") != string::npos);
    
    files = {{"src/a.h", "xxx"},
             {"src/a.cc", "#include \"a.h\"\nxxx"},
             {"a.o.d", "a.o: src/a.cc src/a.h\n"}};
    CHECK(cache_compile("echo src/a.cc -o a.o"));
    CHECK(files.count("a.o.key"));
    CHECK(!cache_compile("echo src/a.cc -o a.o"));
    files["src/a.h"] ="xxxx";
    CHECK(cache_compile("echo src/a.cc -o a.o"));
    files["src/a.cc"] ="xxxx";
    CHECK(cache_compile("echo src/a.cc -o a.o"));
    CHECK(cache_compile("echo src/a.cc -ff -o a.o"));

    // test include
    files = {{"ex/a.h", "xxx"},
             {"src/a.cc", "#include \"a.h\"\nxxx"},
             {"a.o.d", "a.o: src/a.cc ex/a.h\n"}};
    CHECK(cache_compile("echo src/a.cc -Iex -o a.o"));
    CHECK(files.count("a.o.key"));
    CHECK(files["a.o.key"].find("ex/a.h") >= 0);
}

#endif
