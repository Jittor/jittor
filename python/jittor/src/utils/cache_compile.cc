// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <fstream>
#include <streambuf>
#ifdef _WIN32
#include <filesystem>
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

size_t skip_comments(const string& src, size_t i) {
    if (src[i] == '/' && (i+1<src.size() && src[i+1] == '/')) {
        size_t j=i+1;
        while (j<src.size() && src[j] != '\n') j++;
        if (j<src.size()) j++;
        return j;
    } else
    if (src[i] == '/' && (i+1<src.size() && src[i+1] == '*')) {
        size_t j=i+1;
        while (j<src.size() && !(src[j] == '/' && src[j-1] == '*')) j++;
        if (j<src.size()) j++;
        return j;
    }
    return i;
}

map<string,string> jt_env;

void process(string src, vector<string>& input_names, string& cmd) {
    for (size_t i=0; i<src.size(); i++) {
        i = skip_comments(src, i);
        if (i>=src.size()) break;
        if (src[i] == '#') {
            // #include "a.h"
            // i       jk    l
            auto j=i+1;
            while (j<src.size() && (src[j] != ' ' && src[j] != '\"' && src[j] != '\n' && src[j] != '\r')) j++;
            if (j>=src.size()) return;
            if (j-i != 8 && j-i != 6) continue;
            auto k=src[j] == '\"' ? j : j+1;
            while (k<src.size() && src[k] == ' ') k++;
            if (k>=src.size()) return;
            auto l=k+1;
            while (l<src.size() && (src[l] != ' ' && src[l] != '\n' && src[l] != '\r')) l++;
            if (src[k] == '"' && src[l-1] == '"' && j-i==8 && src.substr(i,j-i) == "#include") {
                auto inc = src.substr(k+1, l-k-2);
                if (inc != "test.h" && inc != "helper_cuda.h") {
                    LOGvvvv << "Found include" << inc; 
                    input_names.push_back(inc);
                }
            }
            if (l-k>2 && src[k] == 'J' && src[k+1] == 'T' && j-i==6 && src.substr(i,j-i) == "#ifdef") {
                auto inc = strip(src.substr(k, l-k));
                auto env = getenv(inc.c_str());
                if (env && string(env)!="0") {
                    auto senv = string(env);
                    if (!jt_env.count(inc)) {
                        LOGe << "Load JT env ok:" << inc << senv;
                        jt_env[inc] = senv;
                    }
                    string dflag = " -D"+inc+"="+senv;
                    if (cmd.find(dflag) == string::npos) {
                        // -D flags should insert before -o flag
                        #ifdef _MSC_VER
                        string patt = " -Fo: ";
                        #else
                        string patt = " -o ";
                        #endif
                        auto cmds = split(cmd, patt, 2);
                        if (cmds.size() == 2) {
                            cmd = cmds[0] + dflag + patt + cmds[1];
                        }
                    }
                }
            }
            i=l;
        }
    }
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

static inline bool is_full_path(const string& name) {
#ifdef _WIN32
    return name.size()>=2 && (name[1]==':' || (name[0]=='\\' && name[1]=='\\'));
#else
    return name.size() && name[0]=='/';
#endif
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
    string cache_key;
    unordered_set<string> processed;
    auto src_path = join(jittor_path, "src");
    const auto& extra_include = extra["I"];
    string tmp_dir =join(cache_path, "obj_files");
    for (size_t i=0; i<input_names.size(); i++) {
        if (processed.count(input_names[i]) != 0)
            continue;
        if (input_names[i] == "dynamic_lookup")
            continue;
        processed.insert(input_names[i]);
        auto src = read_all(input_names[i]);
        #ifdef _WIN32
        src = _to_winstr(src);
        #endif
        auto back = input_names[i].back();
        // *.lib
        if (back == 'b') continue;
        ASSERT(src.size()) << "Source read failed:" << input_names[i] << "cmd:" << cmd;
        auto hash = S(hash64(src));
        vector<string> new_names;
        // *.obj, *.o, *.pyd
        if (back != 'j' && back != 'o' && back != 'd')
            process(src, new_names, cmd);
        for (auto& name : new_names) {
            string full_name;
            if (name.substr(0, 4) == "jit/" || name.substr(0, 4) == "gen/")
                full_name = join(cache_path, name);
            else if (is_full_path(name))
                full_name = name;
            else
                full_name = join(src_path, name);
            if (!file_exist(full_name)) {
                bool found = 0;
                for (const auto& inc : extra_include) {
                    full_name = join(inc, name);
                    if (file_exist(full_name)) {
                        found = 1;
                        break;
                    }
                }
                ASSERT(found) << "Include file" << name << "not found in" << extra_include
                    >> "\nCommands:" << cmd;
                LOGvvvv << "Include file found:" << full_name;
            }
            input_names.push_back(full_name);
        }
        cache_key += "# ";
        cache_key += input_names[i];
        cache_key += ": ";
        cache_key += hash;
        cache_key += "\n";
    }
    cache_key = cmd + "\n" + cache_key;
    if (output_cache_key.size() == 0) {
        LOGvv << "Cache key of" << output_name << "not found.";
        LOGvvv << "Run cmd:" << cmd;
        check_win_file(output_name);
        system_with_check(cmd.c_str(), tmp_dir.c_str());
        ran = true;
    }
    if (output_cache_key.size() != 0 && output_cache_key != cache_key) {
        LOGvv << "Cache key of" << output_name << "changed.";
        LOGvvv << "Run cmd:" << cmd;
        check_win_file(output_name);
        system_with_check(cmd.c_str(), tmp_dir.c_str());
        ran = true;
    }
    if (output_cache_key != cache_key) {
        LOGvvvv << "Prev cache key" << output_cache_key;
        LOGvvvv << "Write cache key" << output_name+".key:\n" >> cache_key;
        write(output_name+".key", cache_key);
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

void test_process(string src, vector<string> files) {
    vector<string> ifiles;
    string cmd;
    jittor::jit_compiler::process(src, ifiles, cmd);
    CHECK(files.size() == ifiles.size());
    for (size_t i=0; i<files.size(); i++)
        CHECKop(files[i],==,ifiles[i]);
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
    
    test_process("", {});
    test_process("#inc <asd>", {});
    test_process("#include <asd>", {});
    test_process("#include \"asd\"", {"asd"});
    test_process("//#include \"asd\"", {});
    test_process("/*#include \"asd\"*/", {});
    test_process("#include \"asd\"\n#include \"zxc\"", {"asd", "zxc"});
    
    files = {{"src/a.h", "xxx"}, {"src/a.cc", "#include \"a.h\"\nxxx"}};
    CHECK(cache_compile("echo src/a.cc -o a.o"));
    CHECK(files.count("a.o.key"));
    CHECK(!cache_compile("echo src/a.cc -o a.o"));
    files["src/a.h"] ="xxxx";
    CHECK(cache_compile("echo src/a.cc -o a.o"));
    files["src/a.cc"] ="xxxx";
    CHECK(cache_compile("echo src/a.cc -o a.o"));
    CHECK(cache_compile("echo src/a.cc -ff -o a.o"));

    // test include
    files = {{"ex/a.h", "xxx"}, {"src/a.cc", "#include \"a.h\"\nxxx"}};
    CHECK(cache_compile("echo src/a.cc -Iex -o a.o"));
    CHECK(files.count("a.o.key"));
    CHECK(files["a.o.key"].find("ex/a.h") >= 0);
    expect_error([&]() {
        cache_compile("echo src/a.cc -o a.o");
    });
}

#endif
