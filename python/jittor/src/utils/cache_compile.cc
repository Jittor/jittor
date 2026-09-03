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

// Is NAME defined by a -D on this command line?
//
// The scanner cannot evaluate the preprocessor, but it can read the one thing
// that decides most of the conditionals that matter here: what the build is
// configured with. -DHAS_CUDA is the whole difference between a source's
// `#include "helper_cuda.h"` being real and being dead text.
bool macro_defined_on_cmd(const string& cmd, const string& name) {
    string pattern = "-D" + name;
    size_t pos = 0;
    while ((pos = cmd.find(pattern, pos)) != string::npos) {
        size_t after = pos + pattern.size();
        bool left = pos == 0 || cmd[pos-1]==' ' || cmd[pos-1]=='"' || cmd[pos-1]=='\'';
        bool right = after >= cmd.size() || cmd[after]==' ' || cmd[after]=='='
                  || cmd[after]=='"' || cmd[after]=='\'';
        if (left && right) return true;
        pos = after;
    }
    return false;
}

// Scan a source for the headers it includes.
//
// It used to do a second, unrelated job as well: find `#ifdef JT_XXX`, look the
// name up in the environment, and rewrite the compiler command line in place.
// That coupled "decide the command line", which has to happen before a
// compile, to "collect dependencies", which the compiler can only report after
// one -- and while they shared a scanner the compiler's own `-MD -MF` could
// not be used, because the first cold compile would have gone out without its
// `-D`. Those flags are now decided in Python from a declared list
// (`compiler.JT_CONFIG_MACROS`), which is what unblocks replacing the rest of
// this function with a depfile.
//
// `strict` marks each name with whether failing to resolve it is an error.
// A name is strict only when every enclosing conditional is known to be true
// -- which in practice means "not inside any conditional" or "inside
// `#ifdef X` with -DX on the command line". Everything else is best effort:
// tracked if it can be found, ignored if it cannot.
//
// That rule is what replaced two hardcoded exceptions. `helper_cuda.h` and
// `test.h` were skipped by name because the scanner does not understand
// `#ifdef`: both are included under conditions that are false in an ordinary
// build (`#ifdef HAS_CUDA` in 47 files, `#ifdef TEST` here), the include path
// that would resolve them is only present when those conditions hold, and an
// unresolvable include was fatal. Skipping them by name also meant that
// editing `helper_cuda.h` -- included by 47 files -- rebuilt nothing at all.
//
// This is still a hand-written approximation of the preprocessor, and asking
// the compiler for its own dependency list (`-MD -MF`) remains the end state.
// The coupling that blocked it is gone; what is left is the mechanical work of
// reading depfiles, plus the wrappers that rewrite the command line by string
// matching (asm_tuner.py, dlink_compiler.py) and MSVC, which has no -MF.
void process(string src, vector<string>& input_names, const string& cmd,
             vector<char>* strict_out) {
    // 1 = this conditional is known to be true, 0 = we cannot say.
    vector<char> conds;
    auto all_known = [&]() {
        for (char c : conds) if (!c) return false;
        return true;
    };
    for (size_t i=0; i<src.size(); i++) {
        i = skip_comments(src, i);
        if (i>=src.size()) break;
        if (src[i] == '#') {
            // #include "a.h"
            // i       jk    l
            auto j=i+1;
            while (j<src.size() && (src[j] != ' ' && src[j] != '\"' && src[j] != '\n' && src[j] != '\r')) j++;
            if (j>=src.size()) return;
            auto directive = src.substr(i, j-i);
            // Everything below reads the *argument* of this directive, so it
            // must not run past the end of the line. `#else` and `#endif` have
            // no argument: without this bound the scan walked onto the next
            // line, and `i = l` at the bottom then skipped over whatever
            // directive was there. An `#endif` immediately followed by
            // `#ifdef HAS_CUDA` swallowed the `#ifdef`, so the conditional was
            // never opened and the include inside it looked unconditional.
            auto eol = j;
            while (eol < src.size() && src[eol] != '\n') eol++;
            auto k=src[j] == '\"' ? j : j+1;
            if (k > eol) k = eol;
            while (k<eol && src[k] == ' ') k++;
            auto l=k<eol ? k+1 : eol;
            while (l<eol && (src[l] != ' ' && src[l] != '\r')) l++;
            bool has_argument = l > k;
            if (directive == "#endif") {
                if (conds.size()) conds.pop_back();
                i = eol;
                continue;
            }
            if (directive == "#else" || directive == "#elif") {
                // The branch we are entering is one we did not evaluate.
                if (conds.size()) conds.back() = 0;
                i = eol;
                continue;
            }
            if (directive == "#ifdef" || directive == "#ifndef") {
                auto name = has_argument ? strip(src.substr(k, l-k)) : string();
                bool defined = macro_defined_on_cmd(cmd, name);
                // `#ifndef GUARD_H` around a whole header is the common case:
                // the guard is never on the command line, so the body is
                // known-live and keeps the error for a missing include.
                conds.push_back((directive == "#ifdef") == defined ? 1 : 0);
                i = eol;
                continue;
            }
            if (directive == "#if") {
                conds.push_back(0);
                i = eol;
                continue;
            }
            bool quoted = has_argument && src[k] == '"' && src[l-1] == '"';
            // Angle brackets were not tracked at all, so a project header
            // included as <...> could be edited without rebuilding anything.
            // They are resolved against the same search path and simply
            // ignored when they turn out to be system headers.
            bool angled = has_argument && src[k] == '<' && src[l-1] == '>';
            if ((quoted || angled) && directive == "#include") {
                auto inc = src.substr(k+1, l-k-2);
                LOGvvvv << "Found include" << inc;
                input_names.push_back(inc);
                if (strict_out)
                    strict_out->push_back(quoted && all_known() ? 1 : 0);
            }
            i=eol;
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

static string temporary_name(const string& name) {
#ifdef _WIN32
    auto pid = _getpid();
#else
    auto pid = getpid();
#endif
    return name + ".tmp." + std::to_string(pid);
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
                            const string& tmp_dir) {
#ifdef _WIN32
    check_win_file(output_name);
    system_with_check(cmd.c_str(), tmp_dir.c_str());
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
        throw;
    }
    if (!file_exist(tmp_name)) {
        // The command succeeded but wrote nothing where we redirected it: a
        // wrapper that ignores its -o argument, or the TEST build, where
        // system_with_check is a stub and cache_compile writes the output
        // itself. Leave the path exactly as the command left it.
        LOGvv << "no product at" << tmp_name >> ", installed nothing";
        return;
    }
    install_file(tmp_name, output_name);
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
        auto hash = content_hash(src);
        vector<string> new_names;
        vector<char> new_strict;
        // Scan only files that belong to this project.
        //
        // Now that `<...>` includes are followed, the search reaches system
        // headers -- `<cuda_fp16.h>` resolves through -I/usr/local/cuda/include.
        // Those headers include their own private files relative to their own
        // directory (`#include "detail/__target_macros"`), which this resolver
        // knows nothing about, so descending into them turned every CUDA
        // toolkit header into a fatal "include not found". A system header is
        // still hashed -- upgrading the toolkit should rebuild -- but its
        // insides are the toolkit's business.
        // jittor_path, not src_path: extern/ headers are ours too. With
        // neither root configured there is nothing to be outside of, so scan
        // everything -- that is the TEST harness and the default arguments.
        bool in_project = jittor_path.empty() && cache_path.empty();
        if (!in_project)
            in_project =
                (jittor_path.size() &&
                 input_names[i].compare(0, jittor_path.size(), jittor_path) == 0) ||
                (cache_path.size() &&
                 input_names[i].compare(0, cache_path.size(), cache_path) == 0);
        // *.obj, *.o, *.pyd
        if (in_project && back != 'j' && back != 'o' && back != 'd')
            process(src, new_names, cmd, &new_strict);
        for (size_t n=0; n<new_names.size(); n++) {
            const auto& name = new_names[n];
            bool strict = n < new_strict.size() ? new_strict[n] != 0 : true;
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
                if (!found) {
                    // Not an error unless every enclosing conditional is known
                    // to hold and the include was quoted. A `<...>` include is
                    // usually a system header, and a quoted one under an
                    // `#ifdef` we could not evaluate is text the compiler will
                    // never see -- `#include "helper_cuda.h"` under
                    // `#ifdef HAS_CUDA` in a CPU build is exactly that, and
                    // used to be excluded by name for this reason.
                    if (!strict) {
                        LOGvvvv << "Include file" << name
                            << "not resolved and not required here, skipping";
                        continue;
                    }
                    ASSERT(found) << "Include file" << name << "not found in" << extra_include
                        >> "\nCommands:" << cmd;
                }
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
        run_and_install(cmd, output_name, tmp_dir);
        ran = true;
    }
    if (output_cache_key.size() != 0 && output_cache_key != cache_key) {
        LOGvv << "Cache key of" << output_name << "changed.";
        LOGvvv << "Run cmd:" << cmd;
        run_and_install(cmd, output_name, tmp_dir);
        ran = true;
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

void test_process(string src, vector<string> files) {
    vector<string> ifiles;
    string cmd;
    jittor::jit_compiler::process(src, ifiles, cmd, nullptr);
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
    // Angle brackets are tracked now: a project header included as <...> used
    // to be invisible to the cache. It is resolved against the same search
    // path and dropped later if it turns out to be a system header.
    test_process("#include <asd>", {"asd"});
    test_process("#include \"asd\"", {"asd"});
    // A conditional whose macro is not on the command line makes what is
    // inside best-effort rather than fatal -- this is what replaced the
    // hardcoded "test.h"/"helper_cuda.h" exceptions.
    test_process("#ifdef HAS_CUDA\n#include \"helper_cuda.h\"\n#endif",
        {"helper_cuda.h"});
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
