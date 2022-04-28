// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <string.h>
#include <signal.h>
#include <iomanip>
#include <thread>
#include <unordered_map>
#include <fstream>
#include "utils/cross_platform.h"
#include "utils/log.h"
#include "utils/mwsr_list.h"
#include "utils/str_utils.h"

namespace jittor {

bool peek_logged = 0;
typedef uint32_t uint;
using string = std::string;
using stringstream = std::stringstream;
using std::move;
template <class Ta, class Tb> using unordered_map = std::unordered_map<Ta,Tb>;

template<> string get_from_env(const char* name, const string& _default) {
    auto s = getenv(name);
    if (s == NULL) return _default;
    return string(s);
}

uint32_t get_tid() {
    stringstream ss;
    ss << std::this_thread::get_id();
    uint32_t id = static_cast<uint32_t>(std::stoull(ss.str()));
    return id;
}

static bool supports_color() {
    #ifdef _WIN32
    HANDLE hOut = GetStdHandle(STD_OUTPUT_HANDLE);
    if (hOut == INVALID_HANDLE_VALUE) return 0;

    DWORD dwMode = 0;
    if (!GetConsoleMode(hOut, &dwMode)) return 0;

    dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
    if (!SetConsoleMode(hOut, dwMode)) return 0;
    return 1;

    #endif
    bool term_supports_color = false;
    const char* const term = getenv("TERM");
    if (term != NULL && term[0] != '\0') {
    term_supports_color =
        !strcmp(term, "xterm") ||
        !strcmp(term, "xterm-color") ||
        !strcmp(term, "xterm-256color") ||
        !strcmp(term, "screen-256color") ||
        !strcmp(term, "konsole") ||
        !strcmp(term, "konsole-16color") ||
        !strcmp(term, "konsole-256color") ||
        !strcmp(term, "screen") ||
        !strcmp(term, "linux") ||
        !strcmp(term, "cygwin");
    }
    return term_supports_color;
}
bool g_supports_color = supports_color();
string thread_local thread_name;

struct timeval start_tv;

struct tm get_start_tm() {
    gettimeofday (&start_tv, NULL);
    time_t t = start_tv.tv_sec;
    return *localtime(&t);
}

struct tm start_tm = get_start_tm();

void print_prefix(std::ostream* out) {
    struct timeval tv;
    gettimeofday (&tv, NULL);
    struct tm lt = start_tm;
    auto dt = tv.tv_sec - start_tv.tv_sec;
    lt.tm_sec += dt;
    lt.tm_min += lt.tm_sec / 60; lt.tm_sec %= 60;
    lt.tm_hour += lt.tm_min / 60; lt.tm_min %= 60;
    // localtime is slow, cache time call
    if (lt.tm_hour >= 24) {
        start_tm = get_start_tm();
        tv = start_tv;
        lt = start_tm;
    }
    
    auto usecs = tv.tv_usec;
    
    thread_local uint32_t tid = get_tid()%100;
    
    #define PRINT_W2(x) \
        char('0'+(x)/10%10) << char('0'+(x)%10)
    #define PRINT_W6(x) \
        PRINT_W2((x)/10000) << PRINT_W2((x)/100) << PRINT_W2(x)
    
    *out << PRINT_W2(1+lt.tm_mon)
        << PRINT_W2(lt.tm_mday) << ' '
        << PRINT_W2(lt.tm_hour) << ':'
        << PRINT_W2(lt.tm_min) << ':'
        << PRINT_W2(lt.tm_sec) << "."
        << PRINT_W6(usecs) << ' '
        << PRINT_W2(tid);
    if (thread_name.size())
        *out << ":" << thread_name;
    *out << ' ';
}

#ifdef LOG_ASYNC
MWSR_LIST(log, std::ostringstream);
#endif
DECLARE_FLAG(int, log_sync);

std::mutex sync_log_m;
std::mutex sync_log_capture;
std::vector<std::map<string,string>> logs;
int log_capture_enabled = 0;

void log_capture(const string& s) {
    // find [ and ]
    uint i=0;
    while (i+2<s.size() && !(s[i]=='[' && s[i+2]==' ')) i++;
    ASSERT(i+2<s.size());
    uint j=i;
    while (j<s.size() && s[j]!=']') j++;
    ASSERT(j<s.size());
    // find all spaces
    std::vector<uint> spaces;
    spaces.reserve(5);
    for (uint k=i; k<j; k++)
        if (s[k]==' ') spaces.push_back(k);
    ASSERT(spaces.size()==4 || spaces.size()==5);
    std::map<string, string> log;
    log["level"] = s.substr(i+1, spaces[0]-i-1);
    log["verbose"] = spaces.size()==4 ? "0" : s.substr(spaces[3]+2, spaces[4]-spaces[3]-2);
    // split asdad.cc:asd
    uint l = spaces.back();
    while (l<j && s[l]!=':') l++;
    ASSERT(l<j);
    log["name"] = s.substr(spaces.back()+1, l-spaces.back()-1);
    log["lineno"] = s.substr(l+1, j-l-1);
    j++;
    if (j<s.size() && s[j]==' ') j++;
    uint end = s.size()-1;
    if (s[end]=='\n') end--;
    if (s[end-2]=='\033') end-=3;
    log["msg"] = s.substr(j, end-j+1);
    {
        std::lock_guard<std::mutex> lg(sync_log_capture);
        logs.emplace_back(std::move(log));
    }
}

DECLARE_FLAG(int, log_silent);

void send_log(std::ostringstream&& out, char level, int verbose) {
    if (log_capture_enabled)
        log_capture(out.str());
    if ((level=='i' || level=='w') && log_silent) return;
    if (!log_sync) {
        #if LOG_ASYNC
        mwsr_list_log::push(move(out));
        #endif
    } else {
        std::lock_guard<std::mutex> lk(sync_log_m);
        // std::cerr << "[SYNC]";
        std::cerr << _to_winstr(out.str());
        std::cerr.flush();
    }
}

void flush_log() {
    if (!log_sync) {
        #if LOG_ASYNC
        mwsr_list_log::flush();
        #endif
    } else {
        std::cerr.flush();
    }
}

void log_capture_start() { log_capture_enabled=1; }
void log_capture_stop() { log_capture_enabled=0; }
std::vector<std::map<string,string>> log_capture_read() {
    return move(logs);
}

void log_exiting();

bool exited = false;
size_t thread_local protected_page = 0;
int segfault_happen = 0;
static int _pid = getpid();
vector<void(*)()> cleanup_callback;
vector<void(*)()> sigquit_callback;
int64 last_q_time;

string& get_thread_name() {
    return thread_name;
}

#ifdef _WIN32
void handle_signal(int signal) {
    std::cerr << "Caught SIGNAL " << signal << ", quick exit";
    std::cerr.flush();
    abort();
}
#else
static inline void do_exit() {
    #ifdef __APPLE__
    _Exit(1);
    #else
    std::quick_exit(1);
    #endif
}

void segfault_sigaction(int signal, siginfo_t *si, void *arg) {
    if (signal == SIGQUIT) {
        if (_pid == getpid()) {
            std::cerr << "Caught SIGQUIT" << std::endl;
            int64 now = clock();
            if (now > last_q_time && last_q_time+CLOCKS_PER_SEC/10 > now) {
                last_q_time = now;
                std::cerr << "GDB attach..." << std::endl;
                breakpoint();
            } else {
                last_q_time = now;
                for (auto f : sigquit_callback)
                    f();
            }
        }
        return;
    }
    if (signal == SIGCHLD) {
        if (si->si_code != CLD_EXITED && si->si_status != SIGTERM && _pid == getpid()) {
            LOGe << "Caught SIGCHLD. Maybe out of memory, please reduce your worker size." 
                << "si_errno:" << si->si_errno 
                << "si_code:" << si->si_code 
                << "si_status:" << si->si_status
                << ", quick exit";
            exited = true;
            do_exit();
        }
        return;
    }
    if (signal == SIGINT) {
        if (_pid == getpid()) {
            LOGe << "Caught SIGINT, quick exit";
        }
        exited = true;
        do_exit();
    }
    if (exited) do_exit();
    std::cerr << "Caught segfault at address " << si->si_addr << ", "
        << "thread_name: '" << thread_name << "', flush log..." << std::endl;
    std::cerr.flush();
    if (protected_page && 
        si->si_addr>=(void*)protected_page && 
        si->si_addr<(void*)(protected_page+4*1024)) {
        LOGf << "Accessing protect pages, maybe jit_key too long";
    }
    if (!exited) {
        exited = true;
        if (signal == SIGSEGV) {
            // only print trace in main thread
            if (thread_name.size() == 0)
                print_trace();
            std::cerr << "Segfault, exit" << std::endl;
        } else {
            std::cerr << "Get signal " << signal << ", exit" << std::endl;
        }
    }
    segfault_happen = 1;
    exit(1);
}
#endif

int register_sigaction() {
#ifdef _WIN32
    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);
    // signal(SIGABRT, handle_signal);
    signal(SIGSEGV, handle_signal);
    signal(SIGFPE, handle_signal);
#else
    struct sigaction sa;

    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_sigaction;
    sa.sa_flags = SA_SIGINFO;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGKILL, &sa, NULL);
    sigaction(SIGSTOP, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    // jupyter use sigint to interp
    if (getenv("JPY_PARENT_PID") == nullptr)
        sigaction(SIGINT, &sa, NULL);
    sigaction(SIGCHLD, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
    // sigaction(SIGABRT, &sa, NULL);
#endif
    return 0;
}

static int log_init() {
    #ifdef _WIN32
    // SetConsoleCP(CP_UTF8);
    // SetConsoleOutputCP(CP_UTF8);
    #endif
    register_sigaction();
    std::atexit(log_exiting);
    return 1;
}

int _log_init = log_init();

void log_main() {
    #ifdef LOG_ASYNC
    mwsr_list_log::reduce([&](const std::ostringstream& out) {
        #ifdef TEST_LOG
        string s = out.str();
        if (s[8] == 'm') std::cerr << s;
        #else
        std::cerr << out.str();
        #endif
    }, [&]() {
        std::cerr.flush();
    });
    #endif
}

unordered_map<uint64_t, int> vprefix_map;
void stream_hash(uint64_t& hash, char c) {
    hash = hash * 257 + (uint8_t)c;
}

DEFINE_FLAG(int, log_sync, 1, "Set log printed synchronously.");
DEFINE_FLAG(int, log_silent, 0, "The log will be completely silent.");
DEFINE_FLAG(int, log_v, 0, "Verbose level of logging");
DEFINE_FLAG_WITH_SETTER(string, log_vprefix, "",
    "Verbose level of logging prefix\n"
    "example: log_vprefix='op=1,node=2,executor.cc:38$=1000'");
void setter_log_vprefix(string value) {
    unordered_map<uint64_t, int> new_map;
    auto& s = value;
    for (uint i=0; i<s.size(); i++) {
        uint j=i;
        while (j<s.size() && s[j]!='=') j++;
        uint k=j;
        while (k<s.size() && s[k]!=',') k++;
        // xxx=dd,...
        // i  j  k
        CHECK(i<j && j+1<k) << "load log_vprefix error:" << s;
        string prefix = s.substr(i,j-i);
        int vnum = std::stoi(s.substr(j+1, k-j-1));
        LOGvv << "Set prefix verbose:" << prefix << "->" << vnum;
        uint64_t phash=0;
        for (char c : prefix) stream_hash(phash, c);
        new_map[phash] = vnum;
        i = k;
    }
    vprefix_map = move(new_map);
}
DEFINE_FLAG_WITH_SETTER(string, log_file, "",
    "log to file, mpi env will add $OMPI_COMM_WORLD_RANK suffix\n");
void setter_log_file(string value) {
    if (value.size() == 0)
        return;
    auto c = getenv("OMPI_COMM_WORLD_RANK");
    if (c) value += string("_") + c;
    static std::ofstream out;
    out = std::ofstream(value);
    std::cerr.rdbuf(out.rdbuf());
}

bool check_vlog(const char* fileline, int verbose) {
    uint64_t phash=0;
    for (int i=0;; i++) {
        char c = fileline[i];
        if (!c) c = '$';
        stream_hash(phash, c);
        auto iter = vprefix_map.find(phash);
        if (iter != vprefix_map.end())
            return verbose <= iter->second;
        if (c=='$') break;
    }
    return verbose <= log_v;
}

static inline void check_cuda_unsupport_version(const string& output) {
    // check error like:
    // /usr/include/crt/host_config.h:121:2: error: #error -- unsupported GNU version! gcc versions later than 6 are not supported!
    // #error -- unsupported GNU version! gcc versions later than 6 are not supported!
    string pat = "crt/host_config.h";
    auto id = output.find(pat);
    if (id == string::npos) return;
    auto end = id + pat.size();
    while (id>=0 && !(output[id]==' ' || output[id]=='\t' || output[id]=='\n'))
        id--;
    id ++;
    auto fname = output.substr(id, end-id);
    LOGw << R"(
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Dear user, your nvcc and gcc version are not match, 
but you can hot fix it by this command:
>>> sudo python3 -c 's=open(")" >> fname >> R"(","r").read().replace("#error", "//#error");open(")" >> fname >> R"(","w").write(s)'
        )";
}

static inline void check_cuda_gcc_version(const string& output) {
    /*  if such error occur: 
    error: identifier "__is_assignable" is undefined
    this means your gcc version is not match with nvcc,
    for example, nvcc 10 support gcc<=7, nvcc 11 support gcc<=9,

    https://gist.github.com/ax3l/9489132
    */
    string pat = "__is_assignable";
    auto id = output.find(pat);
    if (id == string::npos) return;
    LOGf << output << R"(
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Dear user, your nvcc and gcc version are still not match
after dirty hack, your should install the correct version of g++
or nvcc, for example, nvcc 10 support g++<=7, nvcc 11 support g++<=9,
here is the NVCC Compatibility Matrix:
    https://gist.github.com/ax3l/9489132
Please install correct version of gcc, for example:
    >>> sudo apt install g++-7
After your g++ is installed, using enviroment variable `cc_path` to
tell jittor use the correct version of g++, for example:
    >>> cc_path='g++-7' python3.7 -m jittor.test.test_core
If you still have problems, please contact us:
    https://github.com/Jittor/jittor/issues
    )";
}

#ifdef _WIN32

string GbkToUtf8(const char *src_str)
{
	int len = MultiByteToWideChar(CP_ACP, 0, src_str, -1, NULL, 0);
	wchar_t* wstr = new wchar_t[len + 1];
	memset(wstr, 0, len + 1);
	MultiByteToWideChar(CP_ACP, 0, src_str, -1, wstr, len);
	len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, NULL, 0, NULL, NULL);
	char* str = new char[len + 1];
	memset(str, 0, len + 1);
	WideCharToMultiByte(CP_UTF8, 0, wstr, -1, str, len, NULL, NULL);
	string strTemp = str;
	if (wstr) delete[] wstr;
	if (str) delete[] str;
	return strTemp;
}

string Utf8ToGbk(const char *src_str)
{
	int len = MultiByteToWideChar(CP_UTF8, 0, src_str, -1, NULL, 0);
	wchar_t* wszGBK = new wchar_t[len + 1];
	memset(wszGBK, 0, len * 2 + 2);
	MultiByteToWideChar(CP_UTF8, 0, src_str, -1, wszGBK, len);
	len = WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, NULL, 0, NULL, NULL);
	char* szGBK = new char[len + 1];
	memset(szGBK, 0, len + 1);
	WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, szGBK, len, NULL, NULL);
	string strTemp(szGBK);
	if (wszGBK) delete[] wszGBK;
	if (szGBK) delete[] szGBK;
	return strTemp;
}

int system_popen(const char *cmd, const char* cwd) {
    HANDLE g_hChildStd_OUT_Rd = NULL;
    HANDLE g_hChildStd_OUT_Wr = NULL;
    SECURITY_ATTRIBUTES saAttr;
    // Set the bInheritHandle flag so pipe handles are inherited.

    saAttr.nLength = sizeof(SECURITY_ATTRIBUTES);
    saAttr.bInheritHandle = TRUE;
    saAttr.lpSecurityDescriptor = NULL;

    // Create a pipe for the child process's STDOUT.
    if (!CreatePipe(&g_hChildStd_OUT_Rd, &g_hChildStd_OUT_Wr, &saAttr, 0))
        LOGf << "StdoutRd CreatePipe error";
    // Ensure the read handle to the pipe for STDOUT is not inherited.
    if (!SetHandleInformation(g_hChildStd_OUT_Rd, HANDLE_FLAG_INHERIT, 0))
        LOGf << "Stdout SetHandleInformation error";

    // Create the child process.
    PROCESS_INFORMATION piProcInfo;
    STARTUPINFO siStartInfo;
    BOOL bSuccess = FALSE;
    // Set up members of the PROCESS_INFORMATION structure.
    ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

    // Set up members of the STARTUPINFO structure.
    // This structure specifies the STDIN and STDOUT handles for redirection.
    ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
    siStartInfo.cb = sizeof(STARTUPINFO);
    siStartInfo.hStdError = g_hChildStd_OUT_Wr;
    siStartInfo.hStdOutput = g_hChildStd_OUT_Wr;
    siStartInfo.dwFlags |= STARTF_USESTDHANDLES;

    // Create the child process.
    bSuccess = CreateProcess(NULL,
                             (char *)cmd,  // command line
                             NULL,         // process security attributes
                             NULL,         // primary thread security attributes
                             TRUE,         // handles are inherited
                             0,            // creation flags
                             NULL,         // use parent's environment
                             cwd,          // use cwd directory
                             &siStartInfo, // STARTUPINFO pointer
                             &piProcInfo); // receives PROCESS_INFORMATION

    // If an error occurs, exit the application.
    if (!bSuccess)
        LOGf << "CreateProcess error";
    // Close handles to the stdin and stdout pipes no longer needed by the child process.
    // If they are not explicitly closed, there is no way to recognize that the child process has ended.
    CloseHandle(g_hChildStd_OUT_Wr);

    DWORD dwRead, dwWritten;
    CHAR chBuf[BUFSIZ];
    HANDLE hParentStdOut = GetStdHandle(STD_OUTPUT_HANDLE);


    string output;
    for (;;)
    {
        bSuccess = ReadFile(g_hChildStd_OUT_Rd, chBuf, BUFSIZ, &dwRead, NULL);
        if (!bSuccess || dwRead == 0)
            break;
        output += string(chBuf, dwRead);

        if (log_v)
            bSuccess = WriteFile(hParentStdOut, chBuf,
                             dwRead, &dwWritten, NULL);
        if (!bSuccess)
            break;
    }
    WaitForSingleObject(piProcInfo.hProcess, INFINITE);
    DWORD ec;
    GetExitCodeProcess(piProcInfo.hProcess, &ec);
    // Close handles to the child process and its primary thread.
    // Some applications might keep these handles to monitor the status
    // of the child process, for example.
    CloseHandle(piProcInfo.hProcess);
    CloseHandle(piProcInfo.hThread);
    if (ec && !log_v)
        LOGe << output;

    if (ec) {
        check_cuda_unsupport_version(output);
        check_cuda_gcc_version(output);
    }
    return ec;
}
#else
int system_popen(const char* cmd, const char* cwd) {
    char buf[BUFSIZ];
    string cmd2;
    cmd2 = cmd;
    cmd2 += " 2>&1 ";
    FILE *ptr = popen(cmd2.c_str(), "r");
    if (!ptr) return -1;
    string output;
    while (fgets(buf, BUFSIZ, ptr) != NULL) {
        output += buf;
        if (log_v)
            std::cerr << buf;
    }
    if (output.size()) std::cerr.flush();
    auto ret = pclose(ptr);
    if (ret && !log_v)
        std::cerr << output;
    if (output.size()<10 && ret) {
        // maybe overcommit
        return -1;
    }
    if (ret) {
        check_cuda_unsupport_version(output);
        check_cuda_gcc_version(output);
    }
    return ret;
}
#endif

void system_with_check(const char* cmd, const char* cwd) {
    auto ret = system_popen(cmd, cwd);
    CHECK(ret>=0 && ret<=256) << "Run cmd failed:" << cmd <<
            "\nreturn ">> ret >> ". This might be an overcommit issue or out of memory."
            << "Try : sudo sysctl vm.overcommit_memory=1, or set enviroment variable `export DISABLE_MULTIPROCESSING=1`";
    CHECKop(ret,==,0) << "Run cmd failed:" << cmd;
}

#ifdef LOG_ASYNC
std::thread log_thread(log_main);
#endif

int log_exit = 0;

void log_exiting() {
    if (log_exit) return;
    log_exit = true;
    for (auto cb : cleanup_callback)
        cb();
    cleanup_callback.clear();
#ifdef LOG_ASYNC
    mwsr_list_log::stop();
    log_thread.join();
#endif
}

} // jittor


void expect_error(std::function<void()> func) {
    try {
        func();
    } catch (...) {
        return;
    }
    LOGf << "Missing error";
}

#ifdef TEST_LOG

#include <chrono>
#include <assert.h>
#include <streambuf>
#include "test.h"


DEFINE_FLAG (int, nthread, 4, "Number of thread");

void test_log_time(std::ostream* out) {
    int n = 100000;
    auto log_lot = [&]() {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i=0; i<n; i++) {
            LOGvvvv << "log time test" << i;
        }
        auto finish = std::chrono::high_resolution_clock::now();
        auto total_ns =  std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
        LOGi << "total_ns" << total_ns << "each_ns" << total_ns/n;
        CHECKop(total_ns/n,<=,6500);
    };
    std::list<std::thread> ts;
    for (int i=0; i<nthread; i++) ts.emplace_back(log_lot);
    for (auto& t : ts) t.join();
    CHECKop(jittor::mwsr_list_log::glist.size(),==,nthread+1);
}

void test_main() {
    LOGi << "test log info" << 1;
    LOGw << "test log warning";
    LOGe << "test log error";
    expect_error([&]() { LOGf << "test log fatal"; });
    LOGv << "test v1";
    LOGvv << "test v1";
    LOGvvv << "test v1";
    LOGvvvv << "test v1";
    expect_error([&]() { CHECKop(1,<,0) << "check error"; });
    test_log_time(&std::cerr);
}

#endif