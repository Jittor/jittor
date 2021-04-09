// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <string.h>
#include <signal.h>
#include <sys/time.h>
#include <iomanip>
#include <thread>
#include <unordered_map>
#include <unistd.h>
#include "utils/log.h"
#include "utils/mwsr_list.h"

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
    bool term_supports_color = false;
    #ifdef _WIN32
    // TODO: windows color not supported yet.
    term_supports_color = false;
    #else
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
    #endif
    return term_supports_color;
}
bool g_supports_color = supports_color();

struct timeval start_tv;

struct tm get_start_tm() {
    gettimeofday (&start_tv, NULL);
    return *localtime(&start_tv.tv_sec);
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

MWSR_LIST(log, std::ostringstream);
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

void send_log(std::ostringstream&& out) {
    if (log_capture_enabled)
        log_capture(out.str());
    if (log_silent) return;
    if (!log_sync) {
        mwsr_list_log::push(move(out));
    } else {
        std::lock_guard<std::mutex> lk(sync_log_m);
        // std::cerr << "[SYNC]";
        std::cerr << out.str();
        std::cerr.flush();
    }
}

void flush_log() {
    if (!log_sync) {
        mwsr_list_log::flush();
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
string thread_local thread_name;
static int _pid = getpid();

void segfault_sigaction(int signal, siginfo_t *si, void *arg) {
    if (signal == SIGINT) {
        if (_pid == getpid()) {
            LOGe << "Caught SIGINT, quick exit";
        }
        exited = true;
        std::quick_exit(1);
    }
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


int register_sigaction() {
    struct sigaction sa;

    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_sigaction;
    sa.sa_flags = SA_SIGINFO;

    sigaction(SIGSEGV, &sa, NULL);
    sigaction(SIGKILL, &sa, NULL);
    sigaction(SIGSTOP, &sa, NULL);
    sigaction(SIGFPE, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGILL, &sa, NULL);
    sigaction(SIGBUS, &sa, NULL);
    sigaction(SIGQUIT, &sa, NULL);
    // sigaction(SIGABRT, &sa, NULL);
    return 0;
}

void log_main() {
    register_sigaction();
    std::atexit(log_exiting);
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
}

unordered_map<uint64_t, int> vprefix_map;
void stream_hash(uint64_t& hash, char c) {
    hash = hash * 257 + (uint8_t)c;
}

DEFINE_FLAG(int, log_silent, 0, "The log will be completely silent.");
DEFINE_FLAG(int, log_v, 0, "Verbose level of logging");
DEFINE_FLAG(int, log_sync, 1, "Set log printed synchronously.");
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

int system_popen(const char* cmd) {
    char buf[BUFSIZ];
    string cmd2;
    cmd2 = cmd;
    cmd2 += " 2>&1 ";
    FILE *ptr = popen(cmd2.c_str(), "r");
    if (!ptr) return -1;
    int64 len=0;
    while (fgets(buf, BUFSIZ, ptr) != NULL) {
        len += strlen(buf);
        puts(buf);
    }
    auto ret = pclose(ptr);
    if (len<10 && ret) {
        // maybe overcommit
        return -1;
    }
    return ret;
}

void system_with_check(const char* cmd) {
    auto ret = system_popen(cmd);
    CHECK(ret>=0 && ret<=256) << "Run cmd failed:" << cmd <<
            "\nreturn ">> ret >> ". This might be an overcommit issue or out of memory."
            << "Try : sudo sysctl vm.overcommit_memory=1";
    CHECKop(ret,==,0) << "Run cmd failed:" << cmd;
}

std::thread log_thread(log_main);

void log_exiting() {
    if (exited) return;
    exited = true;
    mwsr_list_log::stop();
    log_thread.join();
}

} // jittor


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