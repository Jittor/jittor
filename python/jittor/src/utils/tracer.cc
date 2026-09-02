// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#ifndef _WIN32
#include <signal.h>
#endif
#include "utils/cross_platform.h"
#include <fcntl.h>
#include <poll.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string>
#include "utils/tracer.h"

namespace jittor {
    
DEFINE_FLAG_WITH_SETTER(string, gdb_path, "", "Path of GDB.");
DEFINE_FLAG(string, addr2line_path, "", "Path of addr2line.");
DEFINE_FLAG(string, extra_gdb_cmd, "", "Extra command pass to GDB, seperate by(;) .");
DEFINE_FLAG(int, has_pybt, 0, "GDB has pybt or not.");
DEFINE_FLAG(int, trace_depth, 10, "trace depth for GDB.");
DEFINE_FLAG(int, gdb_trace_timeout, 30,
    "Seconds to wait for the GDB backtrace child before giving up. "
    "Zero or a negative value waits forever.");
DEFINE_FLAG_WITH_SETTER(int, gdb_attach, 0, "gdb attach self process.");

string _extra_gdb_cmd;

int system_popen(const char* cmd, const char* cwd=nullptr);

#ifdef _WIN32
string get_cmds(const vector<const char*>& argv) {
    auto cmds = gdb_path;
    for (auto p : argv) {
        if (!p) continue;
        string cmd = p;
        cmds += " ";
        if (cmd.find(' ') != string::npos && cmd[0] != '"')
            cmds += '"' + cmd + '"';
        else
            cmds += cmd;
    }
    return cmds;
}
#endif

void setter_gdb_attach(int v) {
    if (v && gdb_path.size()) {
        static int gdb_attached = 0;
        if (gdb_attached) return;
        gdb_attached = 1;
        // using gdb to print the stack trace
        char pid_buf[30];
        sprintf(pid_buf, "%d", getpid());

        vector<const char*> argv{
            gdb_path.c_str(),
            "-ex", "catch throw"
        };
        if (auto n = extra_gdb_cmd.size()) {
            _extra_gdb_cmd = extra_gdb_cmd;
            _extra_gdb_cmd += '\0';
            argv.push_back("-ex");
            argv.push_back(&_extra_gdb_cmd[0]);
            for (uint i=0; i<n; i++) {
                if (_extra_gdb_cmd[i]==';') {
                    argv.push_back("-ex");
                    _extra_gdb_cmd[i] = '\0';
                    argv.push_back(&_extra_gdb_cmd[i+1]);
                }
            }
        }
        LOGi << "gdb attach for" << "pid=" >> pid_buf << argv;
        // argv.insert(argv.end(), {name_buf, pid_buf, NULL});
        argv.insert(argv.end(), {"-p", pid_buf, NULL});

        #ifdef _WIN32
        // _spawnvp(_P_OVERLAY, gdb_path.c_str(), (char* const*)&argv[0]);
        // system_popen((gdb_path+" -p "+pid_buf).c_str());
        auto cmds = get_cmds(argv);

        // Create the child process.
        PROCESS_INFORMATION piProcInfo;
        STARTUPINFO siStartInfo;
        BOOL bSuccess = false;
        // Set up members of the PROCESS_INFORMATION structure.
        ZeroMemory(&piProcInfo, sizeof(PROCESS_INFORMATION));

        // Set up members of the STARTUPINFO structure.
        // This structure specifies the STDIN and STDOUT handles for redirection.
        ZeroMemory(&siStartInfo, sizeof(STARTUPINFO));
        siStartInfo.cb = sizeof(STARTUPINFO);
        // siStartInfo.hStdError = g_hChildStd_OUT_Wr;
        // siStartInfo.hStdOutput = g_hChildStd_OUT_Wr;
        siStartInfo.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
        siStartInfo.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
        siStartInfo.hStdError = GetStdHandle(STD_ERROR_HANDLE);
        siStartInfo.dwFlags |= STARTF_USESTDHANDLES;
        // Create the child process.
        bSuccess = CreateProcess(
            NULL,
            (char *)&cmds[0],  // command line
            NULL,         // process security attributes
            NULL,         // primary thread security attributes
            true,         // handles are inherited
            0,            // creation flags
            NULL,         // use parent's environment
            NULL,         // use parent's current directory
            &siStartInfo, // STARTUPINFO pointer
            &piProcInfo); // receives PROCESS_INFORMATION

        // If an error occurs, exit the application.
        if (!bSuccess)
            LOGf << "CreateProcess error, command:" << cmds;
        // sleep 5s, wait gdb attach
        sleep(5);
        #else
        int child_pid = fork();
        if (!child_pid) {
            auto ret = execvp(gdb_path.c_str(), (char* const*)&argv[0]);
            LOGf << "execvp failed return" << ret << gdb_path << extra_gdb_cmd;
            exit(1);
        } else {
            // allow children ptrace parent
#if defined(__linux__) && defined(PR_SET_PTRACER)
    		prctl(PR_SET_PTRACER, child_pid, 0, 0, 0);
#endif
            // sleep 5s, wait gdb attach
            sleep(5);
        }
        #endif
    }
}

void setter_gdb_path(string v) {
    gdb_path = v;
    setter_gdb_attach(gdb_attach);
}

void breakpoint() {
    static bool is_attached = 0;
    if (is_attached) return;
    setter_gdb_attach(1);
}

// ---------------------------------------------------------------------------
// Pre-forked symbolizer
// ---------------------------------------------------------------------------
// print_trace() runs from inside the SIGSEGV handler, and almost everything it
// used to do there is forbidden: backtrace_symbols() allocates, system() forks
// a shell through the C library, and the gdb path builds std::vector and
// std::string and formats with sprintf and LOGi. If the fault happened inside
// the allocator -- and "crashed while allocating" is one of the ordinary ways
// to crash -- re-entering the allocator from the handler deadlocks. The process
// then hangs with no report at all, which is strictly worse than crashing:
// a hung process is indistinguishable from a slow one.
//
// The work therefore moves to a process forked BEFORE anything went wrong, so
// it owns a healthy heap. At crash time the handler does only what POSIX
// permits in a signal handler: backtrace() into a stack array (its lazy libgcc
// load is forced during startup, so it does not allocate here), one write(2) of
// a fixed-size POD, and a bounded wait for the reply.
//
// The helper resolves addresses through the *parent's* /proc/<pid>/maps rather
// than its own backtrace_symbols(). That is not a detail: the helper is forked
// at import time, and jittor dlopens JIT-compiled modules continuously
// afterwards, so the helper's own address space is missing exactly the modules
// a crash is most likely to be in. Reading the parent's maps resolves every
// frame to module+offset, which is also the form `addr2line -e <module> <off>`
// wants.

#define JT_TRACE_MAX 64
#define JT_TRACE_MAGIC 0x6a747472  // "jttr"

struct TraceRequest {
    int32_t magic;
    int32_t depth;
    int32_t pid;
    int32_t signal;
    void* fault_pc;
    void* caller_pc;
    void* frames[JT_TRACE_MAX];
};

static int trace_request_fd = -1;   // parent -> helper
static int trace_ack_fd = -1;       // helper -> parent
static pid_t trace_helper_pid = -1;

// write(2) only; used on both sides so the helper cannot deadlock either.
static void trace_write(int fd, const char* s) {
    if (!s) return;
    size_t n = 0;
    while (s[n]) n++;
    ssize_t written = write(fd, s, n);
    (void)written;
}

static void trace_write_hex(int fd, uint64_t v) {
    static const char digits[] = "0123456789abcdef";
    char buf[18];   // "0x" + 16 hex digits, no terminator: write() takes a length
    buf[0] = '0'; buf[1] = 'x';
    for (int i = 0; i < 16; i++)
        buf[2 + 15 - i] = digits[(v >> (4 * i)) & 0xf];
    ssize_t written = write(fd, buf, sizeof(buf));
    (void)written;
}

static void trace_write_dec(int fd, long long v) {
    char buf[24];
    int i = sizeof(buf);
    bool neg = v < 0;
    unsigned long long m = neg ? -(unsigned long long)v : (unsigned long long)v;
    if (!m) buf[--i] = '0';
    while (m) { buf[--i] = (char)('0' + m % 10); m /= 10; }
    if (neg) buf[--i] = '-';
    ssize_t written = write(fd, buf + i, sizeof(buf) - i);
    (void)written;
}

namespace {

// One line of /proc/<pid>/maps that we care about.
struct MapEntry {
    uint64_t lo, hi, offset;
    std::string path;
};

std::vector<MapEntry> read_maps(int pid) {
    std::vector<MapEntry> entries;
    char path[64];
    snprintf(path, sizeof(path), "/proc/%d/maps", pid);
    FILE* f = fopen(path, "r");
    if (!f) return entries;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        uint64_t lo = 0, hi = 0, off = 0;
        char perms[8] = {0};
        int name_pos = 0;
        if (sscanf(line, "%lx-%lx %7s %lx %*s %*s %n",
                   &lo, &hi, perms, &off, &name_pos) < 4)
            continue;
        if (perms[2] != 'x') continue;          // executable mappings only
        std::string name = name_pos > 0 ? line + name_pos : "";
        while (name.size() && (name.back() == '\n' || name.back() == ' '))
            name.pop_back();
        if (name.empty() || name[0] == '[') continue;
        entries.push_back(MapEntry{lo, hi, off, name});
    }
    fclose(f);
    return entries;
}

const MapEntry* find_map(const std::vector<MapEntry>& maps, uint64_t addr) {
    for (auto& m : maps)
        if (addr >= m.lo && addr < m.hi) return &m;
    return nullptr;
}

// Runs in the helper, which has a healthy heap and may allocate freely.
void symbolize(const TraceRequest& req) {
    auto maps = read_maps(req.pid);
    auto one = [&](const char* tag, void* pc) {
        if (!pc) return;
        uint64_t addr = (uint64_t)pc;
        trace_write(2, tag);
        trace_write(2, " ");
        trace_write_hex(2, addr);
        const MapEntry* m = find_map(maps, addr);
        if (m) {
            uint64_t file_off = addr - m->lo + m->offset;
            trace_write(2, " in ");
            trace_write(2, m->path.c_str());
            trace_write(2, "+");
            trace_write_hex(2, file_off);
            // getenv, not the addr2line_path flag: this process was forked
            // during static initialization, and whether tracer.cc's globals
            // were constructed before log.cc's initializer ran is unspecified
            // across translation units. The environment is set up by the C
            // runtime before any of that, and jittor's flags take their
            // defaults from identically-named variables anyway.
            const char* a2l = getenv("addr2line_path");
            if (a2l && *a2l) {
                trace_write(2, "\n    ");
                fflush(nullptr);
                std::string cmd = std::string(a2l) + " -f -p -i -e '" + m->path +
                    "' " + std::to_string(file_off);
                int ret = system(cmd.c_str());
                (void)ret;
                return;
            }
        } else {
            trace_write(2, " (no mapping)");
        }
        trace_write(2, "\n");
    };

    trace_write(2, "[bt] crash report for pid ");
    trace_write_dec(2, req.pid);
    trace_write(2, ", signal ");
    trace_write_dec(2, req.signal);
    trace_write(2, "\n");
    one("[bt] fault pc  ", req.fault_pc);
    one("[bt] caller pc ", req.caller_pc);
    for (int i = 0; i < req.depth && i < JT_TRACE_MAX; i++) {
        char tag[24];
        snprintf(tag, sizeof(tag), "[bt] #%-2d", i);
        one(tag, req.frames[i]);
    }
    const char* a2l_hint = getenv("addr2line_path");
    if (!a2l_hint || !*a2l_hint)
        trace_write(2, "[bt] set addr2line_path=$(which addr2line) to resolve"
                       " these to source lines\n");
    fflush(stderr);
}

void helper_main(int request_fd, int ack_fd) {
    TraceRequest req;
    while (true) {
        size_t got = 0;
        while (got < sizeof(req)) {
            ssize_t n = read(request_fd, (char*)&req + got, sizeof(req) - got);
            if (n <= 0) return;                 // parent gone: we are done
            got += (size_t)n;
        }
        if (req.magic != JT_TRACE_MAGIC) return;   // shutdown, or garbage
        // Deliberately no gdb here. Attaching a debugger to the crashed parent
        // needs ptrace and a bounded wait on a process that may itself wedge --
        // reintroducing, one level out, the hang this change exists to remove.
        // `jt.print_trace()` still runs the full gdb path when a human asks for
        // it from a healthy process.
        symbolize(req);
        char done = 1;
        ssize_t written = write(ack_fd, &done, 1);
        (void)written;
    }
}

}  // namespace

void stop_trace_helper() {
    // Tell it to leave, rather than relying on EOF.
    //
    // EOF is not reliable here and the reason is worth writing down: both
    // jit_utils_core and jittor_core carry their own copy of these statics, so
    // two helpers get started. The second one is forked from a process that
    // already holds the *parent* end of the first one's pipe, and FD_CLOEXEC
    // does not cover fork. Helper B therefore keeps helper A's write end open,
    // A never sees EOF, and a wait for it to leave on its own always runs out.
    // An explicit message does not care who else holds the pipe.
    if (trace_request_fd >= 0) {
        TraceRequest bye;
        memset(&bye, 0, sizeof(bye));
        bye.magic = 0;
        ssize_t written = write(trace_request_fd, &bye, sizeof(bye));
        (void)written;
        close(trace_request_fd);
        trace_request_fd = -1;
    }
    if (trace_ack_fd >= 0) { close(trace_ack_fd); trace_ack_fd = -1; }
    if (trace_helper_pid > 0) {
        for (int waited_ms = 0; waited_ms < 2000; waited_ms += 20) {
            if (waitpid(trace_helper_pid, nullptr, WNOHANG) != 0) {
                trace_helper_pid = -1;
                return;
            }
            usleep(20 * 1000);
        }
        // Backstop. SIGTERM and not SIGKILL on purpose: the SIGCHLD reporter in
        // log.cc deliberately says nothing about a child that was terminated,
        // so a slow helper does not print a scary line on every clean exit.
        kill(trace_helper_pid, SIGTERM);
        waitpid(trace_helper_pid, nullptr, 0);
        trace_helper_pid = -1;
    }
}

EXTERN_LIB vector<void(*)()> cleanup_callback;

void start_trace_helper() {
    if (trace_helper_pid > 0) return;
    int req[2], ack[2];
    if (pipe(req)) return;
    if (pipe(ack)) { close(req[0]); close(req[1]); return; }
    // Close-on-exec on every end. Without it each compiler subprocess, each
    // dataloader worker, each `system()` jittor runs inherits the write end of
    // the request pipe -- and the helper, which leaves on EOF, never sees one.
    // The parent end stays open in this process only.
    for (int fd : {req[0], req[1], ack[0], ack[1]})
        fcntl(fd, F_SETFD, fcntl(fd, F_GETFD) | FD_CLOEXEC);
    // Force libgcc's lazy unwinder load now, so backtrace() in the handler does
    // not dlopen (and allocate) at the worst possible moment.
    void* warm[2];
    backtrace(warm, 2);
    pid_t pid = fork();
    if (pid < 0) {
        close(req[0]); close(req[1]); close(ack[0]); close(ack[1]);
        return;
    }
    if (pid == 0) {
        close(req[1]); close(ack[0]);
        // The helper runs system() for addr2line; clear CLOEXEC nowhere, the
        // read end is all it needs and its children get neither end.
        helper_main(req[0], ack[1]);
        _exit(0);
    }
    close(req[0]); close(ack[1]);
    trace_request_fd = req[1];
    trace_ack_fd = ack[0];
    trace_helper_pid = pid;
#if defined(__linux__) && defined(PR_SET_PTRACER)
    // So the helper may attach gdb to us when gdb_path is configured.
    prctl(PR_SET_PTRACER, pid, 0, 0, 0);
#endif
    cleanup_callback.push_back(&stop_trace_helper);
}

// Called from the signal handler. Does only what is allowed there.
void print_trace_from_signal(int signal, void* fault_pc, void* caller_pc) {
    if (trace_request_fd < 0) {
        // No helper (not started, or fork failed). Print the raw addresses --
        // useless-looking but not useless: `addr2line -e <module> <offset>`
        // offline still resolves them, and it beats deadlocking in the
        // allocator to produce something prettier.
        trace_write(2, "[bt] no symbolizer helper; raw addresses follow\n");
        if (fault_pc) { trace_write(2, "[bt] fault pc  "); trace_write_hex(2, (uint64_t)fault_pc); trace_write(2, "\n"); }
        if (caller_pc) { trace_write(2, "[bt] caller pc "); trace_write_hex(2, (uint64_t)caller_pc); trace_write(2, "\n"); }
        void* frames[JT_TRACE_MAX];
        int n = backtrace(frames, JT_TRACE_MAX);
        for (int i = 0; i < n; i++) {
            trace_write(2, "[bt] #");
            trace_write_dec(2, i);
            trace_write(2, " ");
            trace_write_hex(2, (uint64_t)frames[i]);
            trace_write(2, "\n");
        }
        return;
    }
    TraceRequest req;
    req.magic = JT_TRACE_MAGIC;
    req.pid = (int32_t)getpid();
    req.signal = signal;
    req.fault_pc = fault_pc;
    req.caller_pc = caller_pc;
    req.depth = backtrace(req.frames, JT_TRACE_MAX);
    ssize_t written = write(trace_request_fd, &req, sizeof(req));
    if (written != (ssize_t)sizeof(req)) return;
    // Bounded wait: a helper that wedges must not turn a crash into a hang,
    // which is the failure this whole change exists to remove.
    struct pollfd pfd;
    pfd.fd = trace_ack_fd;
    pfd.events = POLLIN;
    if (poll(&pfd, 1, gdb_trace_timeout > 0 ? gdb_trace_timeout * 1000 : 30000) > 0) {
        char done = 0;
        ssize_t got = read(trace_ack_fd, &done, 1);
        (void)got;
    } else {
        trace_write(2, "[bt] symbolizer helper timed out\n");
    }
}

void print_trace() {
    if (gdb_path.size()) {
        // using gdb to print the stack trace
        char pid_buf[30];
        sprintf(pid_buf, "%d", getpid());
        char st_buf[30];
        sprintf(st_buf, "set backtrace limit %d", trace_depth);

        LOGi << "stack trace for pid=" << pid_buf;

        vector<const char*> argv{
            gdb_path.c_str(), "--batch", "-n",
            "-ex", "thread",
            "-ex", st_buf, // "set backtrace limit 10",
            "-ex", "bt",
        };
        if (has_pybt)
            argv.insert(argv.end(), {"-ex", "set backtrace limit 0", "-ex", "py-bt"});
        if (auto n = extra_gdb_cmd.size()) {
            _extra_gdb_cmd = extra_gdb_cmd;
            _extra_gdb_cmd += '\0';
            argv.push_back("-ex");
            argv.push_back(&_extra_gdb_cmd[0]);
            for (uint i=0; i<n; i++) {
                if (_extra_gdb_cmd[i]==';') {
                    argv.push_back("-ex");
                    _extra_gdb_cmd[i] = '\0';
                    argv.push_back(&_extra_gdb_cmd[i+1]);
                }
            }
        }
        argv.insert(argv.end(), {"-p", pid_buf, NULL});
        #ifndef _WIN32
        // This child is ours and is reaped below. Left alone, killing a wedged
        // debugger delivers SIGCHLD to Jittor's own handler, which reads any
        // non-clean child death as an out-of-memory worker and exits the whole
        // process -- turning the bounded wait into a hard stop. Blocking the
        // signal is not enough: sigprocmask only covers the calling thread, and
        // the signal would land on any other thread. Restore the default
        // disposition, which discards SIGCHLD while still leaving the child
        // reapable, and put the handler back afterwards.
        struct sigaction default_child = {}, previous_child;
        default_child.sa_handler = SIG_DFL;
        sigemptyset(&default_child.sa_mask);
        sigaction(SIGCHLD, &default_child, &previous_child);

        int child_pid = fork();
        if (!child_pid) {
            execvp(gdb_path.c_str(), (char* const*)&argv[0]);
            exit(0);
        } else {
            // allow children ptrace parent
#if defined(__linux__) && defined(PR_SET_PTRACER)
    		prctl(PR_SET_PTRACER, child_pid, 0, 0, 0);
#endif
            // A GDB that hangs, or that is itself intercepted by a crash
            // reporter, used to block this process forever and take a whole
            // test or training run with it.  Wait for a bounded time, then
            // stop the debugger and continue with the diagnosis we have.
            if (gdb_trace_timeout <= 0) {
                waitpid(child_pid, NULL, 0);
            } else {
                int waited_ms = 0;
                const int limit_ms = gdb_trace_timeout * 1000;
                const int step_ms = 50;
                bool reaped = false;
                while (waited_ms < limit_ms) {
                    int status = 0;
                    int done = waitpid(child_pid, &status, WNOHANG);
                    if (done == child_pid || done < 0) { reaped = true; break; }
                    usleep(step_ms * 1000);
                    waited_ms += step_ms;
                }
                if (!reaped) {
                    std::cerr << "[bt] gdb backtrace timed out after "
                        << gdb_trace_timeout << "s, terminating pid "
                        << child_pid << std::endl;
                    kill(child_pid, SIGKILL);
                    waitpid(child_pid, NULL, 0);
                }
            }
            sigaction(SIGCHLD, &previous_child, nullptr);
        }
        #else
        auto cmds = get_cmds(argv);
        LOGv << cmds;
        system_popen(cmds.c_str());
        #endif
    }
#ifndef _WIN32
    else {
        void *trace[16];
        char **messages = (char **)NULL;
        int i, trace_size = 0;

        trace_size = backtrace(trace, 16);
        messages = backtrace_symbols(trace, trace_size);
        // skip first stack frame (points here)
        std::cerr << "[bt] Execution path:" << std::endl;
        for (i=1; i<trace_size; ++i) {
            std::cerr << "[bt] #" << i << " " << messages[i] << std::endl;
            // find first occurence of '(' or ' ' in message[i] and assume
            // everything before that is the file name.
            int p = 0;
            while(messages[i][p] != '(' && messages[i][p] != ' '
                    && messages[i][p] != 0)
                ++p;

            if (!trace[i]) continue;
            if (!addr2line_path.size()) continue;
            // Size the command from the arguments. The module name here is a
            // path into the JIT cache and routinely passes 200 characters, so
            // formatting into a fixed 256-byte buffer overflowed it and glibc's
            // fortify check aborted the process -- inside the backtrace helper,
            // which is the worst possible place to die.
            const char* format = "%s %p -f -p -i -e %.*s";
            int needed = snprintf(nullptr, 0, format,
                addr2line_path.c_str(), trace[i], p, messages[i]);
            if (needed < 0) continue;
            vector<char> syscom(needed + 1);
            snprintf(syscom.data(), syscom.size(), format,
                addr2line_path.c_str(), trace[i], p, messages[i]);
            //last parameter is the file name of the symbol
            int ret = system(syscom.data());
            (void)ret;
        }
    }
#endif
}

} // jittor
