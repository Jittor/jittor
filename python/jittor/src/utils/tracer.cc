// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include "utils/cross_platform.h"
#include "utils/tracer.h"

namespace jittor {
    
DEFINE_FLAG_WITH_SETTER(string, gdb_path, "", "Path of GDB.");
DEFINE_FLAG(string, addr2line_path, "", "Path of addr2line.");
DEFINE_FLAG(string, extra_gdb_cmd, "", "Extra command pass to GDB, seperate by(;) .");
DEFINE_FLAG(int, has_pybt, 0, "GDB has pybt or not.");
DEFINE_FLAG(int, trace_depth, 10, "trace depth for GDB.");
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
        int child_pid = fork();
        if (!child_pid) {
            execvp(gdb_path.c_str(), (char* const*)&argv[0]);
            exit(0);
        } else {
            // allow children ptrace parent
#if defined(__linux__) && defined(PR_SET_PTRACER)
    		prctl(PR_SET_PTRACER, child_pid, 0, 0, 0);
#endif
            waitpid(child_pid,NULL,0);
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
            char syscom[256];
            sprintf(syscom,"%s %p -f -p -i -e %.*s", addr2line_path.c_str(), trace[i], p, messages[i]);
            //last parameter is the file name of the symbol
            // printf("run '%s'\n", syscom);
            int ret = system(syscom);
            (void)ret;
        }
    }
#endif
}

} // jittor
