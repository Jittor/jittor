// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <sys/prctl.h>
#include <execinfo.h>
#include <iostream>
#include "utils/tracer.h"

namespace jittor {
    
DEFINE_FLAG_WITH_SETTER(string, gdb_path, "", "Path of GDB.");
DEFINE_FLAG(string, addr2line_path, "", "Path of addr2line.");
DEFINE_FLAG(string, extra_gdb_cmd, "", "Extra command pass to GDB, seperate by(;) .");
DEFINE_FLAG(int, has_pybt, 0, "GDB has pybt or not.");
DEFINE_FLAG(int, trace_depth, 10, "trace depth for GDB.");
DEFINE_FLAG_WITH_SETTER(int, gdb_attach, 0, "gdb attach self process.");

string _extra_gdb_cmd;

void setter_gdb_attach(int v) {
    if (v && gdb_path.size()) {
        static int gdb_attached = 0;
        if (gdb_attached) return;
        gdb_attached = 1;
        // using gdb to print the stack trace
        char pid_buf[30];
        sprintf(pid_buf, "%d", getpid());
        char name_buf[512];
        name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;
        int child_pid = fork();
        if (!child_pid) {
            LOGi << "gdb attach for" << name_buf << "pid=" >> pid_buf;

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
            argv.insert(argv.end(), {name_buf, pid_buf, NULL});
            auto ret = execvp(gdb_path.c_str(), (char* const*)&argv[0]);
            LOGf << "execvp failed return" << ret << gdb_path << extra_gdb_cmd;
            exit(1);
        } else {
            // allow children ptrace parent
    		prctl(PR_SET_PTRACER, child_pid, 0, 0, 0);
            // sleep 5s, wait gdb attach
            sleep(5);
        }
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
        char name_buf[512];
        name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;
        int child_pid = fork();
        if (!child_pid) {
            std::cerr << "stack trace for " << name_buf << " pid=" << pid_buf << std::endl;

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
            argv.insert(argv.end(), {name_buf, pid_buf, NULL});
            execvp(gdb_path.c_str(), (char* const*)&argv[0]);
            exit(0);
        } else {
            // allow children ptrace parent
    		prctl(PR_SET_PTRACER, child_pid, 0, 0, 0);
            waitpid(child_pid,NULL,0);
        }
    } else {
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
}

} // jittor