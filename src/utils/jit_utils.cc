// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "utils/cache_compile.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef __clang__
#pragma clang diagnostic ignored "-Wdefaulted-function-deleted"
#endif
#ifdef __GNUC__
#endif
#include <pybind11/iostream.h>
#include <sys/prctl.h>
#include <signal.h>

namespace jittor {

void init_subprocess() {
    prctl(PR_SET_PDEATHSIG, SIGKILL);
}

}

PYBIND11_MODULE(jit_utils_core, m) {
    pybind11::add_ostream_redirect(m, "ostream_redirect");
    m.def("cache_compile", &jittor::jit_compiler::cache_compile);
    m.def("log", [&](
        const std::string& fileline,
        char level,
        int verbose,
        const std::string& s)
    {
        // return if verbose level not match
        if (level=='i' && !(
            jittor::log_vprefix.size() ?
                jittor::check_vlog(fileline.c_str(), verbose) :
                verbose <= jittor::log_v))
            return;
        if (level != 'f')
            jittor::LogVoidify() &&
            jittor::Log(fileline.c_str(), level, verbose) << s;
        else
            jittor::LogFatalVoidify() &&
            jittor::Log(fileline.c_str(), level, verbose) << s;
    });
    m.def("log_capture_start", &jittor::log_capture_start);
    m.def("log_capture_stop", &jittor::log_capture_stop);
    m.def("log_capture_read", &jittor::log_capture_read);
    m.def("init_subprocess", &jittor::init_subprocess);
}
