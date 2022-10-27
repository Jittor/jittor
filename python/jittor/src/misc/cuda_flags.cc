// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************

#include "common.h"
#ifdef HAS_CUDA
#include <cuda_runtime.h>
#ifdef __linux__
#include <fstream>
#include <unistd.h>
#endif
#endif

namespace jittor {

DEFINE_FLAG_WITH_SETTER(int, use_cuda, 0,
    "Use cuda or not. 1 for trying to use cuda, 2 for forcing to use cuda.");
DEFINE_FLAG_WITH_SETTER(int, device_id, -1,
    "number of the device to used");

EXTERN_LIB void sync_all(bool device_sync);

void setter_use_cuda(int value) {
    if (use_cuda == value) return;
#ifdef HAS_CUDA
    if (value) {
        int count=0;
        cudaGetDeviceCount(&count);
        if (count == 0) {
            if (getenv("CUDA_VISIBLE_DEVICES")) {
                LOGf << "No device found, please unset your "
                "enviroment variable 'CUDA_VISIBLE_DEVICES'";
            } else
                LOGf << "No device found";
        }
        LOGi << "CUDA enabled.";
    } else {
        LOGv << "CUDA disabled.";
    }
#else
    CHECK(value==0) << "No CUDA found.";
#endif
    if (use_cuda != value)
        sync_all(0);
}

void setter_device_id(int value) {
#if defined(HAS_CUDA) && defined(__linux__)
    // case1: set env device_id, not restart
    // case2: set in python, restart
    // case3: restart, device id and CUDA env set both
    if (value<0)
        return;
    int count=0;
    cudaGetDeviceCount(&count);
    auto s = getenv("CUDA_VISIBLE_DEVICES");
    auto s2 = getenv("device_id");
    auto sv = std::to_string(value);
    if (s2 && s2 == sv && (!s || count!=1)) {
        // only handle case1 and case3(not cuda)
        LOGi << "change to device #" >> value;
        cudaSetDevice(value);
        return;
    }
    if (s && s == sv)
        return;
    setenv("CUDA_VISIBLE_DEVICES", sv.c_str(), 1);
    setenv("device_id", sv.c_str(), 1);
    std::ifstream ifs("/proc/self/cmdline");
    if (!(ifs && ifs.good())) return;
    string cmd((std::istreambuf_iterator<char>(ifs)),
               (std::istreambuf_iterator<char>()));
    vector<char*> ss;
    auto cstr = (char*)cmd.c_str();
    ss.push_back(cstr);
    for (int i=0; i<cmd.size(); i++)
        if (cstr[i] == '\0')
            ss.push_back(&cstr[i+1]);
    ss.pop_back();
    ss.push_back(nullptr);
    LOGi << "[restart] change to device #" >> value;
    execvp(ss[0], &ss[0]);
    ss.pop_back();
    LOGe << "restart failed" << ss;
#endif
}

} // jittor