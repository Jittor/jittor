// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "opencl_warper.h"

namespace jittor {

cl_context opencl_context;
cl_command_queue opencl_queue;

struct opencl_initer {

inline opencl_initer() {
    cl_int err;
    cl_uint num;
    err = clGetPlatformIDs(0, 0, &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platforms\n";
        return;
    }
    std::vector<cl_platform_id> platforms(num);
    err = clGetPlatformIDs(num, &platforms[0], &num);
    if(err != CL_SUCCESS) {
        std::cerr << "Unable to get platform ID\n";
        return;
    }
    cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[0]), 0 };
    opencl_context = clCreateContextFromType(prop, CL_DEVICE_TYPE_DEFAULT, NULL, NULL, NULL);
    if(opencl_context == 0) {
        std::cerr << "Can't create opencl_context\n";
        return;
    }
    size_t cb;
    clGetContextInfo(opencl_context, CL_CONTEXT_DEVICES, 0, NULL, &cb);
    std::vector<cl_device_id> devices(cb / sizeof(cl_device_id));
    clGetContextInfo(opencl_context, CL_CONTEXT_DEVICES, cb, &devices[0], 0);
    opencl_queue = clCreateCommandQueue(opencl_context, devices[0], 0, 0);
    if(opencl_queue == 0) {
        std::cerr << "Can't create command queue\n";
        clReleaseContext(opencl_context);
        return;
    }
    LOGv << "opencl create finished";
}

inline ~opencl_initer() {
    clReleaseCommandQueue(opencl_queue);
    clReleaseContext(opencl_context);
    LOGv << "opencl destroy finished";
}

} init;

}

