// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "cudnn_warper.h"

namespace jittor {

cudnnHandle_t cudnn_handle;

struct cudnn_initer {

inline cudnn_initer() {
    checkCudaErrors(cudnnCreate(&cudnn_handle));
    LOGv << "cudnnCreate finished";
}

inline ~cudnn_initer() {
    checkCudaErrors(cudnnDestroy(cudnn_handle));
    LOGv << "cudnnDestroy finished";
}

} init;

} // jittor
