// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "clblas_warper.h"
#include "misc/cuda_flags.h"

namespace jittor {

struct clblas_initer {

inline clblas_initer() {
    if (!get_device_count()) return;

    /* Setup clblas. */
    clblasSetup();
    LOGv << "clblasCreate finished";
}

inline ~clblas_initer() {
    if (!get_device_count()) return;

    /* Finalize work with clblas. */
    clblasTeardown();
    LOGv << "clblasDestroy finished";
}

} init;

} // jittor
