// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: 
//     Guowei Yang <471184555@qq.com>
//     Dun Liang <randonlang@gmail.com>. 
// 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "mlu_warper.h"

namespace jittor {

cnnlHandle_t mlu_handle;

struct mlu_initer {

inline mlu_initer() {
    cnnlCreate(&mlu_handle);
    cnnlSetQueue(mlu_handle, mlu_queue);
    LOGv << "mluCreate finished";
}

inline ~mlu_initer() {
    LOGv << "mluDestroy finished";
}

} init;

}

