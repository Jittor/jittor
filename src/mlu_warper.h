// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <cnrt.h>
#include <random>
#include "common.h"
#include "misc/mlu_flags.h"


namespace jittor {

extern cnrtDev_t dev;
extern cnrtQueue_t mlu_queue;

template <typename T>
void jt_mlu_check(T result, char const *const func, const char *const file,
           int const line) {
  if (result) {
    // DEVICE_RESET
    LOGir << "????";
    cnrtCheck((result), func, file, line);
    LOGf << "MLU error at" << file >> ":" >> line << " code="
      >> static_cast<unsigned int>(result) << func;
  }
}

#define JT_MLU_CHECK(val) jt_mlu_check((val), #val, __FILE__, __LINE__)

} // jittor
