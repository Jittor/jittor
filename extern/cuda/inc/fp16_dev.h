/**
* Copyright 2014 NVIDIA Corporation.  All rights reserved.
*
* Please refer to the NVIDIA end user license agreement (EULA) associated
* with this source code for terms and conditions that govern your use of
* this software. Any use, reproduction, disclosure, or distribution of
* this software and related documentation outside the terms of the EULA
* is strictly prohibited.
*
*/

#if !defined(_FP16_DEV_H_)
#define _FP16_DEV_H_

#include "fp16_emu.h"

template <class value_type>
void gpu_float2half_rn(int size, const value_type *buffIn, half1 *buffOut);

#endif // _FP16_DEV_H_

