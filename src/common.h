// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <stddef.h>
#include <memory>
#include <functional>
#include "utils/log.h"

#define JIT_TEST(name) extern "C" void jit_test_ ## name ()
void expect_error(std::function<void()> func);

#define VAR_MEMBER_NAME_AND_OFFSET(name, op) { #name , offsetof(struct op, name) }
#define GET_VAR_MEMBER(op, offset) (*((Var**)(((char*)(op))+(offset))))

#ifdef __clang__
#pragma clang diagnostic ignored "-Winvalid-offsetof"
#pragma clang diagnostic ignored "-Wtautological-compare"
#else
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wdiv-by-zero"
#endif
#endif