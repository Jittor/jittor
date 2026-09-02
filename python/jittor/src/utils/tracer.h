// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

void print_trace();
// Signal-handler-safe: hands the frames to a process forked before
// the crash. See tracer.cc.
void print_trace_from_signal(int signal, void* fault_pc, void* caller_pc);
void start_trace_helper();
void stop_trace_helper();
void breakpoint();

} // jittor