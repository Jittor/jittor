// ***************************************************************
// Copyright (c) 2020 Jittor. All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>

namespace jittor {

// @pyjt(dump_trace_data)
PyObject* dump_trace_data();

// @pyjt(clear_trace_data)
void clear_trace_data();

} // jittor
