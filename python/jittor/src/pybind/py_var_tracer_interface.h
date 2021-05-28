// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>. 
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
