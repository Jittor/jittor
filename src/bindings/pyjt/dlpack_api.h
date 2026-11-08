// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// DLPack module-level free functions, mirroring
// torch.utils.dlpack.to_dlpack(tensor)/from_dlpack(capsule)'s calling
// convention. Kept in a small standalone header (rather than declared
// inline in dlpack.cc) because pyjt_compiler.py's `compile()` only scans
// src/**/*.h for `@pyjt(...)` annotations, never .cc files -- see
// core/var_holder.h/VarHolder::dlpack() for the per-tensor method form.
#pragma once
#include "core/common.h"
#include "core/var_holder.h"

namespace jittor {

// @pyjt(to_dlpack)
PyObject* to_dlpack(VarHolder* v);

// `flat_len`/`elem_offset` are an internal detail used by from_dlpack()'s
// Python wrapper to import a producer whose strides Jittor's layout cannot
// alias (a negative stride) -- see dlpack.cc. Without them a contiguous or
// non-negatively strided producer is aliased. `producer_ordered` says the
// producer was handed this consumer's stream (`dlpack_consumer_stream`), so
// its work is already ordered before Jittor's and no device sync is needed.
// @pyjt(from_dlpack_capsule)
VarHolder* from_dlpack_capsule(PyObject* capsule, PyObject* flat_len=nullptr, int64 elem_offset=0,
                               bool producer_ordered=false);

// The stream a CUDA producer should order its work before, as the int a
// consumer passes to `__dlpack__(stream=...)`: Jittor's compute stream on
// `device_id` (1, the legacy default stream, when that is the null stream).
// @pyjt(dlpack_consumer_stream)
int64 dlpack_consumer_stream(int device_id);

// Read-only inspection of a live (not-yet-consumed) DLPack capsule: its
// shape, its strides, and whether from_dlpack_capsule() can alias that
// layout -- without renaming the capsule or touching its deleter, so
// from_dlpack() can choose the aliasing import or the copy before consuming
// the capsule exactly once via from_dlpack_capsule().
// @pyjt(dlpack_peek)
PyObject* dlpack_peek(PyObject* capsule);

} // jittor
