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
// Python wrapper to import a non-contiguous (strided) producer -- see
// dlpack.cc for the full design. Omitting them (the default) is 100%
// unchanged, contiguous-only behavior.
// @pyjt(from_dlpack_capsule)
VarHolder* from_dlpack_capsule(PyObject* capsule, PyObject* flat_len=nullptr, int64 elem_offset=0);

// Read-only inspection of a live (not-yet-consumed) DLPack capsule's shape
// and strides, without renaming the capsule or touching its deleter -- lets
// from_dlpack() decide (in Python) whether the contiguous fast path or the
// strided materialize-a-copy path applies, before committing to consuming
// the capsule exactly once via from_dlpack_capsule().
// @pyjt(dlpack_peek)
PyObject* dlpack_peek(PyObject* capsule);

} // jittor
