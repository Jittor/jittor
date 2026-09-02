// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers:
//     Dun Liang <randonlang@gmail.com>.
//
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once

/**
The single canonical list of dtypes jittor's collective backends know about.

Every backend (MPI, NCCL, HCCL) expands this list exactly once, with its own
per-dtype mapping, instead of keeping a hand-written table inside each operator
file. Before this existed there were twelve copies of "which library type does
this jittor dtype map to" (3 MPI ops + 2 MPI wrapper helpers + 5 NCCL ops +
4 HCCL ops) and they had already drifted: the MPI operator tables mapped int64
to MPI_DOUBLE_INT -- a MAXLOC (double,int) pair, 12 or 16 bytes wide, not an
integer type at all -- so an int64 all-reduce read past the end of its input
and returned garbage, while the MPI wrapper's own table two hundred lines away
had it right.

A backend that genuinely has no type for an entry maps it to a
`<backend>_dtype_unsupported(dtype)` helper, so the hole is declared in the
table and reported as a clear error at run time rather than silently expanding
to nothing.

Usage:

    #define MY_CASE(T) if (dtype == ns_##T) return MY_DTYPE_##T;
    JT_COLLECTIVE_DTYPES(MY_CASE)
    #undef MY_CASE
*/
#define JT_COLLECTIVE_DTYPES(F) \
    F(float16) \
    F(bfloat16) \
    F(float32) \
    F(float64) \
    F(int16) \
    F(int32) \
    F(int64) \
    F(uint8)
