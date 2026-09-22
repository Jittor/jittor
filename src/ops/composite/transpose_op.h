// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/op.h"

namespace jittor {

struct TransposeOp : Op {
    Var* x, * y;
    NanoVector axes;
    TransposeOp(Var* x, NanoVector axes=NanoVector());

    // A permutation moves no data: it is the same allocation read with the
    // axes' strides swapped, which is what torch returns and what makes a
    // write through the result reach its base. Off by default because every
    // consumer then sees a non-dense input -- correct (elementwise, reduce,
    // broadcast and reindex all read `storage_strides`, and a slice already
    // reaches `.numpy()`, `.cpu()` and the fused kernels this way today), but
    // not necessarily as fast as materialising once for a cuBLAS or cuTT path
    // that wants dense memory. Flip it with `transpose_storage_view=1` and
    // benchmark before making it the default.
    bool storage_view = false;
    bool is_storage_view() const override { return storage_view; }

    const char* name() const override { return "transpose"; }
    VarPtr grad(Var* out, Var* dout, Var* v, int v_index) override;
    void infer_shape() override;
    DECLARE_jit_run;
};

} // jittor