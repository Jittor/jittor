// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/var.h"

namespace jittor {

// Elementwise ops follow their operands' memory layout.
//
// A tensor whose storage is a permutation of a dense one -- a channels-last
// activation read as NCHW, which is what a half-precision convolution hands
// out -- used to lose that layout at the first elementwise op: the kernel read
// it strided and wrote a dense NCHW result, and the next convolution paid for
// the layout it wanted all over again (cuDNN converts NCHW input to NHWC
// through a workspace: 147 MB for one ResNet-50 layer at batch 64). PyTorch
// keeps the input's layout through elementwise work; this does the same. The
// op runs on the dense source and the result is the same permutation of its
// output, as a storage view, so it moves no data.
//
// Returns true and fills `sources` -- the operands expressed in the source's
// order, all of one shape -- and `axes` when every operand is either such a
// view with one and the same permutation or a broadcast that can be expressed
// in that order for free. Any other operand declines: turning a lazily
// computed dense tensor into a strided view of itself would force it to be
// materialised.
bool storage_layout_operands(const vector<Var*>& inputs, NanoVector& axes,
                             vector<VarPtr>& sources);

// `v` permuted by `axes` as a view of its storage.
VarPtr storage_view_transpose(Var* v, NanoVector axes);

} // jittor
