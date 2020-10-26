// ***************************************************************
// Copyright (c) 2020 Jittor.  All Rights Reserved.
// Authors: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "var_slices.h"
#include "var.h"

namespace jittor {

std::ostream& operator<<(std::ostream& os, const VarSlices& vs) {
    os << '[';
    for (int i=0; i<vs.n; i++)
        os << vs.slices[i] << ",";
    return os << ']';
}

std::ostream& operator<<(std::ostream& os, const VarSlice& s) {
    if (s.is_var()) return os << s.var->dtype() << s.var->shape;
    if (s.is_ellipsis()) return os << "...";
    if (s.is_slice()) return os << s.slice;
    if (s.is_int()) return os << s.i;
    return os << "-";
}

std::ostream& operator<<(std::ostream& os, const Slice& s) {
    if (!(s.mask & 1)) os << s.start; 
    os << ':';
    if (!(s.mask & 2)) os << s.stop;
    os << ':';
    if (!(s.mask & 4)) os << s.step; 
    return os;
}

} // jittor
