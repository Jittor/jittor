// ***************************************************************
// Copyright (c) 2021 Jittor.  All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "misc/nano_vector.h"
#include "var.h"

namespace jittor {

struct Slice;

union VarSlice {
    Slice slice;
    Var* var;
    int64 i;
    inline bool is_var() const { return slice.mask == -1; }
    inline bool is_ellipsis() const { return slice.mask == -2; }
    inline bool is_none() const { return slice.mask == -3; }
    inline bool is_int() const { return slice.mask == -4; }
    inline bool is_str() const { return slice.mask == -5; }
    inline bool is_slice() const { return slice.mask >= 0; }
    inline void set_var(Var* v) { slice.mask = -1; var = v; }
    inline void set_ellipsis() { slice.mask = -2; }
    inline void set_none() { slice.mask = -3; }
    inline void set_int(int64 v) { slice.mask = -4; i = v; }
    inline void set_str(const string& s) {
        slice.mask = -5;
        CHECK(s.size() < 16) << "String slice too long" << s;
        auto v = (int64*)s.c_str();
        slice.start = v[0];
        slice.stop = v[1];
        slice.step = s.size();
    }
    inline char* get_str() {return (char*)this;}
};

struct VarSlices  {
    VarSlice* slices;
    int n;
    inline VarSlices() : slices(nullptr) {}
    inline VarSlices(int n) : slices(new VarSlice[n]), n(n) {}
    inline ~VarSlices() {if (slices) delete[] slices;}
    inline VarSlices(VarSlices&& other) : slices(other.slices), n(other.n) {
        other.slices = nullptr;
    }
    inline VarSlices(const VarSlices& other) : slices(new VarSlice[other.n]), n(other.n) {
        for (int i=0; i<n; i++)
            slices[i] = other.slices[i];
    }
    inline void operator=(VarSlices&& other) {
        if (slices) delete[] slices;
        n = other.n;
        slices = other.slices;
        other.slices = nullptr;
    }
    inline void operator=(const VarSlices& other) {
        if (slices) delete[] slices;
        slices = new VarSlice[other.n];
        n = other.n;
        for (int i=0; i<n; i++)
            slices[i] = other.slices[i];
    }
};

std::ostream& operator<<(std::ostream& os, const VarSlices& vs);
std::ostream& operator<<(std::ostream& os, const VarSlice& s);
std::ostream& operator<<(std::ostream& os, const Slice& s);

// @pyjt(_print_var_slice)
inline void _print_var_slice(VarSlices&& vs) {
    LOGi << vs;
}

} // jittor
