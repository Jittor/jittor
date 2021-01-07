// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

static inline int lzcnt(int64 v) {
    #ifdef __clang__
    #if __has_feature(__builtin_ia32_lzcnt_u64)
        return __builtin_ia32_lzcnt_u64(v);
    #else
        return v ? __builtin_clzll(v) : 64;
    #endif
    #else
        return __builtin_clzll(v);
    #endif
}

struct Slice {
    int64 start, stop, step, mask;
    inline void fill(int64 size) {
        if (step>0) {
            if (mask&2) 
                stop = size;
            else if (stop<0)
                stop += size;
            else
                stop = std::min(size, stop);
        } else {
            if (mask&1) start = size-1;
            if (mask&2)
                stop = -1;
            else if (stop<0)
                stop = std::max((int64)0, stop+size);
        }
        if (start<0) start += size;
        mask = 0;
        ASSERT(start==stop || (start>=0 && stop>=-1 && start<size && stop<=size))
            << "slice overflow:" << start << stop << step;
    }
};

// return a / ceil_to_2_pow(b)
inline uint64 fast_div(uint64 a, uint64 b) {
    return a >> (64 - lzcnt(b));
}

// @pyjt(NanoVector)
struct NanoVector {
    int64 data=0, offset=0;

    enum {
        size_nbits=4,
        offset_nbits=6,
    };

    // @pyjt(__init__)
    inline NanoVector() {}
    inline NanoVector(std::nullptr_t) {}
    // @pyjt(__init__)
    inline NanoVector(const NanoVector& nv) : data(nv.data), offset(nv.offset) {}
    
    inline void clear() { data = offset = 0; }

    // @pyjt(__len__, __map_len__)
    inline int size() const {
        return offset & ((1<<size_nbits)-1);
    }

    static inline int get_nbits(int64 v) {
        // return 1bit if v in [-1,0]
        // return 2bit if v in [-2,1,0,1]
        if (v<0) v = ~v;
        return 65 - lzcnt(v);
    }

    inline int get_offset(int i) const {
        return (offset >> (size_nbits+i*offset_nbits)) 
            & ((1<<offset_nbits)-1);
    }

    inline int total_bits() const {
        int s = size();
        return s ? get_offset(s-1) : 0;
    }

    inline void set_offset(int i, int next_offset) {
        offset |= ((uint64)next_offset) << (i*offset_nbits+size_nbits);
    }

    inline void set_data(int64 v, int nbits, int pre_offset) {
        // don't have to clean bit if init
        // data &= ~(((1<<nbits)-1) << pre_offset);
        data |= (v & ((1ll<<nbits)-1)) << pre_offset;
    }

    inline void push_back(int64 v) {
        auto s = size();
        offset ++;
        auto nbits = get_nbits(v);
        int pre_offset = s ? get_offset(s-1) : 0;
        int next_offset = pre_offset+nbits;
        set_offset(s, next_offset);
        set_data(v, nbits, pre_offset);
    }

    // @pyjt(append)
    inline void push_back_check_overflow(int64 v) {
        auto s = size();
        auto nbits = get_nbits(v);
        int pre_offset = s ? get_offset(s-1) : 0;
        int next_offset = pre_offset+nbits;
        ASSERT(s<10 && next_offset<=64);
        offset ++; 
        set_offset(s, next_offset);
        set_data(v, nbits, pre_offset);
    }

    // @pyjt(__getitem__, __map_getitem__)
    inline int64 at(int i) const {
        if (i<0) i+= size();
        ASSERT(i>=0 && i<size());
        int pre_offset = i ? get_offset(i-1) : 0;
        int next_offset = get_offset(i);
        int nbits = next_offset - pre_offset;
        return (data << ((64-next_offset)&63))
            >> ((64-nbits)&63);
    }

    // @pyjt(__map_getitem__)
    inline NanoVector slice(Slice slice) {
        slice.fill(size());
        NanoVector v;
        if (slice.step>0) {
            for (int i=slice.start; i<slice.stop; i+=slice.step)
                v.push_back(this->operator[](i));
        } else {
            for (int i=slice.start; i>slice.stop; i+=slice.step)
                v.push_back(this->operator[](i));
        }
        return v;
    }

    inline int64 operator[](int i) const {
        int pre_offset = i ? get_offset(i-1) : 0;
        int next_offset = get_offset(i);
        int nbits = next_offset - pre_offset;
        return (data << ((64-next_offset)&63))
            >> ((64-nbits)&63);
    }

    // @pyjt(__init__)
    inline NanoVector(const vector<int64>& v) {
        for (auto a : v) push_back_check_overflow(a);
    }

    inline static NanoVector make(const int64* v, int n) {
        NanoVector nv;
        for (int i=0; i<n; i++) nv.push_back_check_overflow(v[i]);
        return nv;
    }

    inline static NanoVector make(const int32* v, int n) {
        NanoVector nv;
        for (int i=0; i<n; i++) nv.push_back_check_overflow(v[i]);
        return nv;
    }

    inline NanoVector(int64 x) { push_back(x); }

    // @pyjt(__repr__)
    inline string to_string() {
        string s="[";
        for (int i=0; i<size(); i++) {
            s += S(at(i));
            s += ',';
        }
        s += ']';
        return s;
    }

    template<class... Args>
    NanoVector(Args... args) {
        auto f = [&](int64 c) { push_back(c); };
        // Brace-enclosed initializers
        int dummy[] = {(f(args), 0)...};
        (void)dummy;
    }

    struct Iter {
        const NanoVector* self;
        int index;
        inline int64 operator*() { return self->at(index); }
        inline Iter& operator++() { index++; return *this; }
        inline Iter operator+(int i) { return {self, i+index}; }
        inline bool operator!=(Iter& other) { return index != other.index; }
    };

    inline Iter begin() { return {this, 0}; }
    inline Iter end() { return {this, size()}; }

    inline void pop_back() { offset--; data &= (1ll<<total_bits())-1; }
    inline void push_back(Iter s, Iter t) {
        while (s != t) push_back(*s), ++s;
    }
    inline void reserve(int64 max_value, int n) {
        auto nbits = get_nbits(max_value);
        ASSERT(n<=10 && nbits*n<=64) << n << nbits;
        offset = n;
        for (int64 i=0; i<n; i++)
            offset |= ((i+1)*nbits)<<(size_nbits+i*offset_nbits);
    }
    inline void set_data(int i, int64 v) {
        int pre_offset = i ? get_offset(i-1) : 0;
        int next_offset = get_offset(i);
        if (next_offset==0) next_offset=64;
        int nbits = next_offset - pre_offset;
        set_data(v, nbits, pre_offset);
    }

    inline void set_data_check_overflow(int i, int64 v) {
        int pre_offset = i ? get_offset(i-1) : 0;
        int next_offset = get_offset(i);
        if (next_offset==0) next_offset=64;
        int nbits = next_offset - pre_offset;
        ASSERT(nbits>=get_nbits(v));
        set_data(v, nbits, pre_offset);
    }

    inline vector<int64> to_vector() const {
        vector<int64> v(size());
        for (int i=0; i<size(); i++)
            v[i] = at(i);
        return v;
    }
};


// @pyjt(NanoVector.__add__)
inline NanoVector add(NanoVector self, NanoVector other) {
    for (int i=0; i<other.size(); i++)
        self.push_back_check_overflow(other[i]);
    return self;
}

// @pyjt(NanoVector.__iadd__)
// @attrs(return_self)
inline NanoVector* iadd(NanoVector* self, NanoVector other) {
    for (int i=0; i<other.size(); i++)
        self->push_back_check_overflow(other[i]);
    return self;
}

inline std::ostream& operator<<(std::ostream& os, const NanoVector& v) {
    os << '[';
    for (int i=0; i<v.size(); i++)
        os << v[i] << ',';
    return os << ']';
}


// @pyjt(NanoVector.__eq__)
inline bool eq(const NanoVector& a, const NanoVector& b) {
    return a.data == b.data && a.offset == b.offset;
}

// @pyjt(NanoVector.__ne__)
inline bool ne(const NanoVector& a, const NanoVector& b) {
    return a.data != b.data || a.offset != b.offset;
}

inline bool operator==(const NanoVector& a, const NanoVector& b) {
    return eq(a, b);
}
inline bool operator!=(const NanoVector& a, const NanoVector& b) {
    return ne(a, b);
}

inline bool operator<(const NanoVector& a, const NanoVector& b) {
    return a.data < b.data || (a.data == b.data && a.offset < b.offset);
}

} // jittor


namespace std {
template<> struct hash<jittor::NanoVector> {
inline std::size_t operator()(jittor::NanoVector const& s) const noexcept {
        std::size_t h1 = std::hash<jittor::int64>{}(s.data);
        std::size_t h2 = std::hash<jittor::int64>{}(s.offset);
        return h1 ^ (h2 << 1);
}
};
}
 
