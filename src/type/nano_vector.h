// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "type/intrin.h"
#include <atomic>

namespace jittor {

struct Slice {
    int64 start, stop, step, mask;
    inline void fill(int64 size) {
        if (step>0) {
            if (mask&1) start = 0;
            else if (start<0) start = std::max((int64)0, start+size);
            else start = std::min(start, size);
            if (mask&2) stop = size;
            else if (stop<0) stop = std::max((int64)0, stop+size);
            else stop = std::min(stop, size);
        } else {
            if (mask&1) start = size-1;
            else if (start<0) start = std::max((int64)-1, start+size);
            else start = std::min(start, size-1);
            if (mask&2)
                stop = -1;
            else if (stop<0)
                stop = std::max((int64)-1, stop+size);
            else stop = std::min(stop, size-1);
        }
        mask = 0;
        ASSERT(step>0 ? (start>=0 && start<=size && stop>=0 && stop<=size)
                      : (start>=-1 && start<size && stop>=-1 && stop<size))
            << "slice overflow:" << start << stop << step << "size:" << size;
    }
};

// return a / ceil_to_2_pow(b)
inline uint64 fast_div(uint64 a, uint64 b) {
    return a >> (64 - lzcnt(b));
}

// Out-of-line storage of a NanoVector whose values do not fit the packed form.
// Shared between copies and copied before a write, so a spilled vector is
// still cheap to pass by value.
struct NanoVectorHeap {
    std::atomic<int> refcount;
    int size, capacity;
    int64 values[1];

    static inline NanoVectorHeap* create(int capacity) {
        auto bytes = sizeof(NanoVectorHeap) + sizeof(int64)*(capacity-1);
        auto* heap = (NanoVectorHeap*)::operator new(bytes);
        new (&heap->refcount) std::atomic<int>(1);
        heap->size = 0;
        heap->capacity = capacity;
        return heap;
    }
    static inline void release(NanoVectorHeap* heap) {
        if (heap->refcount.fetch_sub(1, std::memory_order_acq_rel) == 1)
            ::operator delete(heap);
    }
};

// A short integer vector -- shapes, strides, axes -- passed by value through
// the whole core and used as a JIT cache key.
//
// The common case is packed into two words: `data` holds every value at the
// smallest two's-complement width that represents it, and `offset` holds the
// entry count (4 bits) and the end bit of each entry (6 bits each). That form
// holds at most ten entries and 64 value bits. A vector beyond either bound
// spills: `offset`'s count field is set to `spilled_tag` and `data` points to
// a reference-counted NanoVectorHeap. Spilling only ever happens when the
// packed form cannot hold the values, and a spilled vector that shrinks back
// within the bounds is packed again, so for vectors built by appending each
// content has exactly one representation and the raw-word comparison of two
// packed vectors stays an exact equality test.
//
// `max_size` bounds the entry count. Tensor rank has its own, lower bound
// (`max_tensor_rank`, checked by Var::set_shape) because per-axis masks
// throughout the core are 32-bit.
// @pyjt(NanoVector)
struct NanoVector {
    int64 data=0, offset=0;

    enum {
        size_nbits=4,
        offset_nbits=6,
        packed_max_size=10,
        spilled_tag=(1<<size_nbits)-1,
        max_size=64,
        max_tensor_rank=32,
    };

    // @pyjt(__init__)
    inline NanoVector() {}
    inline NanoVector(std::nullptr_t) {}
    // @pyjt(__init__)
    inline NanoVector(const NanoVector& nv) : data(nv.data), offset(nv.offset) {
        if (is_spilled()) heap()->refcount.fetch_add(1, std::memory_order_relaxed);
    }
    inline NanoVector(NanoVector&& nv) noexcept : data(nv.data), offset(nv.offset) {
        nv.data = nv.offset = 0;
    }
    inline NanoVector& operator=(const NanoVector& nv) {
        if (this != &nv) {
            NanoVector copy(nv);
            swap(copy);
        }
        return *this;
    }
    inline NanoVector& operator=(NanoVector&& nv) noexcept {
        if (this != &nv) {
            release();
            data = nv.data, offset = nv.offset;
            nv.data = nv.offset = 0;
        }
        return *this;
    }
    // @pyjt(__dealloc__)
    inline ~NanoVector() { release(); }

    inline void swap(NanoVector& nv) noexcept {
        std::swap(data, nv.data);
        std::swap(offset, nv.offset);
    }

    inline bool is_spilled() const {
        return (offset & ((1<<size_nbits)-1)) == spilled_tag;
    }
    inline NanoVectorHeap* heap() const { return (NanoVectorHeap*)data; }

    inline void release() {
        if (is_spilled()) NanoVectorHeap::release(heap());
        data = offset = 0;
    }

    inline void clear() { release(); }

    // @pyjt(__len__, __map_len__)
    inline int size() const {
        int s = offset & ((1<<size_nbits)-1);
        return s == spilled_tag ? heap()->size : s;
    }

    // @pyjt(numel)
    inline int64 numel() const {
        int64 value = 1;
        for (int i=0; i<size(); i++)
            value *= at(i);
        return value;
    }

    static inline int get_nbits(int64 v) {
        // return 1bit if v in [-1,0]
        // return 2bit if v in [-2,1,0,1]
        if (v<0) v = ~v;
        return 65 - lzcnt(v);
    }

    // Whether a value list can be held at all.
    static inline bool fits(const vector<int64>& v) {
        return (int)v.size() <= max_size;
    }

    // Packed-form internals; meaningless for a spilled vector.
    // An entry ends at bit 1..64, and 64 does not fit the 6-bit field: it is
    // stored as 0, which no existing entry can otherwise end at.
    inline int get_offset(int i) const {
        int o = (offset >> (size_nbits+i*offset_nbits)) 
            & ((1<<offset_nbits)-1);
        return o ? o : 64;
    }

    inline int total_bits() const {
        int s = size();
        return s && !is_spilled() ? get_offset(s-1) : 0;
    }

    inline void set_offset(int i, int next_offset) {
        offset |= ((uint64)(next_offset & ((1<<offset_nbits)-1)))
            << (i*offset_nbits+size_nbits);
    }

    inline void set_data(int64 v, int nbits, int pre_offset) {
        // don't have to clean bit if init
        // data &= ~(((1<<nbits)-1) << pre_offset);
        data |= (v & ((1ll<<nbits)-1)) << pre_offset;
    }

    // Make the heap exclusively owned with room for `capacity` entries,
    // converting a packed vector first. Used before every write to a spilled
    // vector and by the push that overflows the packed form.
    inline NanoVectorHeap* mutable_heap(int capacity) {
        int s = size();
        if (capacity < s) capacity = s;
        if (is_spilled() && heap()->refcount.load(std::memory_order_acquire) == 1
            && heap()->capacity >= capacity)
            return heap();
        int new_capacity = std::max(capacity, std::max(16, s*2));
        auto* h = NanoVectorHeap::create(new_capacity);
        for (int i=0; i<s; i++) h->values[i] = at(i);
        h->size = s;
        release();
        data = (int64)h;
        offset = spilled_tag;
        return h;
    }

    // A spilled vector that fits the packed form again is stored packed, so
    // that appending always yields the same representation for the same values.
    inline void repack_if_fits() {
        if (!is_spilled()) return;
        auto* h = heap();
        if (h->size > packed_max_size) return;
        int bits = 0;
        for (int i=0; i<h->size; i++) bits += get_nbits(h->values[i]);
        if (bits > 64) return;
        NanoVector packed;
        for (int i=0; i<h->size; i++) packed.push_back(h->values[i]);
        swap(packed);
    }

    inline void push_back(int64 v) {
        if (!is_spilled()) {
            auto s = size();
            auto nbits = get_nbits(v);
            int pre_offset = s ? get_offset(s-1) : 0;
            int next_offset = pre_offset+nbits;
            if (s < packed_max_size && next_offset <= 64) {
                offset ++;
                set_offset(s, next_offset);
                set_data(v, nbits, pre_offset);
                return;
            }
        }
        auto* h = mutable_heap(size()+1);
        h->values[h->size++] = v;
    }

    // @pyjt(append)
    inline void push_back_check_overflow(int64 v) {
        auto s = size();
        USER_CHECK(s < max_size) << "NanoVector supports at most" << (int)max_size
            << "entries, got one more than" << s << ": value" << v;
        push_back(v);
    }

    // @pyjt(__getitem__, __map_getitem__)
    inline int64 user_at(int i) const {
        USER_CHECK(i >= -size() && i < size()) << "NanoVector index out of range" << i;
        return at(i);
    }

    inline int64 at(int i) const {
        if (i<0) i+= size();
        ASSERT(i>=0 && i<size());
        return operator[](i);
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
        if (PREDICT_BRANCH_NOT_TAKEN(is_spilled())) return heap()->values[i];
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

#ifdef __linux__
    inline NanoVector(const vector<int64_t>& v) {
        for (auto a : v) push_back_check_overflow((int64)a);
    }
#endif

    template<typename TMakeV>
    inline static NanoVector make(const TMakeV* v, int n) {
        NanoVector nv;
        for (int i=0; i<n; i++) nv.push_back_check_overflow(v[i]);
        return nv;
    }

    // @pyjt(__init__)
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

    inline void pop_back() {
        if (is_spilled()) {
            mutable_heap(size())->size--;
            repack_if_fits();
            return;
        }
        offset--;
        offset &= ~(((1ll<<offset_nbits)-1) << (size()*offset_nbits+size_nbits));
        data &= total_bits() == 64 ? -1ll : (1ll<<total_bits())-1;
    }
    inline void push_back(Iter s, Iter t) {
        while (s != t) push_back(*s), ++s;
    }
    // Size the vector to `n` zero entries, each able to hold any value with
    // magnitude up to `max_value`; fill them with `set_data(i, v)`.
    inline void reserve(int64 max_value, int n) {
        release();
        auto nbits = get_nbits(max_value);
        ASSERT(n>=0 && n<=max_size) << n;
        if (n<=packed_max_size && nbits*n<=64) {
            offset = n;
            for (int i=0; i<n; i++)
                set_offset(i, (i+1)*nbits);
            return;
        }
        auto* h = mutable_heap(n);
        for (int i=0; i<n; i++) h->values[i] = 0;
        h->size = n;
    }
    inline void set_data(int i, int64 v) {
        if (is_spilled()) {
            mutable_heap(size())->values[i] = v;
            return;
        }
        int pre_offset = i ? get_offset(i-1) : 0;
        int next_offset = get_offset(i);
        int nbits = next_offset - pre_offset;
        set_data(v, nbits, pre_offset);
    }

    inline void set_data_check_overflow(int i, int64 v) {
        if (is_spilled()) {
            set_data(i, v);
            return;
        }
        int pre_offset = i ? get_offset(i-1) : 0;
        int next_offset = get_offset(i);
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

    inline void _unpack(int i) {
        return;
    }

    template<class... Args>
    void _unpack(int i, int& x, Args&&... args) {
       x = this->operator[](i);
       _unpack(i+1, std::forward<Args>(args)...);
    }

    template<class... Args>
    void unpack(Args&&... args) {
        _unpack(0, std::forward<Args>(args)...);
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

inline bool nano_vector_content_eq(const NanoVector& a, const NanoVector& b) {
    if (a.size() != b.size()) return false;
    for (int i=0; i<a.size(); i++)
        if (a[i] != b[i]) return false;
    return true;
}

// @pyjt(NanoVector.__eq__)
inline bool eq(const NanoVector& a, const NanoVector& b) {
    if (PREDICT_BRANCH_NOT_TAKEN(a.is_spilled() || b.is_spilled()))
        return nano_vector_content_eq(a, b);
    return a.data == b.data && a.offset == b.offset;
}

// @pyjt(NanoVector.__ne__)
inline bool ne(const NanoVector& a, const NanoVector& b) {
    return !eq(a, b);
}

#ifdef Py_PYTHON_H
// @pyjt(NanoVector.__hash__)
inline int64 py_hash(const NanoVector& value) {
    // Delegate to the running interpreter so NanoVector and an equal tuple have
    // exactly the same hash on every supported Python version.
    PyObject* tuple = PyTuple_New(value.size());
    if (!tuple) return -1;
    for (int i=0; i<value.size(); ++i) {
        PyObject* item = PyLong_FromLongLong(value[i]);
        if (!item) {
            Py_DECREF(tuple);
            return -1;
        }
        PyTuple_SET_ITEM(tuple, i, item);
    }
    Py_hash_t result = PyObject_Hash(tuple);
    Py_DECREF(tuple);
    return (int64)result;
}
#endif

inline bool operator==(const NanoVector& a, const NanoVector& b) {
    return eq(a, b);
}
inline bool operator!=(const NanoVector& a, const NanoVector& b) {
    return ne(a, b);
}

// Packed vectors order by their raw words; a spilled vector orders after every
// packed one and spilled vectors order by (size, values).
inline bool operator<(const NanoVector& a, const NanoVector& b) {
    bool as = a.is_spilled(), bs = b.is_spilled();
    if (!as && !bs)
        return a.data < b.data || (a.data == b.data && a.offset < b.offset);
    if (as != bs) return bs;
    if (a.size() != b.size()) return a.size() < b.size();
    for (int i=0; i<a.size(); i++)
        if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

} // jittor


namespace std {
template<> struct hash<jittor::NanoVector> {
inline std::size_t operator()(jittor::NanoVector const& s) const noexcept {
        if (s.is_spilled()) {
            std::size_t h = s.size();
            for (int i=0; i<s.size(); i++)
                h = h * 1000003u ^ std::hash<jittor::int64>{}(s[i]);
            return h;
        }
        std::size_t h1 = std::hash<jittor::int64>{}(s.data);
        std::size_t h2 = std::hash<jittor::int64>{}(s.offset);
        return h1 ^ (h2 << 1);
}
};
}
 
