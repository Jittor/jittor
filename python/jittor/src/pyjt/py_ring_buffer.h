// ***************************************************************
// Copyright (c) 2022 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include <Python.h>
#include "misc/ring_buffer.h"

namespace jittor {

// @pyjt(RingBuffer)
struct PyMultiprocessRingBuffer {
    RingBuffer* rb;
    uint64 buffer;
    bool _keep_numpy_array = false;
    bool init;
    // @pyjt(__init__)
    PyMultiprocessRingBuffer(uint64 size, uint64 buffer=0, bool init=true);
    // @pyjt(__dealloc__)
    ~PyMultiprocessRingBuffer();
    // @pyjt(push,send)
    void push(PyObject* obj);
    // @pyjt(pop,recv)
    PyObject* pop();
    // @pyjt(clear)
    inline void clear() { rb->clear(); }
    // @pyjt(keep_numpy_array)
    inline void keep_numpy_array(bool keep) { _keep_numpy_array = keep; }
    // @pyjt(stop)
    inline void stop() { rb->stop(); }
    // @pyjt(is_stop)
    inline bool is_stop() { return rb->is_stop; }

    // @pyjt(total_pop)
    inline uint64 total_pop() { return rb->l; }
    // @pyjt(total_push)
    inline uint64 total_push() { return rb->r; }
    // @pyjt(__repr__)
    inline string to_string() {
        string s="Buffer(free=";
        auto size = rb->size;
        auto used = rb->r - rb->l;
        s += S(100 - used*100.0/size);
        s += "% size=";
        s += S(size);
        s += ")";
        return s;
    }

    // @pyjt(__get__size)
    inline uint64 size() { return rb->size; }
};


}
