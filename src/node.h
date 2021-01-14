// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"
#include "misc/nano_string.h"
#include "misc/nano_vector.h"
#include "pybind/py_var_tracer.h"

namespace jittor {

extern unordered_map<void*, int64> lived_nodes;
extern int64 total_node;
extern int64 nt;
extern vector<Node*> free_buffer;

struct NodeFlags {
    typedef uint16 nf_t;
    nf_t flags=0;
    enum Flags {
        // bit0: is_var
        _var=0,
        // bit1: state
        _finished=1,
        // bit2: stop grad
        _stop_grad=2,
        // bit3: is fetch
        _fetch=3,
        _n=4,

        // var related flags
        _force_fuse=_n+0,
        _stop_fuse=_n+1,
        _in_update_queue=_n+2,

        // op related flags
        // bit0: support cpu
        _cpu=_n+0,
        // bit1: support cuda
        _cuda=_n+1,
        // bit2: forward op
        _forwarded=_n+2,
        // bit3: vary shape op
        _vary_shape=_n+3,
        // bit4~5: op type
        _op_type=_n+4, _op_type_nbits=2,
        // bit6: backprop grad at ones
        _grads=_n+6,
        // bit7: has graph optimize
        _has_gopt=_n+7,
        // bit7: has vary input
        _has_vary_input=_n+8,
    };

    inline void set(Flags f, int a=1, int nbits=1) {
        nf_t mask = (((1u<<nbits)-1)<<f);
        flags = (flags & ~mask) | ((a<<f)&mask);
    }

    inline nf_t get(Flags f, int nbits=1) const {
        return (flags>>f) & ((1u<<nbits)-1);
    }
};

struct Node {
    struct input_t;
    struct output_t;
    struct var_output_t {
        Op* op;
        int index;
    };
    struct input_t {
        Node* node;
        list<output_t>::iterator back;
        input_t(Node* n) : node(n) {}
        operator Node*() { return node; }
        operator Op*() { return (Op*)node; }
        operator Var*() { return (Var*)node; }
    };
    struct output_t {
        Node* node;
        int index;
        list<input_t>::iterator back;
        output_t(Node* n, int i) : node(n), index(i) {}
        operator Node*() { return node; }
        operator Op*() { return (Op*)node; }
        operator Var*() { return (Var*)node; }
        operator var_output_t() { return {(Op*)node, index}; }
    };
    static int64_t tflag_count;
    NodeFlags flags;
    NanoString ns;
    inline bool is_var() const { return flags.get(NodeFlags::_var); }
    inline bool is_stop_grad() const { return flags.get(NodeFlags::_stop_grad); }
    inline bool is_finished() const { return flags.get(NodeFlags::_finished); }
    // forward_liveness can propergate forward(from input to output)
    // f1. var_holder contrib one forward_liveness
    // f2. var ptr contrib one forward_liveness
    // f3. input(has_grad and f>0) contrib one forward_liveness
    int forward_liveness = 0;
    // forward_liveness can propergate backward(from output to input)
    // b1. var ptr contrib one backward_liveness
    // b2. var holder contrib one backward_liveness
    // b3. output(b>0) contrib one backward_liveness
    int backward_liveness = 0;
    // pending liveness can propergate backward(from output to input)
    // p1: pending and f>0 and b>0 contrib pending_liveness
    // p2: output(p>0 and pending) contrib pending_liveness
    int pending_liveness = 0;
    inline bool need_free()
    { return !pending_liveness && (!forward_liveness || !backward_liveness); }
    
    int64_t tflag = 0;
    int64_t custom_data;
    list<input_t> _inputs;
    list<output_t> _outputs;

#ifdef NODE_MEMCHECK
    inline Node() {
        lived_nodes[(void*)this] = ++total_node;
    }

    inline virtual ~Node() {
        lived_nodes.erase((void*)this);
        if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var)) trace_data.release_node(this);
    }
#else
    inline Node() {};
    inline virtual ~Node() { if (PREDICT_BRANCH_NOT_TAKEN(trace_py_var)) trace_data.release_node(this);};
#endif
    inline Var* var() { return (Var*)this; }
    inline Op* op() { return (Op*)this; }
    inline Node* node() { return this; }
    void free();
    // this function is used for debug memory
    inline bool exist() const {
    #ifdef NODE_MEMCHECK
        return lived_nodes.count((void*)this);
    #else
        return true;
    #endif
    }
    void memcheck_all_exist() const;
    inline int64 __id() const {
    #ifdef NODE_MEMCHECK
        return lived_nodes.at((void*)this);
    #else
        return 0;
    #endif
    }
    // release from counter and memory checker
    void __release();
    #define CHECK_NODE_EXIST(node) \
        ASSERT(node->exist()) << "Node("#node")" << (void*)node << "not exist."
    #define CHECK_EXIST CHECK_NODE_EXIST(this)
    #define CHECK_NODE_EXIST2(a,b) \
        CHECK_NODE_EXIST(a); CHECK_NODE_EXIST(b);
    #define CHECK_NODE_EXIST3(a,b,c) \
        CHECK_NODE_EXIST2(a,b); CHECK_NODE_EXIST(c);

    inline Caster<Node*, Node::input_t> inputs() { CHECK_EXIST; return &_inputs; }
    inline Caster<Node*, Node::output_t> outputs() { CHECK_EXIST; return &_outputs; }
    inline Node* input(uint i) {
        CHECK_EXIST;
        auto iter = _inputs.begin();
        while (i--) iter++;
        return iter->node;
    }
    inline Node* output(uint i) {
        CHECK_EXIST;
        auto iter = _outputs.begin();
        while (i--) iter++;
        return iter->node;
    }
    
    void release_inputs();
    void set_inputs(list<Node*> nodes);
    void add_inputs(const vector<Node*>& nodes);
    void add_inputs(const vector<Var*>& nodes);
    void release_forward_liveness();
    void own_forward_liveness();
    void release_backward_liveness();
    void own_backward_liveness();
    void release_pending_liveness();
    void own_pending_liveness();
    void release_both_liveness();
    void own_both_liveness();
    void finish_pending_liveness();
    void set_stop_grad();
};

struct SetupFreeBuffer {

bool outside;
inline SetupFreeBuffer() {
    outside = !nt;
    if (outside) {
        nt = ++Node::tflag_count;
    }
}

inline ~SetupFreeBuffer() {
    if (outside) {
        for (int i=0; i<free_buffer.size(); i++)
            delete free_buffer[i];
        free_buffer.clear();
        nt = 0;
    }
}

};

std::ostream& operator<<(std::ostream& os, const Node* node);

} // jittor