// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

struct Node;

/**
 * A dense index over the nodes of one traversal, kept beside the traversal.
 *
 * Every algorithm that walks the graph needs the same thing: "given this node,
 * what is its position in my array". That used to be `Node::custom_data`, one
 * int per node shared by the executor, the fuser, FusedOp, grad() and the
 * topological sorts, with no marker saying whose turn it was. Two of them
 * interleaving corrupted each other, and the defences were hand-written: the
 * memory profiler copied the whole field out and back around its own sort, and
 * grad() snapshotted every index it would need into a side buffer before it
 * started building ops, because building an op can re-enter run_sync.
 *
 * This is open addressing with linear probing over a fixed capacity, so:
 *   - a reference handed out by operator[] stays valid for the life of the
 *     table (there is no rehash to move it), which several call sites rely on
 *     to do `--index[node]`;
 *   - one allocation per traversal, reusable across traversals via reset();
 *   - a probe is a multiply and a masked array read, which is the same order
 *     of cost as the pointer dereference it replaces.
 *
 * `reset(n)` must be given an upper bound on the number of distinct nodes.
 * Exceeding it is a hard error rather than a silent rehash: the bound always
 * comes from a container the caller already has, so being wrong about it means
 * being wrong about the traversal.
 */
struct NodeIndex {
    struct Slot {
        Node* key;
        // Packed into one word beside the key so a slot stays 16 bytes: a
        // probe is one cache line, and the table for a few thousand nodes
        // stays inside L2.
        int32 value;
        uint32 gen;
    };
    vector<Slot> slots;
    size_t mask = 0;
    size_t limit = 0;
    size_t count = 0;
    // A slot belongs to the current traversal only when its generation matches.
    // That is what makes reset() O(1): clearing a table that is reused for
    // every fused op of a run would otherwise memset it thousands of times.
    uint32 generation = 0;

    inline void reset(size_t n) {
        size_t capacity = 8;
        // Load factor stays under 3/4; linear probing is fine there and the
        // table is half the size a 1/2 factor would give, which is what keeps
        // it in cache.
        while (capacity * 3 < (n+1) * 4) capacity <<= 1;
        if (slots.size() < capacity || generation == 0xffffffffu) {
            slots.assign(capacity, Slot{nullptr, 0, 0});
            generation = 0;
        }
        // Reusing a larger table than asked for is fine and saves the realloc,
        // so the mask always follows the table that is actually there.
        mask = slots.size() - 1;
        limit = (slots.size() / 4) * 3;
        generation++;
        count = 0;
    }

    inline size_t probe(Node* node) const {
        // Fibonacci hashing on the pointer; the low bits of a heap pointer are
        // alignment zeros, so they are shifted out first.
        size_t h = ((size_t)node) >> 4;
        h *= (size_t)0x9E3779B97F4A7C15ull;
        return h & mask;
    }

    /// Reference to this node's slot, created with value 0 if new.
    inline int32& operator[](Node* node) {
        size_t i = probe(node);
        while (slots[i].gen == generation && slots[i].key != node) i = (i+1) & mask;
        if (slots[i].gen != generation) {
            ASSERT(count < limit) << "NodeIndex overflow: reset() was given too small a bound";
            slots[i].key = node;
            slots[i].value = 0;
            slots[i].gen = generation;
            count++;
        }
        return slots[i].value;
    }

    /// This node's value, or `_default` when it was never indexed.
    inline int get(Node* node, int _default=-1) const {
        size_t i = probe(node);
        while (slots[i].gen == generation) {
            if (slots[i].key == node) return slots[i].value;
            i = (i+1) & mask;
        }
        return _default;
    }

    inline bool has(Node* node) const {
        size_t i = probe(node);
        while (slots[i].gen == generation) {
            if (slots[i].key == node) return true;
            i = (i+1) & mask;
        }
        return false;
    }
};

} // jittor
