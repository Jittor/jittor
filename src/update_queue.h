// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "common.h"

namespace jittor {

struct UpdateQueue {
    struct Item {
        list<list<Item>>::iterator owner;
        Var* v;
    };
    list<list<Item>> queue;
    unordered_map<Var*, list<Item>::iterator> map;

    void push(Var* v, Var* prev);
    void pop(Var* v);
    void auto_flush();
};

extern UpdateQueue update_queue;

} // jittor

