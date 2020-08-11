// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
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

