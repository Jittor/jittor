// ***************************************************************
// Copyright (c) 2021 Jittor. All Rights Reserved. 
// Maintainers: Dun Liang <randonlang@gmail.com>. 
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "opt/tuner/tuner.h"

namespace jittor {

Tuner::Tuner(const string& name) : name(name), confidence(0), candidates({}) {};
Tuner::~Tuner() {}

void Tuner::add_candidate(const string& key, int value) {
    candidates[key].push_back(value);
}

}