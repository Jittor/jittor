// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: Dun Liang <randonlang@gmail.com>. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <sstream>
#include "opt/pass/pass.h"
#include "opt/pass_manager.h"

namespace jittor {

Pass::Pass(const string& name): name(name) {}
Pass::~Pass() {}

void Pass::init(PassManager* pm) {
    this->pm = pm;
    op = pm->oc->op;
    all = &pm->all;
    ir = pm->main_ir;
}

} // jittor