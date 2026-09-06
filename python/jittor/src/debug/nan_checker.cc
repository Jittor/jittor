// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <cfloat>
#include <cmath>
#include <fstream>
#include "debug/nan_checker.h"
#include "ops/op_register.h"
#include "runtime/backend.h"
#include "mem/allocator.h"
#include "op.h"

namespace jittor {

void dump_var(Var* v, string name) {
    std::stringstream ss;
    ss << name << v->id << v->dtype() << v->shape << ".bin";
    name = ss.str();
    LOGe << "dump" << v << "to" << name;
    vector<char> buffer(v->size);
    backend_copy(buffer.data(), {BackendId::Cpu, 0}, v->mem_ptr,
                 allocation_device(v->allocator), v->size, true);
    std::fstream file(name, std::ios::out | std::ios::binary);
    CHECK(file.is_open()) << "Cannot open tensor dump" << name;
    file.write(buffer.data(), v->size);
    CHECK(file.good()) << "Cannot write tensor dump" << name;
}

bool check_nan(Var* v, Op* op) {
    if (!v->dtype().is_float() || v->num == 0) return true;
    if (v->input() && (
            v->input()->is_op(op_ids::empty()) ||
            v->input()->is_op(op_ids::setitem())))
        return true;
    if (v->allocator->is_cuda()) {
        const auto& backend = backend_ops(allocation_device(v->allocator).backend);
        CHECK(backend.check_nan) << "NaN checking is not supported by backend" << backend.name;
        backend.check_nan(v, op);
    } else {
        if (v->dtype() == ns_float32) {
            auto* __restrict__ ptr = v->ptr<float32>();
            auto num = v->num;
            bool ok = true;
            int64 i=0;
            for (; i<num; i++) {
                if (std::isinf(ptr[i]) || std::isnan(ptr[i])) {
                    ok = false;
                    break;
                }
            }
            ASSERT(ok) << "detect nan at index" << i << v;
        }
        if (v->dtype() == ns_float64) {
            auto* __restrict__ ptr = v->ptr<float64>();
            auto num = v->num;
            bool ok = true;
            int64 i=0;
            for (; i<num; i++) {
                if (std::isinf(ptr[i]) || std::isnan(ptr[i])) {
                    ok = false;
                    break;
                }
            }
            ASSERT(ok) << "detect nan at index" << i << v;
        }
    }
    return true;
}

}
