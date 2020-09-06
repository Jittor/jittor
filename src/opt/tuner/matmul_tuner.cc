// ***************************************************************
// Copyright (c) 2020 Jittor. Authors: 
//     Dun Liang <randonlang@gmail.com>. 
//     Guoye Yang <498731903@qq.com>
// All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include "common.h"
#include "var.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "opt/tuner/matmul_tuner.h"
#include "opt/pass_manager.h"
#include "ops/op_register.h"

namespace jittor {

void MatmulTuner::run(PassManager* pm, TunerManager* tm) {
    FusedOp* fop=tm->oc->op;
    for (Op* op : fop->ops) {
        if (op->name_ex()!="reduce.add") continue;
        auto rop = (ReduceOp*)op;
        if (!(rop->x->input() && rop->x->input()->name_ex()=="binary.multiply" && fop->has(rop->x->input())))
            continue;
        auto bop = (BinaryOp*)(rop->x->input());
        if (!(bop->x->input() && bop->x->input()->name_ex()=="broadcast_to" && fop->has(bop->x->input())))
            continue;
        if (!(bop->y->input() && bop->y->input()->name_ex()=="broadcast_to" && fop->has(bop->y->input())))
            continue;
        auto bcop1 = (BroadcastToOp*)(bop->x->input());
        auto bcop2 = (BroadcastToOp*)(bop->y->input());
        if (bcop1->shape.size() != 3) continue;
        if (bcop1->x->shape.size() != 2) continue;
        if (bcop2->x->shape.size() != 2) continue;
        Var* xx = bcop1->x;
        Var* yy = bcop2->x;
        bool is_matmul = false, t1 = false, t2 = false;
        // xx : n m
        // yy :   m k
        // out: (n,k)
        if ((rop->reduce_mask == (1u<<1)) && (bcop1->bcast_mask == (1u<<2)) && (bcop2->bcast_mask == (1u<<0))) {
            is_matmul = true;
            t1 = false;
            t2 = false;
        }
        if ((rop->reduce_mask == (1u<<1)) && (bcop1->bcast_mask == (1u<<0)) && (bcop2->bcast_mask == (1u<<2))) {
            is_matmul = true;
            t1 = false;
            t2 = false;
            std::swap(xx, yy);
        }
        // xx : m n
        // yy : m   k
        // out: (n,k)
        if ((rop->reduce_mask == (1u<<0)) && (bcop1->bcast_mask == (1u<<2)) && (bcop2->bcast_mask == (1u<<1))) {
            is_matmul = true;
            t1 = true;
            t2 = false;
        }
        if ((rop->reduce_mask == (1u<<0)) && (bcop1->bcast_mask == (1u<<1)) && (bcop2->bcast_mask == (1u<<2))) {
            is_matmul = true;
            t1 = true;
            t2 = false;
            std::swap(xx, yy);
        }
        // xx : n   m
        // yy :   k m
        // out: (n,k)
        if ((rop->reduce_mask == (1u<<2)) && (bcop1->bcast_mask == (1u<<1)) && (bcop2->bcast_mask == (1u<<0))) {
            is_matmul = true;
            t1 = false;
            t2 = true;
        }
        if ((rop->reduce_mask == (1u<<2)) && (bcop1->bcast_mask == (1u<<0)) && (bcop2->bcast_mask == (1u<<1))) {
            is_matmul = true;
            t1 = false;
            t2 = true;
            std::swap(xx, yy);
        }
        if (!is_matmul) continue;
        // TODO: support int8 * int8
        if (!(xx->dtype().is_float() && yy->dtype().is_float())) continue;
        if (fop->flags.get(NodeFlags::_cpu))
            if (xx->dtype().dsize() != 4) continue;

        string relay_matmul_name = fop->flags.get(NodeFlags::_cpu) ?
            "mkl_matmul" : "cublas_matmul";
        if (!has_op(relay_matmul_name))
            return;
        auto make_matmul = get_op_info(relay_matmul_name)
            .get_constructor<VarPtr, Var*, Var*, bool, bool>();
        auto rvar = make_matmul(xx, yy, t1, t2);
        auto rid = fop->context->vrm.add_relay_group({{rvar, rop->y}});
        auto srid = "relay"+S(rid);
        add_candidate(srid, 1);
        add_candidate(srid, 0);
        confidence = 20;
    }
}

}