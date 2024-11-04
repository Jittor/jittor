// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "common.h"
#include "op.h"
#include "acl_jittor.h"
#include "ops/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/transpose_op.h"
#include "ops/array_op.h"
#include "ops/code_op.h"
#include "fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "executor.h"
#include "misc/cuda_flags.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "acl_op.h"
namespace jittor
{
    void free_var_mem(Var *v);

    unordered_map<uint32, string> opname_map = {
        // unary op
        {ns_cast, "Cast"},
        {ns_negative, "Neg"},
        {ns_abs, "Abs"},
        {ns_exp, "Exp"},
        {ns_log, "Log"},
        {ns_sqrt, "Sqrt"},
        {ns_ceil, "Ceil"},
        {ns_floor, "Floor"},
        {ns_round, "Round"},
        // m(round_int)
        // m(floor_int)
        // m(ceil_int)
        {ns_sin, "Sin"},
        {ns_cos, "Cos"},
        {ns_tan, "Tan"},
        {ns_asin, "Asin"},
        {ns_acos, "Acos"},
        {ns_atan, "Atan"},
        {ns_sinh, "Sinh"},
        {ns_cosh, "Cosh"},
        {ns_tanh, "Tanh"},
        {ns_asinh, "Asinh"},
        {ns_acosh, "Acosh"},
        {ns_atanh, "Atanh"},
        {ns_sigmoid, "Sigmoid"},
        {ns_erf, "Erf"},
        {ns_erfinv, "Erfinv"},
        {ns_logical_not, "LogicalNot"},
        {ns_bitwise_not, "BitwiseNot"},
        // binary op
        {ns_pow, "Pow"},
        {ns_maximum, "Maximum"},
        {ns_minimum, "Minimum"},
        {ns_add, "Add"},
        {ns_subtract, "Sub"},
        {ns_multiply, "Mul"},
        {ns_divide, "RealDiv"},
        {ns_floor_divide, "FloorDiv"},
        {ns_mod, "Mod"},
        {ns_less, "Less"},
        {ns_less_equal, "LessEqual"},
        {ns_greater, "Greater"},
        {ns_greater_equal, "GreaterEqual"},
        {ns_equal, "Equal"},
        {ns_not_equal, "NotEqual"},
        {ns_left_shift, "LeftShift"},
        {ns_right_shift, "RightShift"},
        {ns_logical_and, "LogicalAnd"},
        {ns_logical_or, "LogicalOr"},
        {ns_logical_xor, "LogicalXor"},
        {ns_bitwise_and, "BitwiseAnd"},
        {ns_bitwise_or, "BitwiseOr"},
        {ns_bitwise_xor, "BitwiseXor"},

    };

    void fallback_cpu(Op *op)
    {
        LOGy << "!!! fallback_cpu " << op;
        use_cuda = 0;
        for (auto v : op->inputs())
        {
            if (v->mem_ptr && v->allocator->is_cuda())
            {
                migrate_to_cpu(v, exe.allocator);
            }
        }
        for (auto v : op->outputs())
        {
            if (v->mem_ptr && v->allocator->is_cuda())
            {
                migrate_to_cpu(v, exe.allocator);
            }
        }
        op->flags.set(NodeFlags::_cpu);
        op->flags.set(NodeFlags::_cuda, 0);
        if (op->name() == string("fused"))
        {
            auto fop = (FusedOp *)op;
            for (auto op : fop->ops)
            {
                op->flags.set(NodeFlags::_cpu);
                op->flags.set(NodeFlags::_cuda, 0);
            }
        }
        op->do_run();
        use_cuda = 1;
    }

    /*
        check compile
        if compiled: exec
        else: compile
            check is fused
                check is relay
                else
                    compile func = try exec
                        if failed: fallback_cpu
            else
                try compile
                if failed: fallback_cpu
    */

    extern jit_op_entry_t (*do_compile_hook)(Op *);
    jit_op_entry_t do_compile_inner(Op *op);

    void try_exec_and_fallback_cpu(Op *op)
    {
        auto fop = (FusedOp *)op;

        std::set<Var *> new_alloced;
        map<Op *, int> op_indeg;
        map<Var *, int> var_outdeg;
        std::queue<Op *> queue;

        for (Op *op : fop->ops)
            op_indeg[op] = 0;

        map<Op *, vector<Op *>> out_map;
        map<Var *, vector<Op *>> from;

        int len = 0;
        for (Op *v : fop->ops)
        {
            for (auto in : v->inputs())
                from[in].push_back(v);
            ++len;
        }
        for (Op *u : fop->ops)
        {
            for (auto out : u->outputs())
            {
                if (from.find(out) != from.end())
                {
                    for (auto v : from[out])
                    {
                        ++op_indeg[v];
                        ++var_outdeg[out];
                        out_map[u].push_back(v);
                    }
                }
            }
        }
        for (Op *op : fop->ops)
        {
            if (op_indeg[op] == 0)
                queue.push(op);
        }

        int total = 0;
        int fallback = 0;
        try
        {
            while (!queue.empty())
            {
                total++;

                for (auto in : op->inputs())
                {
                    ASSERT(in->mem_ptr);
                }
                auto op = queue.front();
                queue.pop();
                for (auto out : op->outputs())
                {
                    if (out->mem_ptr)
                        continue;
                    out->alloc(exe.allocator);
                    new_alloced.insert(out);
                }
                for (auto out : out_map[op])
                {
                    --op_indeg[out];
                    if (op_indeg[out] == 0)
                        queue.push(out);
                }
                if (op->name() == string("unary"))
                {
                    auto uop = (UnaryOp *)op;
                    AclOpRunner op("...");
                    op.add(uop->x, true);
                    op.add(uop->y, false);
                    auto iter = opname_map.find(uop->ns);
                    ASSERT(iter != opname_map.end()) << "op " << uop->ns << " not found";
                    op.name = iter->second;
                    op.jt_name = uop->name();
                    op.run();
                }
                else if (op->name() == string("binary"))
                {
                    auto bop = (BinaryOp *)op;
                    AclOpRunner op("...");
                    op.add(bop->x, true);
                    op.add(bop->y, true);
                    op.add(bop->z, false);
                    auto iter = opname_map.find(bop->ns);
                    ASSERT(iter != opname_map.end()) << "op " << bop->ns << " not found";
                    op.name = iter->second;
                    op.jt_name = bop->name();

                    if (bop->x->dtype() == ns_bool and bop->y->dtype() == ns_bool)
                    {
                        // BitwiseOr, BitwiseAnd, BitwiseXor -> LogicalOr, LogicalAnd, LogicalXor
                        if (bop->ns == ns_bitwise_or)
                        {
                            op.name = "LogicalOr";
                        }
                        else if (bop->ns == ns_bitwise_and)
                        {
                            op.name = "LogicalAnd";
                        }
                        else if (bop->ns == ns_bitwise_xor)
                        {
                            op.name = "LogicalXor";
                        }
                    }
                    op.run();
                }
                else if (op->name() == string("ternary"))
                {
                    auto top = (TernaryOp *)op;
                    AclOpRunner op("Select");
                    op.add(top->cond, true);
                    op.add(top->x, true);
                    op.add(top->y, true);
                    op.add(top->z, false);
                    op.run();
                }
                else if (op->name() == string("array"))
                {
                    auto aop = (ArrayOp *)op;
                    aclrtMemcpy(aop->output->mem_ptr, aop->output->size, aop->ptr<void>(), aop->output->size, ACL_MEMCPY_HOST_TO_DEVICE);
                }
                else if (op->name() == string("reduce"))
                {
                    auto rop = (ReduceOp *)op;
                    AclOpRunner op("");
                    if (rop->ns == ns_add)
                        op.name = "ReduceSum";
                    else if (rop->ns == ns_multiply)
                        // TODO unsupported the multi dim

                        op.name = "ReduceProd";
                    else if (rop->ns == ns_maximum)
                        op.name = "ReduceMax";
                    else if (rop->ns == ns_minimum)
                        op.name = "ReduceMin";
                    else if (rop->ns == ns_mean)
                        op.name = "ReduceMean";
                    else
                        LOGf << "op " << rop->ns << " not supported";
                    op.jt_name = "reduce";
                    op.add(rop->x, true);

                    ReduceAttr *attr = new ReduceAttr();
                    for (int i = 0; i < rop->x->shape.size(); i++)
                        if (rop->reduce_mask & (1 << i))
                            attr->axes.push_back(i);
                    if (rop->x->shape.size() == rop->y->shape.size())
                        attr->keepdims = true;
                    else
                        attr->keepdims = false;

                    op.op_attr.reset(attr);
                    op.add(rop->y, false);
                    op.run();
                }
                else if (op->name() == string("broadcast_to"))
                {
                    auto bop = (BroadcastToOp *)op;
                    
                    if(bop->x->shape.size() == 1 && bop->x->shape[0] == 1)
                    {
                        aclrtSynchronizeStream(aclstream);
                    }
                    
                    AclOpRunner op("Expand");
                    op.jt_name = "expand";

                    NanoVector xshape, xshape_bk = bop->x->shape;
                    NanoVector zshape = bop->z->shape;
                    for (int i = 0; i < zshape.size(); i++)
                    {
                        if (bop->bcast_mask & (1 << i))
                        {
                            xshape.push_back(1);
                        }
                        else
                        {
                            xshape.push_back(zshape[i]);
                        }
                    }
                    bop->x->shape = xshape;
                    op.add(bop->x, true);
                    // bop->x->shape = xshape_bk;
                    op.add(bop->z, false);
                    op.run();
                    bop->x->shape = xshape_bk;
                }
                else if (op->name() == string("fuse_transpose"))
                {
                    // replace fuse_transpose with transpose
                    auto top = (TransposeOp *)op;
                    AclOpRunner op("Transpose");
                    op.add(top->x, true);
                    op.add(top->y, false);
                    op.jt_name = "transpose";

                    ReduceAttr *attr = new ReduceAttr();
                    for (int i = 0; i < top->axes.size(); i++)
                        attr->axes.push_back(top->axes[i]);
                    op.op_attr.reset(attr);

                    op.run();
                }
                else
                {
                    LOGf << "op " << op->name() << " not supported";
                }

                for (auto in : op->inputs())
                {
                    --var_outdeg[in];
                    if (var_outdeg[in] == 0)
                    {
                        if (new_alloced.find(in) != new_alloced.end())
                        {
                            free_var_mem(in);
                            new_alloced.erase(in);
                        }
                    }
                }
            }
        }
        catch (std::exception &e)
        {
            fallback = 1;
            LOGir << "fallback cpu" << e.what();
        }
        for (auto v : new_alloced)
        {
            free_var_mem(v);
        }
        if (fallback)
        {
            fallback_cpu(op);
        }
    }

    extern int current_seed;
    extern int64 current_offset;

    static unordered_map<string, std::function<void(Op *)>> acl_ops = {
        {"curand_random", [&current_seed, &current_offset](Op *op)
         {
             auto _op = (RandomOp *)op;
             AclOpRunner runner(_op->type == ns_uniform ? "RandomUniform" : "RandomNormal");
             auto out = op->output(0);
             RandomAttr *attr = new RandomAttr();
             attr->seed = current_seed;
             attr->offset = current_offset;
             runner.jt_name = "random";
             runner.op_attr.reset(attr);

             runner.add(out, false);
             runner.run();
             current_offset += out->numel();
         }},
    };

    static void exec_mapped_acl_ops(Op *op)
    {
        auto iter = acl_ops.find(op->name());
        if (iter != acl_ops.end())
        {
            LOGv << "exec acl op " << op->name() << op;
            iter->second(op);
        }
        else
        {
            LOGf << "op " << op->name() << " not supported";
        }
    }

    static jit_op_entry_t acl_do_compile(Op *op)
    {
        LOGv << "compile" << op;
        OpCompiler oc(op);
        string *src = &oc.src;
        for (auto op_type : op_types)
            op_type->post_pass(&oc);
        string src_after_passes;
        // if is fused op
        if (oc.op)
        {
            TunerManager tm(&oc);
            src_after_passes = tm.tune();
            src = &src_after_passes;
        }
        op->compile_optimize(*src);
        if (!op->flags.get(NodeFlags::_cuda))
        {
            LOGv << "compile cpu";
            return oc.compile(op->get_jit_key(get_jk()), *src);
        }
        if (op->name() == string("fused"))
        {
            FusedOp *fop = (FusedOp *)op;
            // if is a relayed op
            if (fop->context->vrm.relay_groups.size())
            {
                LOGv << "relay fused op";
                return oc.compile(op->get_jit_key(get_jk()), *src);
            }
            else
            {
                return &try_exec_and_fallback_cpu;
            }
        }
        else if (op->name() == string("code"))
        {
            CodeOp *cop = (CodeOp *)op;
            if (cop->cuda_src.find("acl") != string::npos)
            {
                LOGv << "compile acl op";
                return oc.compile(op->get_jit_key(get_jk()), *src);
            }
            else
            {
                return &exec_mapped_acl_ops;
            }
        }
        else
        {
            LOGv << "compile finish" << op;
            return &exec_mapped_acl_ops;
        }
        return do_compile_inner(op);
    }

    // from op_register.cc
    extern unordered_map<string, OpInfo> op_info_map;

    void init_acl_ops()
    {
        do_compile_hook = acl_do_compile;
        vector<string> to_erase;
        for (auto &kv : op_info_map)
        {
            if (startswith(kv.first, "cu") && acl_ops.count(kv.first) == 0)
            {
                to_erase.push_back(kv.first);
            }
        }
        for (auto &k : to_erase)
        {
            LOGv << "op not supported: " << k << ", erase it.";
            op_info_map.erase(k);
        }
    }

} // jittor