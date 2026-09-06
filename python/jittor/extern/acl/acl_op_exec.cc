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
#include <cstring>
#include <cstdio>
#include <exception>
#include <mutex>
#include <queue>
#include <set>
#include "common.h"
#include "op.h"
#include "acl_jittor.h"
#include "ops/random_op.h"
#include "ops/reduce_op.h"
#include "ops/arg_reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/transpose_op.h"
#include "ops/array_op.h"
#include "ops/code_op.h"
#include "ops/fused_adamw_op.h"
#include "fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "executor.h"
#include "runtime/device.h"
#include "runtime/backend_fallback.h"
#include "mem/allocator.h"
#include "op_compiler.h"
#include "ops/op_register.h"
#include "opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "aclops/aclops.h"
#include "aclops/native_indexing_op_acl.h"
namespace jittor
{
    void free_var_mem(Var *v);

    class AclScalarHostCache
    {
        static constexpr size_t capacity() { return 1 << 20; }
        std::mutex mutex;
        std::unordered_map<std::string, const void *> values;
        void *storage = nullptr;
        size_t storage_size = 0;
        size_t offset = 0;

        void reset()
        {
            if (storage == nullptr)
                return;
            auto ret = aclrtSynchronizeStream(aclstream);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error(
                    "aclrtSynchronizeStream failed: " +
                    acl_error_to_string(ret));
            ret = aclrtFreeHost(storage);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error(
                    "aclrtFreeHost failed: " + acl_error_to_string(ret));
            storage = nullptr;
            storage_size = 0;
            offset = 0;
            values.clear();
        }

    public:
        ~AclScalarHostCache()
        {
            if (storage != nullptr)
            {
                aclrtSynchronizeStream(aclstream);
                aclrtFreeHost(storage);
            }
        }

        const void *get(const void *data, size_t size)
        {
            std::string key(static_cast<const char *>(data), size);
            std::lock_guard<std::mutex> lock(mutex);
            auto iter = values.find(key);
            if (iter != values.end())
                return iter->second;

            size_t aligned_size = (size + 63) / 64 * 64;
            if (storage == nullptr || offset + aligned_size > storage_size)
            {
                reset();
                storage_size = std::max(capacity(), aligned_size);
                auto ret = aclrtMallocHost(&storage, storage_size);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error(
                        "aclrtMallocHost failed: " +
                        acl_error_to_string(ret));
            }
            void *copy = static_cast<char *>(storage) + offset;
            std::memcpy(copy, data, size);
            offset += aligned_size;
            values.emplace(std::move(key), copy);
            return copy;
        }
    };

    static const void *persistent_acl_scalar_data(const void *data, size_t size)
    {
        // Async H2D copies may outlive temporary scalar buffers owned by an op.
        static AclScalarHostCache cache;
        return cache.get(data, size);
    }

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

    class AclCpuFallbackScope
    {
        int previous_mode;
        vector<std::pair<Op *, std::pair<int, int>>> flags;
        FusedOp *fused = nullptr;
        FusedOpContext *context = nullptr;
        loop_options_t tuned;
        loop_options_t *options = nullptr;

    public:
        AclCpuFallbackScope(const AclCpuFallbackScope &) = delete;
        AclCpuFallbackScope &operator=(const AclCpuFallbackScope &) = delete;

        explicit AclCpuFallbackScope(Op *op)
            : previous_mode(runtime_device_state().use_cuda)
        {
            vector<Op *> operators{op};
            if (op->name() == string("fused"))
            {
                fused = static_cast<FusedOp *>(op);
                operators.insert(operators.end(), fused->ops.begin(), fused->ops.end());
                context = fused->context;
                tuned = fused->loop_options_tuned;
                options = fused->loop_options;
            }
            for (auto *item : operators)
                flags.push_back({item, {item->flag(OpFlags::_cpu), item->flag(OpFlags::_cuda)}});
            runtime_device_state().use_cuda = 0;
            for (const auto &saved : flags)
            {
                saved.first->set_flag(OpFlags::_cpu);
                saved.first->set_flag(OpFlags::_cuda, 0);
            }
        }

        ~AclCpuFallbackScope()
        {
            for (const auto &saved : flags)
            {
                saved.first->set_flag(OpFlags::_cpu, saved.second.first);
                saved.first->set_flag(OpFlags::_cuda, saved.second.second);
            }
            if (fused)
            {
                fused->context = context;
                fused->loop_options_tuned.swap(tuned);
                fused->loop_options = options;
            }
            runtime_device_state().use_cuda = previous_mode;
        }
    };

    template<class Execute, class Fallback, class Cleanup>
    static void dispatch_acl_checked(const string &unsupported,
                                     Execute execute, Fallback fallback, Cleanup cleanup)
    {
        if (!unsupported.empty())
        {
            fallback(unsupported);
            return;
        }
        try
        {
            execute();
        }
        catch (...)
        {
            const auto original = std::current_exception();
            try { cleanup(); }
            catch (...) { std::fprintf(stderr, "ACL cleanup failed; preserving execution error\n"); }
            std::rethrow_exception(original);
        }
    }

    template<class Runner, bool UsesRegistry = true>
    class AclExecutionRunner : public Runner
    {
    public:
        using Runner::Runner;

        void run()
        {
            auto entry = aclOpFuncMap.end();
            if (UsesRegistry)
            {
                entry = aclOpFuncMap.find(this->name);
                INTERNAL_ASSERT(entry != aclOpFuncMap.end())
                    << "ACL launcher disappeared after preflight:" << this->name;
            }
            try
            {
                this->setupInputDesc();
                this->setupOutputDesc();
                this->executeOp(entry);
                this->cleanupDesc();
            }
            catch (...)
            {
                // setup may have constructed only some descriptors; cleanupDesc
                // assumes complete vectors, so use the actual constructed set.
                aclrtSynchronizeStream(aclstream);
                for (auto *tensor : this->inputTensors)
                    if (tensor) aclDestroyTensor(tensor);
                for (auto *tensor : this->outputTensors)
                    if (tensor) aclDestroyTensor(tensor);
                throw;
            }
        }
    };

    void fallback_cpu(Op *op, const string &reason)
    {
        check_backend_fallback(op->name(), accelerator_backend_id(), BackendId::Cpu, reason);
        USER_CHECK(op->definition().implementations.count(BackendId::Cpu))
            << "No CPU implementation for fallback of" << op->name();
        if (op->name() == string("code"))
            USER_CHECK(!static_cast<CodeOp *>(op)->cpu_src.empty())
                << "No CPU source for unsupported ACL code operator";
        AclCpuFallbackScope restore(op);
        for (auto v : op->inputs())
        {
            if (v->mem_ptr && v->allocator->is_cuda())
            {
                migrate_to_cpu(v, runtime_executor().allocator);
            }
        }
        for (auto v : op->outputs())
        {
            if (v->mem_ptr && v->allocator->is_cuda())
            {
                migrate_to_cpu(v, runtime_executor().allocator);
            }
        }
        op->do_run();
    }

    static string binary_acl_name(BinaryOp *op)
    {
        auto found = opname_map.find(op->ns);
        if (found == opname_map.end()) return {};
        if (op->x->dtype() == ns_bool && op->y->dtype() == ns_bool)
        {
            if (op->ns == ns_bitwise_or) return "LogicalOr";
            if (op->ns == ns_bitwise_and) return "LogicalAnd";
            if (op->ns == ns_bitwise_xor) return "LogicalXor";
        }
        return found->second;
    }

    static string fused_acl_name(Op *op)
    {
        const string name = op->name();
        if (name == "unary")
        {
            auto found = opname_map.find(op->ns);
            return found == opname_map.end() ? string() : found->second;
        }
        if (name == "binary") return binary_acl_name(static_cast<BinaryOp *>(op));
        if (name == "ternary") return "Select";
        if (name == "broadcast_to") return "Expand";
        if (name == "fuse_transpose") return "Transpose";
        if (name == "reduce")
        {
            if (op->ns == ns_add) return "ReduceSum";
            if (op->ns == ns_mean) return "ReduceMean";
            if (op->ns == ns_maximum) return "ReduceMax";
            if (op->ns == ns_minimum) return "ReduceMin";
            if (op->ns == ns_multiply) return "ReduceProd";
        }
        return {};
    }

    static bool acl_has_dtype(NanoString dtype)
    {
        return dtype == ns_bfloat16 || dtype == ns_float32 || dtype == ns_float16
            || dtype == ns_int64 || dtype == ns_int32 || dtype == ns_int8
            || dtype == ns_int16 || dtype == ns_uint8 || dtype == ns_uint16
            || dtype == ns_uint32 || dtype == ns_bool || dtype == ns_complex64;
    }

    static string fused_acl_unsupported(const vector<Op *> &ops)
    {
        for (auto *op : ops)
        {
            if (op->name() == string("array")) continue;
            const auto name = fused_acl_name(op);
            if (name.empty()) return string("unregistered fused operator variant: ") + op->name() + "/" + S(op->ns);
            auto found = aclOpFuncMap.find(name);
            if (found == aclOpFuncMap.end()) return "unregistered ACL launcher: " + name;
            INTERNAL_ASSERT(found->second.executeFunc) << "Empty registered ACL launcher:" << name;
            if (op->name() == string("unary"))
                INTERNAL_ASSERT(name == "Cast" ? bool(found->second.getWorkspaceSizeFuncCast)
                                               : bool(found->second.getWorkspaceSizeFuncUnaryNonzero))
                    << "Wrong registered unary launcher signature:" << name;
            if (op->name() == string("binary"))
                INTERNAL_ASSERT(name == "Add" || name == "Sub"
                    ? bool(found->second.getWorkspaceSizeFuncAdd)
                    : bool(found->second.getWorkspaceSizeFuncBinary))
                    << "Wrong registered binary launcher signature:" << name;
            for (auto *input : op->inputs())
                if (!acl_has_dtype(input->dtype())) return name + " does not support input dtype " + S(input->dtype());
            for (auto *output : op->outputs())
                if (!acl_has_dtype(output->dtype())) return name + " does not support output dtype " + S(output->dtype());
            if ((name == "Add" || name == "Sub") && op->input(0)->dtype() == ns_complex64)
                return name + " has no complex alpha-scalar implementation";
        }
        return {};
    }

    static void exec_acl_sequence(Op *op, const vector<Op *> &ops)
    {
        std::set<Var *> new_alloced;
        map<Op *, int> op_indeg;
        map<Var *, int> var_outdeg;
        std::queue<Op *> queue;

        for (Op *op : ops)
            op_indeg[op] = 0;

        map<Op *, vector<Op *>> out_map;
        map<Var *, vector<Op *>> from;

        int len = 0;
        for (Op *v : ops)
        {
            for (auto in : v->inputs())
                from[in].push_back(v);
            ++len;
        }
        for (Op *u : ops)
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
        for (Op *op : ops)
        {
            if (op_indeg[op] == 0)
                queue.push(op);
        }

        int total = 0;
        dispatch_acl_checked(fused_acl_unsupported(ops), [&]
        {
            while (!queue.empty())
            {
                total++;
                auto *current_op = queue.front();
                queue.pop();
                for (auto in : current_op->inputs())
                {
                    ASSERT(in->mem_ptr)
                        << "current fused operator input is not allocated:"
                        << current_op->name() << in;
                }
                for (auto out : current_op->outputs())
                {
                    if (out->mem_ptr)
                        continue;
                    new_alloced.insert(out);
                    INTERNAL_ASSERT(out->alloc(runtime_executor().allocator))
                        << "ACL fused output allocation returned no storage";
                }
                for (auto out : out_map[current_op])
                {
                    --op_indeg[out];
                    if (op_indeg[out] == 0)
                        queue.push(out);
                }
                if (current_op->name() == string("unary"))
                {
                    auto uop = (UnaryOp *)current_op;
                    AclExecutionRunner<UnaryOpRunner> op;
                    op.add(uop->x, true);
                    op.add(uop->y, false);
                    op.name = fused_acl_name(current_op);
                    op.jt_name = uop->name();
                    op.run();
                }
                else if (current_op->name() == string("binary"))
                {
                    auto bop = (BinaryOp *)current_op;
                    AclExecutionRunner<BinaryOpRunner> op;
                    op.add(bop->x, true);
                    op.add(bop->y, true);
                    op.add(bop->z, false);
                    op.name = fused_acl_name(current_op);
                    op.jt_name = bop->name();
                    op.run();
                }
                else if (current_op->name() == string("ternary"))
                {
                    auto top = (TernaryOp *)current_op;
                    AclExecutionRunner<TernaryOpRunner> op;
                    op.name = fused_acl_name(current_op);
                    op.add(top->cond, true);
                    op.add(top->x, true);
                    op.add(top->y, true);
                    op.add(top->z, false);
                    op.run();
                }
                else if (current_op->name() == string("array"))
                {
                    auto aop = (ArrayOp *)current_op;
                    // The fused allocator can reuse a consumed constant's
                    // buffer before earlier ACL kernels finish. Queue the H2D
                    // copy on aclstream so reuse stays ordered with consumers.
                    const void *source = aop->ptr<void>();
                    if (aop->output->flag(VarFlags::_is_scalar))
                        source = persistent_acl_scalar_data(
                            source, aop->output->size);
                    auto ret = aclrtMemcpyAsync(
                        aop->output->mem_ptr, aop->output->size,
                        source, aop->output->size,
                        ACL_MEMCPY_HOST_TO_DEVICE, aclstream);
                    if (ret != ACL_SUCCESS)
                        throw std::runtime_error(
                            "aclrtMemcpyAsync failed: " +
                            acl_error_to_string(ret));
                }
                else if (current_op->name() == string("reduce"))
                {
                    auto rop = (ReduceOp *)current_op;
                    AclExecutionRunner<ReduceOpRunner> op;
                    op.name = fused_acl_name(current_op);
                    if (rop->ns == ns_add)
                        op.op_idx = 9;
                    else if (rop->ns == ns_multiply)
                        op.op_idx = 13;
                    else if (rop->ns == ns_maximum)
                        op.op_idx = 11;
                    else if (rop->ns == ns_minimum)
                        op.op_idx = 12;
                    else if (rop->ns == ns_mean)
                        op.op_idx = 10;
                    else
                        LOGf << "op " << rop->ns << " not supported";
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
                    // sync removed: aclstream is in-order and the shared
                    // workspace is guarded in mallocWorkSpace(). Draining here
                    // stalled the pipeline on every reduce (LayerNorm/softmax
                    // backward/Adam), which dominated NPU step time.
                }
                else if (current_op->name() == string("broadcast_to"))
                {
                    auto bop = (BroadcastToOp *)current_op;
                    AclExecutionRunner<ExpandOpRunner> op;
                    op.name = fused_acl_name(current_op);
                    op.jt_name = "expand";
                    NanoVector xshape, xshape_bk = bop->x->shape;
                    NanoVector zshape = bop->z->shape;
                    struct RestoreShape {
                        Var *value;
                        NanoVector shape;
                        ~RestoreShape() { value->shape = shape; }
                    } restore_shape{bop->x, xshape_bk};

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
                    op.add(bop->z, false);
                    op.run();
                    // shape is copied into the aclTensor synchronously during
                    // op.run() (CreateAclTensor), so restoring it right after
                    // is safe without draining the stream. sync removed for the
                    // same reason as the reduce case above.
                    bop->x->shape = xshape_bk;
                }
                else if (current_op->name() == string("fuse_transpose"))
                {
                    // replace fuse_transpose with transpose
                    auto top = (TransposeOp *)current_op;
                    AclExecutionRunner<TransposeOpRunner> op;
                    op.name = fused_acl_name(current_op);
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
                    LOGf << "op " << current_op->name() << " not supported";
                }

                for (auto in : current_op->inputs())
                {
                    --var_outdeg[in];
                    if (var_outdeg[in] == 0)
                    {
                        if (new_alloced.find(in) != new_alloced.end())
                        {
                            new_alloced.erase(in);
                            free_var_mem(in);
                        }
                    }
                }
            }
            INTERNAL_ASSERT(total == len) << "ACL fused graph has unresolved dependencies";
            while (!new_alloced.empty())
            {
                auto *value = *new_alloced.begin();
                new_alloced.erase(new_alloced.begin());
                free_var_mem(value);
            }
        }, [&](const string &reason)
        {
            fallback_cpu(op, reason);
        }, [&]
        {
            // Only an abandoned execution is drained. No SDK/kernel exception
            // is interpreted as permission to run a different backend.
            auto status = aclrtSynchronizeStream(aclstream);
            if (status != ACL_SUCCESS)
                std::fprintf(stderr, "ACL failure cleanup drain returned %d\n", int(status));
            for (auto *value : new_alloced)
            {
                try { if (value->allocator) free_var_mem(value); }
                catch (...) { std::fprintf(stderr, "ACL temporary cleanup failed\n"); }
            }
        });
    }

    void exec_fused_acl(Op *op)
    {
        exec_acl_sequence(op, static_cast<FusedOp *>(op)->ops);
    }

    static void exec_single_acl(Op *op)
    {
        exec_acl_sequence(op, {op});
    }

    extern int current_seed;
    extern int64 current_offset;

    static unordered_map<string, std::function<void(Op *)>> acl_ops = {
        {"getitem", exec_native_acl_getitem},
        {"setitem", exec_native_acl_setitem},
        {"fused_adamw", [](Op *op)
         {
             auto _op = (FusedAdamwOp *)op;
             AclExecutionRunner<AdamWListOpRunner, false> runner;
             AdamWAttr *attr = new AdamWAttr();
             attr->tensorCount = _op->parameters.size();
             attr->lr = _op->lr;
             attr->beta1 = _op->beta1;
             attr->beta2 = _op->beta2;
             attr->weightDecay = _op->weight_decay;
             attr->eps = _op->eps;
             runner.jt_name = "fused_adamw";
             runner.op_attr.reset(attr);
             for (auto value : _op->parameters) runner.add(value, true);
             for (auto value : _op->moments) runner.add(value, true);
             for (auto value : _op->variances) runner.add(value, true);
             for (auto value : _op->gradients) runner.add(value, true);
             runner.add(_op->step, true);
             for (auto value : _op->new_parameters) runner.add(value, false);
             for (auto value : _op->new_moments) runner.add(value, false);
             for (auto value : _op->new_variances) runner.add(value, false);
             runner.run();
         }},
        {"arg_reduce", [](Op *op)
         {
             auto _op = (ArgReduceOp *)op;
             AclExecutionRunner<ArgReduceOpRunner, false> runner(
                 _op->op == ns_maximum, _op->dim, _op->keepdims);
             runner.jt_name = "arg_reduce";
             runner.add(_op->x, true);
             runner.add(_op->y, false);
             runner.add(_op->y_key, false);
             runner.run();
         }},
        {"curand_random", [&current_seed, &current_offset](Op *op)
         {
             auto _op = (RandomOp *)op;
             AclExecutionRunner<RandomOpRunner> runner(_op->type == ns_uniform ? "RandomUniform" : "RandomNormal");
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
        string unsupported;
        if (iter == acl_ops.end())
            unsupported = string("no registered ACL implementation for ") + op->name();
        else
        {
            INTERNAL_ASSERT(iter->second) << "Empty ACL implementation for" << op->name();
            for (auto *input : op->inputs())
                if (!acl_has_dtype(input->dtype()))
                    unsupported = string(op->name()) + " does not support input dtype " + S(input->dtype());
            for (auto *output : op->outputs())
                if (!acl_has_dtype(output->dtype()))
                    unsupported = string(op->name()) + " does not support output dtype " + S(output->dtype());
            if (unsupported.empty() && op->name() == string("getitem"))
                unsupported = acl_getitem_unsupported_reason(op);
            if (unsupported.empty() && op->name() == string("setitem"))
                unsupported = acl_setitem_unsupported_reason(op);
            if (op->name() == string("arg_reduce"))
            {
                auto *reduce = static_cast<ArgReduceOp *>(op);
                USER_CHECK(reduce->op == ns_maximum || reduce->op == ns_minimum)
                    << "arg_reduce requires min or max";
            }
            if (op->name() == string("curand_random"))
            {
                auto *random = static_cast<RandomOp *>(op);
                USER_CHECK(random->type == ns_uniform || random->type == ns_normal)
                    << "random requires uniform or normal";
                const string name = random->type == ns_uniform ? "RandomUniform" : "RandomNormal";
                if (!aclOpFuncMap.count(name)) unsupported = "unregistered ACL launcher: " + name;
            }
        }
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            iter->second(op);
        }, [&](const string &reason)
        {
            fallback_cpu(op, reason);
        }, [&]
        {
            aclrtSynchronizeStream(aclstream);
        });
    }

    static jit_op_entry_t compile_acl_fused(Op *op)
    {
        LOGv << "compile" << op;
        OpCompiler oc(op);
        string *src = &oc.src;
        for (auto op_type : get_op_types())
            op_type->post_pass(&oc);
        string src_after_passes;
        // if is fused op
        if (oc.op)
        {
            TunerManager tm(&oc);
            src_after_passes = tm.tune();
            src = &src_after_passes;
        }
        op->optimize_generated_source(*src);
        auto *fop = static_cast<FusedOp *>(op);
        // Tuning creates relay groups. Choose the executable only after the
        // registered passes have finished, preserving their source and key.
        if (!fop->context->vrm.relay_groups.empty())
        {
            LOGv << "relay fused op";
            return oc.compile(op->get_jit_key(get_jk()), *src);
        }
        return &exec_fused_acl;
    }

    static void exec_unsupported_acl(Op *op)
    {
        fallback_cpu(op, string("no registered ACL implementation for ") + op->name());
    }

    static jit_op_entry_t compile_acl_unsupported(Op *) { return &exec_unsupported_acl; }
    static jit_op_entry_t compile_acl_mapped(Op *) { return &exec_mapped_acl_ops; }
    static jit_op_entry_t compile_acl_single(Op *) { return &exec_single_acl; }

    static void exec_unmarked_acl_code(Op *op)
    {
        fallback_cpu(op, "accelerator source is not explicitly marked backend=acl");
    }

    static jit_op_entry_t compile_acl_code(Op *op)
    {
        const auto *code = static_cast<CodeOp *>(op);
        if (code->backend != "acl") return &exec_unmarked_acl_code;
        return compile_registered_source(op);
    }

    static OpImplementation compose_acl_implementation(
        const OpDef &definition, const OpImplementation &original)
    {
        // The same launcher sequence handles fused graphs and a standalone
        // primitive. Unsupported implementations stay explicit fallback entries;
        // other backends' OpDefs and constructors are never removed.
        static const set<string> primitives = {
            "unary", "binary", "ternary", "broadcast_to", "fuse_transpose", "reduce"};
        auto implementation = original;
        const auto &name = definition.name;
        // Backend-native extensions such as HCCL declare their own compiler
        // through configure_accelerator_kernel at registration.
        if (implementation.kernel.compile) return implementation;
        if (name == "fused")
            implementation.kernel.compile = compile_acl_fused;
        else if (name == "code")
            implementation.kernel.compile = compile_acl_code;
        else if (acl_ops.count(name))
        {
            implementation.kernel.compile = compile_acl_mapped;
            implementation.kernel.native = exec_mapped_acl_ops;
        }
        else if (primitives.count(name))
        {
            implementation.kernel.compile = compile_acl_single;
            implementation.kernel.native = exec_single_acl;
        }
        else
        {
            implementation.kernel.compile = compile_acl_unsupported;
            implementation.kernel.fallback_only = !implementation.kernel.native;
        }
        return implementation;
    }

    void init_acl_ops()
    {
        register_backend_implementation_composer(
            BackendId::Acl, compose_acl_implementation, "acl-native-v1");
    }

} // jittor
