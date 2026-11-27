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
#include "core/common.h"
#include "core/op.h"
#include "acl_jittor.h"
#include "acl_runtime.h"
#include "ops/composite/random_op.h"
#include "ops/reduce_op.h"
#include "ops/composite/arg_reduce_op.h"
#include "ops/composite/argsort_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/code_op.h"
#include "ops/composite/fused_adamw_op.h"
#include "ops/composite/fused_sgd_op.h"
#include "ops/composite/mapped_matmul_op.h"
#include "ops/composite/mapped_concat_op.h"
#include "ops/composite/mapped_packed_matmul_op.h"
#include "ops/composite/mapped_rms_norm_op.h"
#include "ops/composite/mapped_fp8_linear_op.h"
#include "ops/composite/mapped_unary_op.h"
#include "ops/composite/mapped_grouped_matmul_op.h"
#include "ops/composite/mapped_histc_op.h"
#include "ops/composite/mapped_addmm_op.h"
#include "ops/composite/mapped_flash_attention_op.h"
#include "ops/composite/mapped_conv2d_op.h"
#include "ops/composite/mapped_conv2d_backward_op.h"
#include "ops/composite/mapped_batch_norm_op.h"
#include "ops/composite/mapped_batch_norm_backward_op.h"
#include "ops/composite/mapped_layer_norm_op.h"
#include "ops/composite/mapped_layer_norm_backward_op.h"
#include "ops/composite/mapped_unary_grad_op.h"
#include "ops/composite/write_back_op.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "runtime/rng_state.h"
#include "runtime/backend_fallback.h"
#include "mem/allocator.h"
#include "codegen/op_compiler.h"
#include "ops/op_register.h"
#include "codegen/opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "aclops/aclops.h"
#include "aclops/native_indexing_op_acl.h"
#include "acl_fused_ascendc.h"
namespace jittor
{
    // FP32 cube ops (matmul/bmm/conv) down-convert through aclnn
    // ALLOW_FP32_DOWN_PRECISION only when explicitly enabled. Inference SoCs
    // (Ascend310P) have no FP32 cube path at all, so there KEEP_DTYPE fails;
    // this flag is the opt-in, mirroring torch.npu.matmul.allow_hf32. The
    // storage lives in src/runtime/init.cc: the flags binding generator only
    // scans the core source tree, and a DEFINE here would leave
    // `jt.flags.acl_allow_hf32` unbound.
    DECLARE_FLAG(int, acl_allow_hf32);
    DECLARE_FLAG(int, profiler_enable);
    DECLARE_FLAG(int, trace_py_var);

    void free_var_mem(Var *v);

    class AclScalarHostCache
    {
        static constexpr size_t capacity() { return 1 << 20; }
        std::mutex mutex;
        std::unordered_map<std::string, const void *> values;
        void *storage = nullptr;
        size_t storage_size = 0;
        size_t offset = 0;
        // Storage replaced while a recorded graph lived: its copies read
        // from here. Freed with the last graph (acl_live_graphs).
        std::vector<void *> retired;

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
            if (storage != nullptr || !retired.empty())
                aclrtSynchronizeStream(aclstream);
            if (storage != nullptr)
                aclrtFreeHost(storage);
            for (auto *block : retired)
                aclrtFreeHost(block);
        }

        void release_retired()
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (retired.empty())
                return;
            auto ret = aclrtSynchronizeStream(aclstream);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error(
                    "aclrtSynchronizeStream failed: " +
                    acl_error_to_string(ret));
            for (auto *block : retired)
            {
                ret = aclrtFreeHost(block);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error(
                        "aclrtFreeHost failed: " + acl_error_to_string(ret));
            }
            retired.clear();
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
                if (storage != nullptr && acl_live_graphs() > 0)
                {
                    // Kept, and nothing waited on: the stream may be the
                    // one recording.
                    retired.push_back(storage);
                    storage = nullptr;
                    storage_size = 0;
                    offset = 0;
                    values.clear();
                }
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

    static AclScalarHostCache &scalar_host_cache()
    {
        // Async H2D copies may outlive temporary scalar buffers owned by an op.
        static AclScalarHostCache cache;
        return cache;
    }

    static const void *persistent_acl_scalar_data(const void *data, size_t size)
    {
        return scalar_host_cache().get(data, size);
    }

    void acl_scalar_cache_release_retired()
    {
        scalar_host_cache().release_retired();
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
        {ns_fmod, "Fmod"},
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
        ExecutionBackendScope execution_scope{BackendId::Cpu};
        HostExecutionScope host_execution;
        int previous_mode;
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
            if (op->name() == string("fused"))
            {
                fused = static_cast<FusedOp *>(op);
                context = fused->context;
                tuned = fused->loop_options_tuned;
                options = fused->loop_options;
            }
            runtime_device_state().use_cuda = 0;
        }

        ~AclCpuFallbackScope()
        {
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
            auto entry = acl_op_registry().end();
            if (UsesRegistry)
            {
                entry = acl_op_registry().find(this->name);
                INTERNAL_ASSERT(entry != acl_op_registry().end())
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

    static inline bool op_name_is(const Op *op, const char *target)
    {
        return std::strcmp(op->name(), target) == 0;
    }

    // Fast cached lookup from NanoString index -> (ACL op name, registry entry)
    // so fused preflight and unary/binary launches avoid per-op std::string
    // construction and unordered_map hashing on every operator.
    struct FastAclOpEntry
    {
        const char *name = nullptr;
        AclOpRegistry::const_iterator iter;
        bool valid = false;
        bool is_add_sub = false;
        bool is_cast = false;
    };

    static const FastAclOpEntry &fast_acl_ns_entry(NanoString ns)
    {
        static constexpr size_t kMaxIdx = ns_max_size;
        static const struct Table
        {
            FastAclOpEntry by_idx[kMaxIdx];
            FastAclOpEntry empty;
            Table()
            {
                const auto &reg = acl_op_registry();
                for (const auto &kv : opname_map)
                {
                    NanoString key;
                    key.data = kv.first;
                    const uint32 idx = key.index();
                    if (idx < kMaxIdx)
                    {
                        auto found = reg.find(kv.second);
                        if (found != reg.end())
                        {
                            FastAclOpEntry &e = by_idx[idx];
                            e.name = found->first.c_str();
                            e.iter = found;
                            e.valid = true;
                            e.is_add_sub = (kv.second == "Add" || kv.second == "Sub");
                            e.is_cast = (kv.second == "Cast");
                        }
                    }
                }
            }
        } table;
        const uint32 idx = ns.index();
        return idx < kMaxIdx ? table.by_idx[idx] : table.empty;
    }

    static const FastAclOpEntry *fast_acl_entry_for_op(Op *op, const char *cname)
    {
        if (std::strcmp(cname, "unary") == 0)
        {
            if (PREDICT_BRANCH_NOT_TAKEN(op->ns == ns_conj))
            {
                if (!static_cast<UnaryOp *>(op)->x->dtype().is_complex())
                {
                    const auto &e = fast_acl_ns_entry(ns_cast);
                    return e.valid ? &e : nullptr;
                }
                return nullptr;
            }
            const auto &e = fast_acl_ns_entry(op->ns);
            return e.valid ? &e : nullptr;
        }
        if (std::strcmp(cname, "binary") == 0)
        {
            auto *bop = static_cast<BinaryOp *>(op);
            NanoString key = bop->ns;
            if (bop->x->dtype() == ns_bool && bop->y->dtype() == ns_bool)
            {
                if (bop->ns == ns_bitwise_or) key = ns_logical_or;
                else if (bop->ns == ns_bitwise_and) key = ns_logical_and;
                else if (bop->ns == ns_bitwise_xor) key = ns_logical_xor;
            }
            const auto &e = fast_acl_ns_entry(key);
            return e.valid ? &e : nullptr;
        }
        if (std::strcmp(cname, "contiguous") == 0)
        {
            const auto &e = fast_acl_ns_entry(ns_cast);
            return e.valid ? &e : nullptr;
        }
        return nullptr;
    }

    template<class Runner>
    class AclCachedEntryRunner : public Runner
    {
    public:
        using Runner::Runner;

        void run_cached(AclOpRegistry::const_iterator entry)
        {
            try
            {
                this->setupInputDesc();
                this->setupOutputDesc();
                this->executeOp(entry);
                this->cleanupDesc();
            }
            catch (...)
            {
                aclrtSynchronizeStream(aclstream);
                for (auto *tensor : this->inputTensors)
                    if (tensor) aclDestroyTensor(tensor);
                for (auto *tensor : this->outputTensors)
                    if (tensor) aclDestroyTensor(tensor);
                throw;
            }
        }
    };

    static string fused_acl_name(Op *op)
    {
        const char *name = op->name();
        if (std::strcmp(name, "contiguous") == 0) return "Cast";
        if (std::strcmp(name, "unary") == 0)
        {
            if (PREDICT_BRANCH_NOT_TAKEN(op->ns == ns_conj))
            {
                if (!static_cast<UnaryOp *>(op)->x->dtype().is_complex())
                    return "Cast";
                return {};
            }
            const auto &fast = fast_acl_ns_entry(op->ns);
            if (fast.valid) return string(fast.name);
            auto found = opname_map.find(op->ns);
            return found == opname_map.end() ? string() : found->second;
        }
        if (std::strcmp(name, "binary") == 0)
        {
            auto *bop = static_cast<BinaryOp *>(op);
            NanoString key = bop->ns;
            if (bop->x->dtype() == ns_bool && bop->y->dtype() == ns_bool)
            {
                if (bop->ns == ns_bitwise_or) key = ns_logical_or;
                else if (bop->ns == ns_bitwise_and) key = ns_logical_and;
                else if (bop->ns == ns_bitwise_xor) key = ns_logical_xor;
            }
            const auto &fast = fast_acl_ns_entry(key);
            if (fast.valid) return string(fast.name);
            return binary_acl_name(bop);
        }
        if (std::strcmp(name, "ternary") == 0) return "Select";
        if (std::strcmp(name, "broadcast_to") == 0) return "Expand";
        if (std::strcmp(name, "fuse_transpose") == 0) return "Transpose";
        if (std::strcmp(name, "reduce") == 0)
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
        if (dtype == ns_float8_e4m3fn || dtype == ns_float8_e5m2)
            return acl_soc_supports_fp8();
        return dtype == ns_bfloat16 || dtype == ns_float32 || dtype == ns_float16
            || dtype == ns_int64 || dtype == ns_int32 || dtype == ns_int8
            || dtype == ns_int16 || dtype == ns_uint8 || dtype == ns_uint16
            || dtype == ns_uint32 || dtype == ns_bool || dtype == ns_complex64;
    }

    static bool acl_fp8_cast_pair_supported(NanoString src, NanoString dst)
    {
        auto is_e4_or_e5 = [](NanoString d) {
            return d == ns_float8_e4m3fn || d == ns_float8_e5m2;
        };
        auto is_wide_float = [](NanoString d) {
            return d == ns_float32 || d == ns_float16 || d == ns_bfloat16;
        };
        if (src == dst && is_e4_or_e5(src)) return true;
        if (is_e4_or_e5(src) && is_wide_float(dst)) return true;
        if (is_wide_float(src) && is_e4_or_e5(dst)) return true;
        return false;
    }

    static string fused_acl_unsupported(const vector<Op *> &ops)
    {
        const auto &registry = acl_op_registry();
        for (auto *op : ops)
        {
            const char *cname = op->name();
            if (std::strcmp(cname, "array") == 0) continue;
            const FastAclOpEntry *fast = fast_acl_entry_for_op(op, cname);
            const char *acl_cname = nullptr;
            if (fast)
            {
                acl_cname = fast->name;
            }
            else
            {
                const auto name = fused_acl_name(op);
                if (name.empty()) return string("unregistered fused operator variant: ") + cname + "/" + string(op->ns.to_cstring());
                auto found = registry.find(name);
                if (found == registry.end()) return "unregistered ACL launcher: " + name;
                INTERNAL_ASSERT(found->second.launcher()) << "Empty registered ACL launcher:" << name;
                if (std::strcmp(cname, "unary") == 0 || std::strcmp(cname, "contiguous") == 0)
                    INTERNAL_ASSERT(found->second.supports(name == "Cast"
                        ? AclOpFunctions::QueryKind::Cast : AclOpFunctions::QueryKind::Unary))
                        << "Wrong registered unary launcher signature:" << name;
                if (std::strcmp(cname, "binary") == 0)
                    INTERNAL_ASSERT(found->second.supports(name == "Add" || name == "Sub"
                        ? AclOpFunctions::QueryKind::Add : AclOpFunctions::QueryKind::Binary))
                        << "Wrong registered binary launcher signature:" << name;
                acl_cname = found->first.c_str();
            }
            bool has_fp8 = false;
            for (auto *input : op->inputs())
            {
                const NanoString dt = input->dtype();
                if (PREDICT_BRANCH_NOT_TAKEN(ns_is_fp8(dt))) has_fp8 = true;
                if (PREDICT_BRANCH_NOT_TAKEN(!acl_has_dtype(dt)))
                    return string(acl_cname) + " does not support input dtype " + string(dt.to_cstring());
            }
            for (auto *output : op->outputs())
            {
                const NanoString dt = output->dtype();
                if (PREDICT_BRANCH_NOT_TAKEN(ns_is_fp8(dt))) has_fp8 = true;
                if (PREDICT_BRANCH_NOT_TAKEN(!acl_has_dtype(dt)))
                    return string(acl_cname) + " does not support output dtype " + string(dt.to_cstring());
            }
            if (PREDICT_BRANCH_NOT_TAKEN(has_fp8))
            {
                if (!acl_soc_supports_fp8())
                    return string(acl_cname) + " FP8 requires Ascend950-class hardware";
                if (std::strcmp(acl_cname, "Cast") == 0)
                {
                    if (op->inputs().size() != 1 || op->outputs().size() != 1 ||
                        !acl_fp8_cast_pair_supported(op->input(0)->dtype(), op->output(0)->dtype()))
                        return string(acl_cname) + " does not support FP8 cast pair " +
                               string(op->input(0)->dtype().to_cstring()) + " -> " +
                               string(op->output(0)->dtype().to_cstring());
                }
                else if (std::strcmp(acl_cname, "Transpose") != 0)
                {
                    return string(acl_cname) + " has no native ACL FP8 kernel";
                }
            }
            if (PREDICT_BRANCH_NOT_TAKEN((fast ? fast->is_add_sub : (std::strcmp(acl_cname, "Add") == 0 || std::strcmp(acl_cname, "Sub") == 0))
                && op->input(0)->dtype() == ns_complex64))
                return string(acl_cname) + " has no complex alpha-scalar implementation";
        }
        return {};
    }

    struct SmallVarMap
    {
        vector<std::pair<Var *, Var *>> items;
        void clear() { items.clear(); }
        bool empty() const { return items.empty(); }
        Var *find_or_null(Var *key) const
        {
            for (const auto &kv : items)
                if (kv.first == key) return kv.second;
            return nullptr;
        }
        bool count(Var *key) const { return find_or_null(key) != nullptr; }
        void set(Var *key, Var *val)
        {
            for (auto &kv : items)
                if (kv.first == key) { kv.second = val; return; }
            items.emplace_back(key, val);
        }
    };

    struct SmallHostScalarMap
    {
        vector<std::pair<Var *, ArrayOp *>> items;
        void clear() { items.clear(); }
        bool empty() const { return items.empty(); }
        ArrayOp *find_or_null(Var *key) const
        {
            for (const auto &kv : items)
                if (kv.first == key) return kv.second;
            return nullptr;
        }
        bool count(Var *key) const { return find_or_null(key) != nullptr; }
        void set(Var *key, ArrayOp *val)
        {
            for (auto &kv : items)
                if (kv.first == key) { kv.second = val; return; }
            items.emplace_back(key, val);
        }
    };

    struct SmallVarConsumers
    {
        // Flat indexed consumer table: avoids std::unordered_map node allocs
        // on every fused ACL launch (typical fused op has 2-16 unique Vars).
        vector<Var *> keys;
        vector<vector<size_t>> lists;

        void reset()
        {
            for (size_t i = 0; i < keys.size(); ++i)
                lists[i].clear();
            keys.clear();
        }
        int find_slot(Var *key) const
        {
            for (size_t i = 0; i < keys.size(); ++i)
                if (keys[i] == key) return static_cast<int>(i);
            return -1;
        }
        void add(Var *key, size_t op_idx)
        {
            int slot = find_slot(key);
            if (slot < 0)
            {
                slot = static_cast<int>(keys.size());
                keys.push_back(key);
                if (lists.size() <= static_cast<size_t>(slot))
                    lists.resize(slot + 1);
            }
            lists[slot].push_back(op_idx);
        }
        const vector<size_t> *get(Var *key) const
        {
            int slot = find_slot(key);
            return slot < 0 ? nullptr : &lists[slot];
        }
    };

    struct SmallVarIntMap
    {
        vector<std::pair<Var *, int>> items;
        void clear() { items.clear(); }
        int *find_ptr(Var *key)
        {
            for (auto &kv : items)
                if (kv.first == key) return &kv.second;
            return nullptr;
        }
        int &get_or_zero(Var *key)
        {
            for (auto &kv : items)
                if (kv.first == key) return kv.second;
            items.emplace_back(key, 0);
            return items.back().second;
        }
    };

    struct AclSeqScratch
    {
        vector<int> op_indeg;
        vector<vector<size_t>> adj;
        SmallVarConsumers consumers;
        SmallVarIntMap var_outdeg;
        SmallVarMap scalar_alias;
        SmallHostScalarMap host_scalar_alias;
        vector<std::pair<Op *, size_t>> queue_items;
        vector<Var *> new_alloced;
        vector<Var *> temp_allocs;
        bool in_use = false;
    };

    struct AclSeqScratchLease
    {
        AclSeqScratch *ptr = nullptr;
        std::unique_ptr<AclSeqScratch> fallback;
        AclSeqScratchLease()
        {
            static thread_local AclSeqScratch tl_scratch;
            if (!tl_scratch.in_use)
            {
                tl_scratch.in_use = true;
                ptr = &tl_scratch;
            }
            else
            {
                fallback.reset(new AclSeqScratch());
                ptr = fallback.get();
            }
        }
        ~AclSeqScratchLease()
        {
            if (!fallback && ptr)
                ptr->in_use = false;
        }
        AclSeqScratch &operator*() const { return *ptr; }
        AclSeqScratch *operator->() const { return ptr; }
    };

    // Fast path for ContiguousOp on ACL: when the input is already contiguous
    // use a single async D2D copy; when it is a pure permutation of a dense
    // buffer (e.g. NCHW<->NHWC or attention head transposes), run aclnnPermute
    // directly instead of strided aclnnCast (which is 50-100x slower on 4D views).
    static bool try_exec_acl_contiguous_permute(Var *in, Var *out)
    {
        if (!in || !out || in->dtype() != out->dtype() || !out->is_contiguous())
            return false;
        if (in->num <= 0 || in->num != out->num || in->shape != out->shape)
            return false;
        if (in->is_contiguous())
        {
            if (in->mem_ptr != out->mem_ptr)
            {
                auto st = aclrtMemcpyAsync(
                    out->mem_ptr, out->size, in->mem_ptr, out->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << "contiguous: aclrtMemcpyAsync failed. ERROR:" << st;
            }
            return true;
        }
        if (ns_is_fp8(in->dtype()) || in->storage_span_bytes() != out->size)
            return false;
        const int rank = static_cast<int>(in->shape.size());
        if (rank <= 0 || rank > 16) return false;
        int ns_axes[16];
        int order[16];
        int m = 0;
        for (int d = 0; d < rank; ++d)
        {
            if (in->shape[d] > 1)
            {
                if (in->storage_stride(d) <= 0) return false;
                ns_axes[m] = d;
                order[m] = d;
                ++m;
            }
        }
        if (m <= 1)
        {
            if (m == 1 && in->storage_stride(ns_axes[0]) != 1) return false;
            if (in->mem_ptr != out->mem_ptr)
            {
                auto st = aclrtMemcpyAsync(
                    out->mem_ptr, out->size, in->mem_ptr, out->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << "contiguous: aclrtMemcpyAsync failed. ERROR:" << st;
            }
            return true;
        }
        if (m > 8) return false;
        std::sort(order, order + m, [in](int a, int b) {
            return in->storage_stride(a) > in->storage_stride(b);
        });
        int64_t expected = 1;
        for (int k = m - 1; k >= 0; --k)
        {
            const int axis = order[k];
            if (in->storage_stride(axis) != expected)
                return false;
            expected *= in->shape[axis];
        }
        if (expected != in->num) return false;
        bool is_identity = true;
        int64_t perm[16];
        for (int j = 0; j < m; ++j)
        {
            int pos = -1;
            for (int k = 0; k < m; ++k)
            {
                if (order[k] == ns_axes[j]) { pos = k; break; }
            }
            if (pos < 0) return false;
            perm[j] = pos;
            if (pos != j) is_identity = false;
        }
        if (is_identity)
        {
            if (in->mem_ptr != out->mem_ptr)
            {
                auto st = aclrtMemcpyAsync(
                    out->mem_ptr, out->size, in->mem_ptr, out->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << "contiguous: aclrtMemcpyAsync failed. ERROR:" << st;
            }
            return true;
        }
        AclExecutionRunner<ContiguousPermuteOpRunner, false> runner;
        runner.jt_name = "contiguous";
        runner.perm_rank = m;
        for (int k = 0; k < m; ++k)
        {
            runner.base_shape[k] = in->shape[order[k]];
            runner.out_shape[k] = in->shape[ns_axes[k]];
            runner.perm[k] = perm[k];
        }
        runner.add(in, true);
        runner.add(out, false);
        runner.run();
        return true;
    }

    static bool try_exec_acl_contiguous_inplace_copy(Var *in, Var *out)
    {
        if (!in || !out || in->dtype() != out->dtype() || !out->is_contiguous())
            return false;
        if (in->num <= 0 || in->num != out->num || in->shape != out->shape)
            return false;
        const NanoString ns = in->dtype();
        if (ns != ns_float32 && ns != ns_float16 && ns != ns_bfloat16
                && ns != ns_int8 && ns != ns_int16 && ns != ns_int32 && ns != ns_int64
                && ns != ns_uint8 && ns != ns_uint16 && ns != ns_uint32 && ns != ns_bool)
            return false;
        const int rank = static_cast<int>(in->shape.size());
        if (rank <= 0 || rank > 16)
            return false;
        const int64_t dsize = in->dsize();
        if (dsize <= 0 || in->storage_span_bytes() <= 0 || (in->storage_span_bytes() % dsize) != 0)
            return false;
        for (int d = 0; d < rank; ++d)
        {
            if (in->storage_stride(d) < 0)
                return false;
        }
        AclExecutionRunner<ContiguousCopyOpRunner, false> runner;
        runner.jt_name = "contiguous";
        runner.add(in, true);
        runner.add(out, false);
        runner.run();
        return true;
    }

    static void exec_acl_primitive(Op *current_op,
                                   const SmallVarMap *scalar_alias = nullptr,
                                   const SmallHostScalarMap *host_scalar_alias = nullptr)
    {
        const char *cname = current_op->name();
        if (std::strcmp(cname, "contiguous") == 0)
        {
            Var *in_v = current_op->inputs().front();
            Var *out_v = current_op->outputs().front();
            if (try_exec_acl_contiguous_permute(in_v, out_v))
                return;
            if (try_exec_acl_contiguous_inplace_copy(in_v, out_v))
                return;
            const auto &fast = fast_acl_ns_entry(ns_cast);
            if (fast.valid)
            {
                AclCachedEntryRunner<UnaryOpRunner> runner;
                runner.add(in_v, true);
                runner.add(out_v, false);
                runner.name = "Cast";
                runner.jt_name = cname;
                runner.run_cached(fast.iter);
            }
            else
            {
                AclExecutionRunner<UnaryOpRunner> runner;
                runner.add(in_v, true);
                runner.add(out_v, false);
                runner.name = "Cast";
                runner.jt_name = cname;
                runner.run();
            }
        }
        else if (std::strcmp(cname, "unary") == 0)
        {
            auto uop = (UnaryOp *)current_op;
            if (PREDICT_BRANCH_NOT_TAKEN(uop->ns == ns_conj && !uop->x->dtype().is_complex()))
            {
                if (try_exec_acl_contiguous_permute(uop->x, uop->y))
                    return;
                const auto &fast_cast = fast_acl_ns_entry(ns_cast);
                if (fast_cast.valid)
                {
                    AclCachedEntryRunner<UnaryOpRunner> op;
                    op.add(uop->x, true);
                    op.add(uop->y, false);
                    op.name = "Cast";
                    op.jt_name = cname;
                    op.run_cached(fast_cast.iter);
                }
                else
                {
                    AclExecutionRunner<UnaryOpRunner> runner;
                    runner.add(uop->x, true);
                    runner.add(uop->y, false);
                    runner.name = "Cast";
                    runner.jt_name = cname;
                    runner.run();
                }
                return;
            }
            const auto &fast = fast_acl_ns_entry(uop->ns);
            if (fast.valid)
            {
                AclCachedEntryRunner<UnaryOpRunner> op;
                op.add(uop->x, true);
                op.add(uop->y, false);
                op.name = fused_acl_name(current_op);
                op.jt_name = cname;
                op.run_cached(fast.iter);
            }
            else
            {
                AclExecutionRunner<UnaryOpRunner> runner;
                runner.add(uop->x, true);
                runner.add(uop->y, false);
                runner.name = fused_acl_name(current_op);
                runner.jt_name = cname;
                runner.run();
            }
        }
        else if (std::strcmp(cname, "binary") == 0)
        {
            auto bop = (BinaryOp *)current_op;
            if (host_scalar_alias && !host_scalar_alias->empty())
            {
                ArrayOp *scalar_src = nullptr;
                Var *tensor_arg = nullptr;
                if (ArrayOp *sy = host_scalar_alias->find_or_null(bop->y))
                {
                    if (!host_scalar_alias->count(bop->x))
                    {
                        scalar_src = sy;
                        tensor_arg = bop->x;
                    }
                }
                else if (bop->ns == ns_add || bop->ns == ns_multiply)
                {
                    if (ArrayOp *sx = host_scalar_alias->find_or_null(bop->x))
                    {
                        if (!host_scalar_alias->count(bop->y))
                        {
                            scalar_src = sx;
                            tensor_arg = bop->y;
                        }
                    }
                }
                if (scalar_src && tensor_arg)
                {
                    if (scalar_alias && !scalar_alias->empty())
                    {
                        if (Var *rep = scalar_alias->find_or_null(tensor_arg))
                            tensor_arg = rep;
                    }
                    const auto &fast_sc = fast_acl_ns_entry(bop->ns);
                    AclCachedEntryRunner<BinaryScalarOpRunner> runner;
                    runner.name = fast_sc.valid ? string(fast_sc.name) : fused_acl_name(current_op);
                    runner.jt_name = cname;
                    runner.scalar_data = scalar_src->ptr<void>();
                    runner.scalar_dtype = scalar_src->output->dtype();
                    runner.add(tensor_arg, true);
                    runner.add(bop->z, false);
                    runner.run_cached(fast_sc.iter);
                    return;
                }
            }
            Var *lx = bop->x;
            Var *ly = bop->y;
            if (scalar_alias && !scalar_alias->empty())
            {
                if (Var *rx = scalar_alias->find_or_null(lx)) lx = rx;
                if (Var *ry = scalar_alias->find_or_null(ly)) ly = ry;
            }
            const FastAclOpEntry *fast = fast_acl_entry_for_op(current_op, cname);
            if (fast)
            {
                AclCachedEntryRunner<BinaryOpRunner> op;
                op.add(lx, true);
                op.add(ly, true);
                op.add(bop->z, false);
                op.name = fused_acl_name(current_op);
                op.jt_name = cname;
                op.run_cached(fast->iter);
            }
            else
            {
                AclExecutionRunner<BinaryOpRunner> runner;
                runner.add(lx, true);
                runner.add(ly, true);
                runner.add(bop->z, false);
                runner.name = fused_acl_name(current_op);
                runner.jt_name = cname;
                runner.run();
            }
        }
        else if (std::strcmp(cname, "ternary") == 0)
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
        else if (std::strcmp(cname, "array") == 0)
        {
            auto aop = (ArrayOp *)current_op;
            // The fused allocator can reuse a consumed constant's
            // buffer before earlier ACL kernels finish. Queue the H2D
            // copy on aclstream so reuse stays ordered with consumers.
            const void *source = aop->ptr<void>();
            // ArrayOp force-fuses every one-element tensor, including
            // rank-1 integer indices such as chunk's [i]. Their host
            // allocation can be recycled before this queued copy runs.
            if (aop->output->num == 1)
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
        else if (std::strcmp(cname, "reduce") == 0)
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

            op.use_inline_attr = true;
            op.inline_axes_len = 0;
            for (int i = 0; i < rop->x->shape.size(); i++)
                if (rop->reduce_mask & (1 << i))
                    op.inline_axes[op.inline_axes_len++] = i;
            op.inline_keepdims = (rop->x->shape.size() == rop->y->shape.size());

            op.add(rop->y, false);
            op.run();
        }
        else if (std::strcmp(cname, "broadcast_to") == 0)
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
                    xshape.push_back(1);
                else
                    xshape.push_back(zshape[i]);
            }
            bop->x->shape = xshape;
            op.add(bop->x, true);
            op.add(bop->z, false);
            op.run();
            bop->x->shape = xshape_bk;
        }
        else if (std::strcmp(cname, "fuse_transpose") == 0)
        {
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
            LOGf << "op " << cname << " not supported";
        }
    }

    static bool is_host_scalar_binary_consumer(BinaryOp *bin, Var *scalar_var, NanoString scalar_dt)
    {
        const bool right = (bin->y == scalar_var);
        const bool left = (bin->x == scalar_var);
        if (left == right) return false;
        const bool ok_op = (bin->ns == ns_add || bin->ns == ns_multiply)
            || (right && (bin->ns == ns_subtract || bin->ns == ns_divide));
        if (!ok_op) return false;
        Var *other = left ? bin->y : bin->x;
        if (other->dtype() != scalar_dt || bin->z->dtype() != scalar_dt
            || !(other->shape == bin->z->shape) || other->shape.size() == 0)
            return false;
        // Reject if `other` could itself be elided as a scalar array/broadcast.
        if (Op *op = other->input())
        {
            const char *oname = op->name();
            if (std::strcmp(oname, "array") == 0) return false;
            if (std::strcmp(oname, "broadcast_to") == 0
                && static_cast<BroadcastToOp *>(op)->x->num == 1)
                return false;
        }
        return true;
    }

    static bool can_elide_scalar_array(
        ArrayOp *aop,
        const vector<Op *> &ops,
        const SmallVarConsumers &consumers)
    {
        Var *out = aop->output;
        if (out->mem_ptr || out->num != 1)
            return false;
        const NanoString dt = out->dtype();
        if (dt != ns_float32 && dt != ns_float16 && dt != ns_bfloat16
            && dt != ns_int32 && dt != ns_int64)
            return false;
        const auto *clist = consumers.get(out);
        if (!clist || clist->empty())
            return false;
        for (size_t idx : *clist)
        {
            Op *c = ops[idx];
            if (std::strcmp(c->name(), "binary") == 0)
            {
                if (!is_host_scalar_binary_consumer(static_cast<BinaryOp *>(c), out, dt))
                    return false;
            }
            else if (std::strcmp(c->name(), "broadcast_to") == 0)
            {
                auto *bop = static_cast<BroadcastToOp *>(c);
                if (bop->z->mem_ptr) return false;
                const auto *blist = consumers.get(bop->z);
                if (!blist || blist->empty()) return false;
                for (size_t bidx : *blist)
                {
                    Op *bc = ops[bidx];
                    if (std::strcmp(bc->name(), "binary") != 0
                        || !is_host_scalar_binary_consumer(static_cast<BinaryOp *>(bc), bop->z, dt))
                        return false;
                }
            }
            else
            {
                return false;
            }
        }
        return true;
    }

    static bool can_elide_scalar_broadcast(
        BroadcastToOp *bop,
        const vector<Op *> &ops,
        const SmallVarConsumers &consumers,
        const SmallVarMap &scalar_alias)
    {
        if (bop->z->mem_ptr || bop->x->num != 1 || !bop->x->is_contiguous())
            return false;
        const auto *clist = consumers.get(bop->z);
        if (!clist || clist->empty())
            return false;
        for (size_t idx : *clist)
        {
            Op *consumer = ops[idx];
            if (std::strcmp(consumer->name(), "binary") != 0)
                return false;
            auto *bin = static_cast<BinaryOp *>(consumer);
            Var *other = (bin->x == bop->z) ? bin->y : bin->x;
            if (other == bop->z || scalar_alias.count(other)
                || !(other->shape == bin->z->shape))
                return false;
        }
        return true;
    }

    static void exec_acl_sequence(Op *op, const vector<Op *> &ops)
    {
        const size_t n = ops.size();
        AclSeqScratchLease scratch_lease;
        auto &scratch = *scratch_lease;
        if (n == 1)
        {
            auto &temp_allocs = scratch.temp_allocs;
            temp_allocs.clear();
            dispatch_acl_checked(fused_acl_unsupported(ops), [&]
            {
                Op *current_op = ops[0];
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
                    temp_allocs.push_back(out);
                    INTERNAL_ASSERT(out->alloc(runtime_executor().allocator))
                        << "ACL fused output allocation returned no storage";
                }
                exec_acl_primitive(current_op);
                for (auto *value : temp_allocs)
                    free_var_mem(value);
                temp_allocs.clear();
            }, [&](const string &reason)
            {
                fallback_cpu(op, reason);
            }, [&]
            {
                auto status = aclrtSynchronizeStream(aclstream);
                if (status != ACL_SUCCESS)
                    std::fprintf(stderr, "ACL failure cleanup drain returned %d\n", int(status));
                for (auto *value : temp_allocs)
                {
                    try { if (value->allocator) free_var_mem(value); }
                    catch (...) { std::fprintf(stderr, "ACL temporary cleanup failed\n"); }
                }
            });
            return;
        }

        // Reuse per-thread flat scratch buffers across fused ACL launches
        // (zero heap allocations on steady-state fused groups).
        auto &op_indeg = scratch.op_indeg;
        op_indeg.assign(n, 0);
        auto &adj = scratch.adj;
        if (adj.size() < n) adj.resize(n);
        for (size_t i = 0; i < n; ++i) adj[i].clear();
        auto &consumers = scratch.consumers;
        consumers.reset();
        for (size_t i = 0; i < n; ++i)
        {
            for (auto *in : ops[i]->inputs())
                consumers.add(in, i);
        }
        auto &var_outdeg = scratch.var_outdeg;
        var_outdeg.clear();
        for (size_t u = 0; u < n; ++u)
        {
            for (auto *out : ops[u]->outputs())
            {
                const auto *dsts = consumers.get(out);
                if (dsts && !dsts->empty())
                {
                    var_outdeg.get_or_zero(out) += static_cast<int>(dsts->size());
                    auto &edges = adj[u];
                    for (size_t v : *dsts)
                    {
                        ++op_indeg[v];
                        edges.push_back(v);
                    }
                }
            }
        }
        struct ReadyQueue
        {
            vector<std::pair<Op *, size_t>> &items;
            size_t head = 0;
            bool empty() const { return head == items.size(); }
            Op *front() const { return items[head].first; }
            size_t front_index() const { return items[head].second; }
            void push(Op *p, size_t idx) { items.emplace_back(p, idx); }
            void pop() { ++head; }
        } queue{scratch.queue_items};
        queue.items.clear();
        for (size_t i = 0; i < n; ++i)
            if (op_indeg[i] == 0)
                queue.push(ops[i], i);

        auto &new_alloced = scratch.new_alloced;
        new_alloced.clear();
        auto &scalar_alias = scratch.scalar_alias;
        scalar_alias.clear();
        auto &host_scalar_alias = scratch.host_scalar_alias;
        host_scalar_alias.clear();
        size_t total = 0;
        dispatch_acl_checked(fused_acl_unsupported(ops), [&]
        {
            while (!queue.empty())
            {
                const size_t u = queue.front_index();
                auto *current_op = queue.front();
                queue.pop();
                ++total;
                for (auto in : current_op->inputs())
                {
                    if (!host_scalar_alias.empty() && host_scalar_alias.count(in))
                        continue;
                    Var *actual = in;
                    if (!scalar_alias.empty())
                    {
                        if (Var *rep = scalar_alias.find_or_null(in)) actual = rep;
                    }
                    ASSERT(actual->mem_ptr)
                        << "current fused operator input is not allocated:"
                        << current_op->name() << actual;
                }
                bool elide = false;
                if (std::strcmp(current_op->name(), "array") == 0
                    && can_elide_scalar_array(
                        static_cast<ArrayOp *>(current_op), ops, consumers))
                {
                    auto *aop = static_cast<ArrayOp *>(current_op);
                    host_scalar_alias.set(aop->output, aop);
                    elide = true;
                }
                else if (std::strcmp(current_op->name(), "broadcast_to") == 0)
                {
                    auto *bop = static_cast<BroadcastToOp *>(current_op);
                    if (ArrayOp *hsrc = host_scalar_alias.find_or_null(bop->x))
                    {
                        if (!bop->z->mem_ptr)
                        {
                            host_scalar_alias.set(bop->z, hsrc);
                            elide = true;
                        }
                    }
                    if (!elide && can_elide_scalar_broadcast(bop, ops, consumers, scalar_alias))
                    {
                        scalar_alias.set(bop->z, bop->x);
                        var_outdeg.get_or_zero(bop->x) += var_outdeg.get_or_zero(bop->z);
                        elide = true;
                    }
                }
                if (!elide)
                {
                    for (auto out : current_op->outputs())
                    {
                        if (out->mem_ptr)
                            continue;
                        new_alloced.push_back(out);
                        INTERNAL_ASSERT(out->alloc(runtime_executor().allocator))
                            << "ACL fused output allocation returned no storage";
                    }
                }
                for (size_t v : adj[u])
                {
                    if (--op_indeg[v] == 0)
                        queue.push(ops[v], v);
                }
                if (!elide)
                    exec_acl_primitive(current_op, &scalar_alias, &host_scalar_alias);
                for (auto *in : current_op->inputs())
                {
                    if (!host_scalar_alias.empty() && host_scalar_alias.count(in))
                        continue;
                    Var *actual = in;
                    if (!scalar_alias.empty())
                    {
                        if (Var *rep = scalar_alias.find_or_null(in)) actual = rep;
                    }
                    if (int *deg = var_outdeg.find_ptr(actual))
                    {
                        if (--(*deg) == 0)
                        {
                            auto alloc_it = std::find(new_alloced.begin(), new_alloced.end(), actual);
                            if (alloc_it != new_alloced.end())
                            {
                                *alloc_it = new_alloced.back();
                                new_alloced.pop_back();
                                free_var_mem(actual);
                            }
                        }
                    }
                }
            }
            INTERNAL_ASSERT(total == n) << "ACL fused graph has unresolved dependencies";
            for (auto *value : new_alloced)
                free_var_mem(value);
            new_alloced.clear();
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
        LOGv << "compile" << op;
        auto *fused = static_cast<FusedOp *>(op);
        // One generated AscendC kernel for the whole group when it is a
        // float32 elementwise chain this backend can reproduce exactly;
        // otherwise the per-node launcher sequence below, unchanged.
        if (exec_fused_ascendc(fused)) return;
        exec_acl_sequence(op, fused->ops);
    }

    static void exec_single_acl(Op *op)
    {
        LOGv << "exec acl op " << op->name_ex() << op;
        exec_acl_sequence(op, {op});
    }

    static void exec_acl_random(Op *op)
    {
        auto _op = (RandomOp *)op;
        AclExecutionRunner<RandomOpRunner> runner(
            _op->type == ns_uniform ? "RandomUniform" : "RandomNormal");
        auto out = op->output(0);
        const auto rng = reserve_acl_random(out->device_id, out->numel());
        RandomAttr *attr = new RandomAttr();
        attr->seed = rng.seed;
        attr->offset = rng.offset;
        runner.jt_name = "random";
        runner.op_attr.reset(attr);

        runner.add(out, false);
        runner.run();
    }

    static unordered_map<string, std::function<void(Op *)>> acl_ops = {
        {"getitem", exec_native_acl_getitem},
        {"setitem", exec_native_acl_setitem},
        // jittor's own transpose op is one aclnnPermute -- exactly what the ACL
        // CodeOp override used to build, but through jt.code: several us of
        // python plus a 26-entry string->double attribute map per node, against
        // ~1 us here. TransposeOpRunner reads both extra fields of ReduceAttr,
        // so both are set even though a permute uses neither.
        {"transpose", [](Op *op)
         {
             auto *_op = (TransposeOp *)op;
             AclExecutionRunner<TransposeOpRunner> runner;
             ReduceAttr *attr = new ReduceAttr();
             attr->axes.reserve(_op->axes.size());
             for (int i = 0; i < _op->axes.size(); ++i)
                 attr->axes.push_back(_op->axes[i]);
             attr->prod_dim = 0;
             attr->keepdims = false;
             runner.jt_name = "transpose";
             runner.op_attr.reset(attr);
             runner.add(_op->x, true);
             runner.add(_op->y, false);
             runner.run();
         }},
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
        {"fused_sgd", [](Op *op)
         {
             auto _op = (FusedSgdOp *)op;
             AclExecutionRunner<FusedSgdOpRunner, false> runner;
             FusedSgdAttr *attr = new FusedSgdAttr();
             attr->tensorCount = _op->parameters.size();
             attr->lr = _op->lr;
             attr->momentum = _op->momentum;
             attr->weightDecay = _op->weight_decay;
             attr->dampening = _op->dampening;
             attr->nesterov = _op->nesterov;
             attr->maximize = _op->maximize;
             // jittor-core-gaps.md §3.3: the velocity buffer is zero-
             // initialized at optimizer construction and only holds real
             // running momentum from its SECOND touch onward -- the caller
             // (fused_sgd_acl, from SGD.step()'s per-group step counter)
             // says whether this call is the first, and the kernel below
             // must seed from the raw gradient then, not fold a dampened
             // recurrence into the zero buffer.
             attr->isFirstStep = _op->first_step;
             runner.jt_name = "fused_sgd";
             runner.op_attr.reset(attr);
             for (auto value : _op->parameters) runner.add(value, true);
             for (auto value : _op->velocities) runner.add(value, true);
             for (auto value : _op->gradients) runner.add(value, true);
             for (auto value : _op->new_parameters) runner.add(value, false);
             for (auto value : _op->new_velocities) runner.add(value, false);
             runner.run();
         }},
        {"mapped_matmul", [](Op *op)
         {
             auto *_op = (MappedMatmulOp *)op;
             // aclnn's cubeMathType: 0 = KEEP_DTYPE (true float32), 1 =
             // ALLOW_FP32_DOWN_PRECISION (HF32), the arithmetic torch_npu uses
             // by default and what `jt.acl_allow_hf32` selects.
             const int cube_math_type = _op->allow_reduced_precision ? 1 : 0;
             // aclnnMatmul takes the plain matrices; aclnnBatchMatMul takes a
             // rank-3 stack, and BatchMatMulOpRunner folds any further leading
             // axes into its descriptor, so attention's rank-4 operands arrive
             // here as one node and leave as one launch. Both runners call
             // their aclnn entry point directly and never read the registry
             // iterator, hence UsesRegistry=false.
             if (_op->a->shape.size() == 2)
             {
                 AclExecutionRunner<MatMulOpRunner, false> runner;
                 runner.jt_name = _op->trans_b ? "matmul_trans_1"
                     : _op->trans_a ? "matmul_trans_0" : "matmul";
                 runner.cube_math_type = cube_math_type;
                 runner.add(_op->a, true);
                 runner.add(_op->b, true);
                 runner.add(_op->c, false);
                 runner.run();
             }
             else
             {
                 AclExecutionRunner<BatchMatMulOpRunner, false> runner;
                 runner.jt_name = _op->trans_b ? "bmm_trans_1"
                     : _op->trans_a ? "bmm_trans_0" : "bmm";
                 runner.cube_math_type = cube_math_type;
                 runner.add(_op->a, true);
                 runner.add(_op->b, true);
                 runner.add(_op->c, false);
                 runner.run();
             }
         }},
        {"mapped_rms_norm", [](Op *op)
         {
             auto *_op = static_cast<MappedRmsNormOp *>(op);
             AclExecutionRunner<RmsNormOpRunner, false> runner;
             runner.jt_name = "rmsnorm";
             auto attr = std::unique_ptr<RmsNormAttr>(new RmsNormAttr());
             // Preserve the existing CodeOp bridge's float epsilon rounding.
             attr->eps = _op->eps;
             runner.op_attr = std::move(attr);
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->y, false);
             auto *scratch = _op->rstd_scratch.ptr;
             scratch->device_id = _op->y->device_id;
             scratch->placement = _op->y->placement;
             // The op owns this allocation for its whole lifetime, including
             // device-graph replay. It must not come from a per-launch pool.
             INTERNAL_ASSERT(scratch->alloc(runtime_executor().allocator))
                 << "Unable to allocate RMSNorm auxiliary scratch";
             runner.add(scratch, false);
             runner.run();
         }},
        {"mapped_unary", [](Op *op)
         {
             auto *_op = static_cast<MappedUnaryOp *>(op);
             if (!_op->y->num) return;
             if (_op->kind == MappedUnaryOp::kRsqrt)
             {
                 AclExecutionRunner<RsqrtOpRunner, false> runner;
                 runner.jt_name = "rsqrt";
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kSilu)
             {
                 if (_op->x->dtype() == ns_bfloat16)
                 {
                     AclExecutionRunner<SwishOpRunner, false> runner;
                     runner.jt_name = "swish";
                     runner.add(_op->x, true);
                     runner.add(_op->y, false);
                     runner.run();
                 }
                 else
                 {
                     AclExecutionRunner<SiLUOpRunner, false> runner;
                     runner.jt_name = "silu";
                     runner.add(_op->x, true);
                     runner.add(_op->y, false);
                     runner.run();
                 }
             }
             else if (_op->kind == MappedUnaryOp::kSoftplus)
             {
                 AclExecutionRunner<SoftplusOpRunner, false> runner;
                 runner.jt_name = "softplus";
                 auto attr = std::unique_ptr<SoftplusAttr>(new SoftplusAttr());
                 attr->beta = _op->param0;
                 attr->threshold = _op->param1;
                 runner.op_attr = std::move(attr);
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kTriu)
             {
                 AclExecutionRunner<TriuOpRunner, false> runner;
                 runner.jt_name = "triu";
                 auto attr = std::unique_ptr<TriuAttr>(new TriuAttr());
                 attr->diagonal = static_cast<int64_t>(_op->param0);
                 runner.op_attr = std::move(attr);
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kTril)
             {
                 AclExecutionRunner<TrilOpRunner, false> runner;
                 runner.jt_name = "tril";
                 auto attr = std::unique_ptr<TriuAttr>(new TriuAttr());
                 attr->diagonal = static_cast<int64_t>(_op->param0);
                 runner.op_attr = std::move(attr);
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kCumsum)
             {
                 AclExecutionRunner<CumsumOpRunner, false> runner;
                 runner.jt_name = "cumsum";
                 auto attr = std::unique_ptr<GatherAttr>(new GatherAttr());
                 attr->dim = static_cast<int64_t>(_op->param0);
                 runner.op_attr = std::move(attr);
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kSoftmax)
             {
                 AclExecutionRunner<SoftmaxOpRunner, false> runner;
                 runner.jt_name = "softmax";
                 auto attr = std::unique_ptr<SoftmaxAttr>(new SoftmaxAttr());
                 attr->dim = static_cast<int64_t>(_op->param0);
                 runner.op_attr = std::move(attr);
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kGelu)
             {
                 AclExecutionRunner<GeluOpRunner, false> runner;
                 runner.jt_name = "gelu";
                 auto attr = std::unique_ptr<GeluAttr>(new GeluAttr());
                 attr->approximate = static_cast<int64_t>(_op->param0);
                 runner.op_attr = std::move(attr);
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else if (_op->kind == MappedUnaryOp::kRelu)
             {
                 AclExecutionRunner<UnaryOpRunner> runner;
                 runner.name = "ReLU";
                 runner.jt_name = "relu";
                 runner.add(_op->x, true);
                 runner.add(_op->y, false);
                 runner.run();
             }
             else
             {
                 LOGf << "mapped_unary: unsupported kind " << _op->kind;
             }
         }},
        {"mapped_grouped_matmul", [](Op *op)
         {
             auto *_op = static_cast<MappedGroupedMatmulOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<GroupedMatMulOpRunner, false> runner;
             runner.jt_name = "mapped_grouped_matmul";
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->offs, true);
             runner.add(_op->y, false);
             runner.run();
         }},
        {"mapped_histc", [](Op *op)
         {
             auto *_op = static_cast<MappedHistcOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<HistcOpRunner, false> runner;
             runner.jt_name = "mapped_histc";
             runner.bins = _op->bins;
             runner.min_val = _op->min_val;
             runner.max_val = _op->max_val;
             runner.add(_op->x, true);
             runner.add(_op->y, false);
             runner.run();
         }},
        {"mapped_addmm", [](Op *op)
         {
             auto *_op = static_cast<MappedAddmmOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<AddmmOpRunner, false> runner;
             runner.jt_name = "mapped_addmm";
             runner.cube_math_type = _op->allow_reduced_precision ? 1 : 0;
             runner.add(_op->bias, true);
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->y, false);
             runner.run();
         }},
        {"mapped_flash_attention", [](Op *op)
         {
             auto *_op = static_cast<MappedFlashAttentionOp *>(op);
             if (!_op->out->num) return;
             AclExecutionRunner<DirectFlashAttentionOpRunner, false> runner;
             runner.jt_name = "mapped_flash_attention";
             runner.scale = _op->scale;
             runner.head_num = _op->q->shape[1];
             runner.sparse_mode = _op->sparse_mode;
             runner.has_mask = _op->has_mask;
             runner.mask_is_realshift = _op->mask_is_realshift;
             runner.add(_op->q, true);
             runner.add(_op->k, true);
             runner.add(_op->v, true);
             if (_op->has_mask)
                 runner.add(_op->atten_mask, true);
             auto *smax = _op->softmax_max_scratch.ptr;
             auto *ssum = _op->softmax_sum_scratch.ptr;
             smax->device_id = _op->out->device_id;
             smax->placement = _op->out->placement;
             ssum->device_id = _op->out->device_id;
             ssum->placement = _op->out->placement;
             INTERNAL_ASSERT(smax->alloc(runtime_executor().allocator))
                 << "Unable to allocate FlashAttention softmax_max scratch";
             INTERNAL_ASSERT(ssum->alloc(runtime_executor().allocator))
                 << "Unable to allocate FlashAttention softmax_sum scratch";
             runner.add(smax, false);
             runner.add(ssum, false);
             runner.add(_op->out, false);
             runner.run();
         }},
        {"mapped_conv2d", [](Op *op)
         {
             auto *_op = static_cast<MappedConv2dOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<DirectConv2dOpRunner, false> runner;
             runner.jt_name = "mapped_conv2d";
             runner.stride_h = _op->stride_h;
             runner.stride_w = _op->stride_w;
             runner.pad_h = _op->pad_h;
             runner.pad_w = _op->pad_w;
             runner.dil_h = _op->dil_h;
             runner.dil_w = _op->dil_w;
             runner.groups = _op->groups;
             runner.cube_math_type = _op->cube_math_type;
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             if (_op->bias)
                 runner.add(_op->bias, true);
             runner.add(_op->y, false);
             runner.run();
         }},
        {"mapped_conv2d_backward", [](Op *op)
         {
             auto *_op = static_cast<MappedConv2dBackwardOp *>(op);
             if (!_op->dout->num) return;
             AclExecutionRunner<DirectConv2dBackwardOpRunner, false> runner;
             runner.jt_name = "mapped_conv2d_backward";
             runner.stride_h = _op->stride_h;
             runner.stride_w = _op->stride_w;
             runner.pad_h = _op->pad_h;
             runner.pad_w = _op->pad_w;
             runner.dil_h = _op->dil_h;
             runner.dil_w = _op->dil_w;
             runner.groups = _op->groups;
             runner.cube_math_type = _op->cube_math_type;
             runner.add(_op->dout, true);
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->dx, false);
             runner.add(_op->dw, false);
             if (_op->db)
                 runner.add(_op->db, false);
             runner.run();
         }},
        {"mapped_batch_norm", [](Op *op)
         {
             auto *_op = static_cast<MappedBatchNormOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<DirectBatchNormOpRunner, false> runner;
             runner.jt_name = "mapped_batch_norm";
             runner.is_train = _op->is_train;
             runner.momentum = _op->momentum;
             runner.eps = _op->eps;
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->bias, true);
             runner.add(_op->running_mean, true);
             runner.add(_op->running_var, true);
             runner.add(_op->y, false);
             Var *sm = _op->save_mean;
             Var *si = _op->save_invstd;
             if (!sm)
             {
                 sm = _op->save_mean_scratch.ptr;
                 si = _op->save_invstd_scratch.ptr;
                 sm->device_id = _op->y->device_id;
                 sm->placement = _op->y->placement;
                 si->device_id = _op->y->device_id;
                 si->placement = _op->y->placement;
                 INTERNAL_ASSERT(sm->alloc(runtime_executor().allocator))
                     << "Unable to allocate BatchNorm save_mean scratch";
                 INTERNAL_ASSERT(si->alloc(runtime_executor().allocator))
                     << "Unable to allocate BatchNorm save_invstd scratch";
             }
             runner.add(sm, false);
             runner.add(si, false);
             if (_op->return_running_stats)
             {
                 runner.add(_op->new_running_mean, false);
                 runner.add(_op->new_running_var, false);
             }
             runner.run();
         }},
        {"mapped_batch_norm_backward", [](Op *op)
         {
             auto *_op = static_cast<MappedBatchNormBackwardOp *>(op);
             if (!_op->dx->num) return;
             AclExecutionRunner<DirectBatchNormBackwardOpRunner, false> runner;
             runner.jt_name = "mapped_batch_norm_backward";
             runner.is_train = _op->is_train;
             runner.eps = _op->eps;
             runner.add(_op->dout, true);
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->running_mean, true);
             runner.add(_op->running_var, true);
             runner.add(_op->save_mean, true);
             runner.add(_op->save_invstd, true);
             runner.add(_op->dx, false);
             runner.add(_op->dw, false);
             runner.add(_op->db, false);
             runner.run();
         }},
        {"mapped_layer_norm", [](Op *op)
         {
             auto *_op = static_cast<MappedLayerNormOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<DirectLayerNormOpRunner, false> runner;
             runner.jt_name = "mapped_layer_norm";
             runner.normalized_ndim = _op->normalized_ndim;
             runner.eps = _op->eps;
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->bias, true);
             runner.add(_op->y, false);
             Var *mn = _op->mean;
             Var *rs = _op->rstd;
             if (!mn)
             {
                 mn = _op->mean_scratch.ptr;
                 rs = _op->rstd_scratch.ptr;
                 mn->device_id = _op->y->device_id;
                 mn->placement = _op->y->placement;
                 rs->device_id = _op->y->device_id;
                 rs->placement = _op->y->placement;
                 INTERNAL_ASSERT(mn->alloc(runtime_executor().allocator))
                     << "Unable to allocate LayerNorm mean scratch";
                 INTERNAL_ASSERT(rs->alloc(runtime_executor().allocator))
                     << "Unable to allocate LayerNorm rstd scratch";
             }
             runner.add(mn, false);
             runner.add(rs, false);
             runner.run();
         }},
        {"mapped_layer_norm_backward", [](Op *op)
         {
             auto *_op = static_cast<MappedLayerNormBackwardOp *>(op);
             if (!_op->dx->num) return;
             AclExecutionRunner<DirectLayerNormBackwardOpRunner, false> runner;
             runner.jt_name = "mapped_layer_norm_backward";
             runner.normalized_ndim = _op->normalized_ndim;
             runner.add(_op->dout, true);
             runner.add(_op->x, true);
             runner.add(_op->mean, true);
             runner.add(_op->rstd, true);
             runner.add(_op->weight, true);
             runner.add(_op->bias, true);
             runner.add(_op->dx, false);
             runner.add(_op->dw, false);
             runner.add(_op->db, false);
             runner.run();
         }},
        {"mapped_unary_grad", [](Op *op)
         {
             auto *_op = static_cast<MappedUnaryGradOp *>(op);
             if (!_op->dx->num) return;
             AclExecutionRunner<DirectUnaryGradOpRunner, false> runner;
             runner.jt_name = _op->kind == MappedUnaryOp::kRelu ? "mapped_relu_grad" : "mapped_gelu_grad";
             runner.name = _op->kind == MappedUnaryOp::kRelu ? "LeakyReLUBackward" : "GeluBackward";
             runner.kind = _op->kind;
             runner.approximate = static_cast<int64_t>(_op->param0);
             runner.add(_op->dout, true);
             runner.add(_op->x, true);
             runner.add(_op->dx, false);
             runner.run();
         }},
        {"mapped_fp8_linear", [](Op *op)
         {
             auto *_op = static_cast<MappedFp8LinearOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<Fp8LinearOpRunner, false> runner;
             runner.jt_name = "mapped_fp8_linear";
             runner.mode = _op->mode;
             runner.block_size = _op->block_size;
             auto *qx = _op->qx_scratch.ptr;
             auto *sx = _op->sx_scratch.ptr;
             qx->device_id = _op->y->device_id;
             qx->placement = _op->y->placement;
             sx->device_id = _op->y->device_id;
             sx->placement = _op->y->placement;
             INTERNAL_ASSERT(qx->alloc(runtime_executor().allocator))
                 << "Unable to allocate Fp8Linear quantized activation scratch";
             INTERNAL_ASSERT(sx->alloc(runtime_executor().allocator))
                 << "Unable to allocate Fp8Linear scale scratch";
             runner.qx_scratch = qx;
             runner.sx_scratch = sx;
             runner.add(_op->x, true);
             runner.add(_op->weight, true);
             runner.add(_op->weight_scale, true);
             runner.add(_op->y, false);
             runner.run();
         }},
        {"mapped_packed_matmul", [](Op *op)
         {
             auto *_op = static_cast<MappedPackedMatmulOp *>(op);
             // Empty leading dimensions still have a valid ND output shape,
             // but CANN need not construct or launch a zero-row matrix product.
             if (!_op->c->num) return;
             AclExecutionRunner<PackedMatMulOpRunner, false> runner;
             runner.jt_name = "mapped_packed_matmul";
             runner.add(_op->a, true);
             runner.add(_op->b, true);
             runner.add(_op->c, false);
             runner.run();
         }},
        {"mapped_concat", [](Op *op)
         {
             auto *_op = static_cast<MappedConcatOp *>(op);
             if (!_op->y->num) return;
             AclExecutionRunner<ConcatOpRunner, false> runner;
             auto *attr = new ConcatAttr();
             attr->dim = _op->dim;
             attr->tensorNum = 0;
             runner.jt_name = "mapped_concat";
             runner.op_attr.reset(attr);
             // Empty extents contribute no elements. Omitting them also keeps
             // the vendor descriptor list valid when only one input is nonempty.
             for (auto *value : _op->tensors) {
                 if (!value->num) continue;
                 runner.add(value, true);
                 ++attr->tensorNum;
             }
             runner.add(_op->y, false);
             runner.run();
         }},
        {"argsort", [](Op *op)
         {
             // CANN Sort writes (values, indices); ArgsortOp holds the
             // indices in y and the sorted values in y_key.
             auto *_op = static_cast<ArgsortOp *>(op);
             AclExecutionRunner<SortOpRunner, false> runner(false, _op->dim, _op->descending);
             runner.jt_name = "argsort";
             runner.add(_op->x, true);
             runner.add(_op->y_key, false);
             runner.add(_op->y, false);
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
        {"curand_random", exec_acl_random},
        // `curand_random` is the accelerator capability op the CUDA-family
        // backends publish. ACL has no separate capability op, so the core
        // `random` op reaches the same launcher under its own name.
        {"random", exec_acl_random},
        {"write_back", [](Op *op)
         {
             // Each entry is a plain device-to-device copy: `values[i]` into
             // the storage `written[i]` shares with `targets[i]` (set up in
             // WriteBackOp::infer_shape, backend-agnostic). CUDA batches all
             // entries into one fused kernel; ACL has no such primitive, so
             // each entry is its own aclrtMemcpyAsync on the current ACL
             // stream -- ordered with the rest of a captured step the same
             // way every other op here is. dtype/shape were already
             // USER_CHECKed at construction.
             auto *_op = (WriteBackOp *)op;
             for (uint i = 0; i < _op->targets.size(); ++i)
             {
                 const int64 bytes = _op->targets[i]->size;
                 if (!bytes || _op->written[i]->mem_ptr == _op->values[i]->mem_ptr)
                     continue;
                 auto ret = aclrtMemcpyAsync(
                     _op->written[i]->mem_ptr, bytes,
                     _op->values[i]->mem_ptr, bytes,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                 if (ret != ACL_SUCCESS)
                     throw std::runtime_error(
                         "aclrtMemcpyAsync failed: " +
                         acl_error_to_string(ret));
             }
         }},
    };

    static bool is_acl_random(const char *name)
    {
        return std::strcmp(name, "random") == 0 || std::strcmp(name, "curand_random") == 0;
    }

    static string mapped_acl_dtype_unsupported(Op *op, bool allow_mx_scale = false)
    {
        for (auto *input : op->inputs())
            if (PREDICT_BRANCH_NOT_TAKEN(!acl_has_dtype(input->dtype())
                && !(allow_mx_scale && input->dtype() == ns_float8_e8m0fnu)))
                return string(op->name()) + " does not support input dtype " + string(input->dtype().to_cstring());
        for (auto *output : op->outputs())
            if (PREDICT_BRANCH_NOT_TAKEN(!acl_has_dtype(output->dtype())))
                return string(op->name()) + " does not support output dtype " + string(output->dtype().to_cstring());
        return {};
    }

    static void exec_mapped_acl_ops(Op *op)
    {
        const char *cname = op->name();
        auto iter = acl_ops.find(cname);
        string unsupported;
        if (iter == acl_ops.end())
            unsupported = string("no registered ACL implementation for ") + cname;
        else
        {
            INTERNAL_ASSERT(iter->second) << "Empty ACL implementation for" << cname;
            const bool allow_mx_scale =
                (std::strcmp(cname, "mapped_fp8_linear") == 0 && acl_soc_supports_fp8());
            unsupported = mapped_acl_dtype_unsupported(op, allow_mx_scale);
            if (unsupported.empty() && std::strcmp(cname, "getitem") == 0)
                unsupported = acl_getitem_unsupported_reason(op);
            if (unsupported.empty() && std::strcmp(cname, "setitem") == 0)
                unsupported = acl_setitem_unsupported_reason(op);
            if (std::strcmp(cname, "argsort") == 0)
            {
                auto *sort = static_cast<ArgsortOp *>(op);
                const auto dtype = sort->x->dtype();
                if (dtype != ns_float16 && dtype != ns_float32 &&
                    dtype != ns_int8 && dtype != ns_int16 &&
                    dtype != ns_int32 && dtype != ns_int64 && dtype != ns_uint8)
                    unsupported = "argsort requires a CANN Sort input dtype";
                if (sort->y->dtype() != ns_int64)
                    unsupported = "argsort ACL indices must be int64 before output cast";
            }
            if (std::strcmp(cname, "arg_reduce") == 0)
            {
                auto *reduce = static_cast<ArgReduceOp *>(op);
                USER_CHECK(reduce->op == ns_maximum || reduce->op == ns_minimum)
                    << "arg_reduce requires min or max";
            }
            if (is_acl_random(cname))
            {
                auto *random = static_cast<RandomOp *>(op);
                USER_CHECK(random->type == ns_uniform || random->type == ns_normal)
                    << "random requires uniform or normal";
                const auto dtype = random->output->dtype();
                if (dtype != ns_float16 && dtype != ns_bfloat16 && dtype != ns_float32)
                    unsupported = "random requires float16, bfloat16 or float32 output";
                const string rname = random->type == ns_uniform ? "RandomUniform" : "RandomNormal";
                if (!acl_op_registry().count(rname)) unsupported = "unregistered ACL launcher: " + rname;
            }
        }
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << cname << op;
            iter->second(op);
        }, [&](const string &reason)
        {
            fallback_cpu(op, reason);
        }, [&]
        {
            aclrtSynchronizeStream(aclstream);
        });
    }

    // Direct native entry points for hot ACL mapped operators: bypasses the
    // unordered_map<string, std::function> lookup and JK/jit_ops cache lock
    // on every launch while preserving identical preflight & fallback behavior.
    template<bool AllowMxScale>
    static void exec_acl_direct_fn(Op *op, void (*body)(Op *),
                                   string (*extra_check)(Op *) = nullptr)
    {
        string unsupported = mapped_acl_dtype_unsupported(op, AllowMxScale);
        if (unsupported.empty() && extra_check)
            unsupported = extra_check(op);
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            body(op);
        }, [&](const string &reason)
        {
            fallback_cpu(op, reason);
        }, [&]
        {
            aclrtSynchronizeStream(aclstream);
        });
    }

    static void exec_acl_direct_fp8_linear(Op *op)
    {
        static const auto &fn = acl_ops.at("mapped_fp8_linear");
        string unsupported = mapped_acl_dtype_unsupported(op, acl_soc_supports_fp8());
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            fn(op);
        },
            [&](const string &reason) { fallback_cpu(op, reason); },
            [&] { aclrtSynchronizeStream(aclstream); });
    }

    static void exec_acl_direct_matmul(Op *op)
    {
        static const auto &fn = acl_ops.at("mapped_matmul");
        string unsupported = mapped_acl_dtype_unsupported(op, false);
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            fn(op);
        },
            [&](const string &reason) { fallback_cpu(op, reason); },
            [&] { aclrtSynchronizeStream(aclstream); });
    }

    static void exec_acl_direct_unary(Op *op)
    {
        static const auto &fn = acl_ops.at("mapped_unary");
        string unsupported = mapped_acl_dtype_unsupported(op, false);
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            fn(op);
        },
            [&](const string &reason) { fallback_cpu(op, reason); },
            [&] { aclrtSynchronizeStream(aclstream); });
    }

    static void exec_acl_direct_concat(Op *op)
    {
        static const auto &fn = acl_ops.at("mapped_concat");
        string unsupported = mapped_acl_dtype_unsupported(op, false);
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            fn(op);
        },
            [&](const string &reason) { fallback_cpu(op, reason); },
            [&] { aclrtSynchronizeStream(aclstream); });
    }

    static void exec_acl_direct_transpose(Op *op)
    {
        static const auto &fn = acl_ops.at("transpose");
        string unsupported = mapped_acl_dtype_unsupported(op, false);
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            fn(op);
        },
            [&](const string &reason) { fallback_cpu(op, reason); },
            [&] { aclrtSynchronizeStream(aclstream); });
    }

    static void exec_acl_direct_write_back(Op *op)
    {
        static const auto &fn = acl_ops.at("write_back");
        string unsupported = mapped_acl_dtype_unsupported(op, false);
        dispatch_acl_checked(unsupported, [&]
        {
            LOGv << "exec acl op " << op->name() << op;
            fn(op);
        },
            [&](const string &reason) { fallback_cpu(op, reason); },
            [&] { aclrtSynchronizeStream(aclstream); });
    }

    static void exec_acl_direct_getitem(Op *op)
    {
        exec_acl_direct_fn<false>(op, exec_native_acl_getitem, acl_getitem_unsupported_reason);
    }

    static void exec_acl_direct_setitem(Op *op)
    {
        exec_acl_direct_fn<false>(op, exec_native_acl_setitem, acl_setitem_unsupported_reason);
    }

#define DEFINE_ACL_DIRECT_MAPPED(fn_name, op_key)                         \
    static void fn_name(Op *op)                                           \
    {                                                                     \
        static const auto &fn = acl_ops.at(op_key);                       \
        string unsupported = mapped_acl_dtype_unsupported(op, false);     \
        dispatch_acl_checked(unsupported, [&]                             \
        {                                                                 \
            LOGv << "exec acl op " << op->name() << op;                   \
            fn(op);                                                       \
        },                                                                \
            [&](const string &reason) { fallback_cpu(op, reason); },      \
            [&] { aclrtSynchronizeStream(aclstream); });                  \
    }

    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_addmm, "mapped_addmm")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_flash_attention, "mapped_flash_attention")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_rms_norm, "mapped_rms_norm")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_grouped_matmul, "mapped_grouped_matmul")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_histc, "mapped_histc")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_conv2d, "mapped_conv2d")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_conv2d_backward, "mapped_conv2d_backward")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_batch_norm, "mapped_batch_norm")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_batch_norm_backward, "mapped_batch_norm_backward")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_layer_norm, "mapped_layer_norm")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_layer_norm_backward, "mapped_layer_norm_backward")
    DEFINE_ACL_DIRECT_MAPPED(exec_acl_direct_unary_grad, "mapped_unary_grad")

#undef DEFINE_ACL_DIRECT_MAPPED

    DECLARE_FLAG(int, use_acl_ascendc_fusion);

    static void prepare_acl_fused_direct(Op *op, JK &jk)
    {
        if (PREDICT_BRANCH_NOT_TAKEN(profiler_enable || trace_py_var >= 2 || use_acl_ascendc_fusion))
            static_cast<FusedOp *>(op)->prepare_fused_key(jk);
    }

    static void prepare_acl_mapped_direct(Op *op, JK &jk)
    {
        if (PREDICT_BRANCH_NOT_TAKEN(profiler_enable || trace_py_var >= 2))
            prepare_registered_codegen(op, jk);
    }

    static void (*direct_acl_mapped_native(const string &name))(Op *)
    {
        if (name == "mapped_fp8_linear") return exec_acl_direct_fp8_linear;
        if (name == "mapped_matmul") return exec_acl_direct_matmul;
        if (name == "mapped_unary") return exec_acl_direct_unary;
        if (name == "mapped_unary_grad") return exec_acl_direct_unary_grad;
        if (name == "mapped_concat") return exec_acl_direct_concat;
        if (name == "mapped_addmm") return exec_acl_direct_addmm;
        if (name == "mapped_flash_attention") return exec_acl_direct_flash_attention;
        if (name == "mapped_rms_norm") return exec_acl_direct_rms_norm;
        if (name == "mapped_grouped_matmul") return exec_acl_direct_grouped_matmul;
        if (name == "mapped_histc") return exec_acl_direct_histc;
        if (name == "mapped_conv2d") return exec_acl_direct_conv2d;
        if (name == "mapped_conv2d_backward") return exec_acl_direct_conv2d_backward;
        if (name == "mapped_batch_norm") return exec_acl_direct_batch_norm;
        if (name == "mapped_batch_norm_backward") return exec_acl_direct_batch_norm_backward;
        if (name == "mapped_layer_norm") return exec_acl_direct_layer_norm;
        if (name == "mapped_layer_norm_backward") return exec_acl_direct_layer_norm_backward;
        if (name == "transpose") return exec_acl_direct_transpose;
        if (name == "write_back") return exec_acl_direct_write_back;
        if (name == "getitem") return exec_acl_direct_getitem;
        if (name == "setitem") return exec_acl_direct_setitem;
        return exec_mapped_acl_ops;
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
        // Diagnostic marker for an ACL CodeOp compile, as before the registered
        // execution refactor. Tests capture acl_op_exec.cc at verbosity 100 and
        // look for it; a source without it makes every such capture silent.
        LOGv << "compile acl op";
        return compile_registered_source(op);
    }

    static OpImplementation compose_acl_implementation(
        const OpDef &definition, const OpImplementation &original)
    {
        // The same launcher sequence handles fused graphs and a standalone
        // primitive. Unsupported implementations stay explicit fallback entries;
        // other backends' OpDefs and constructors are never removed.
        static const set<string> primitives = {
            "unary", "binary", "ternary", "broadcast_to", "fuse_transpose", "reduce", "contiguous"};
        auto implementation = original;
        const auto &name = definition.name;
        // Backend-native extensions such as HCCL declare their own compiler
        // through configure_accelerator_kernel at registration.
        if (implementation.kernel.compile) return implementation;
        if (name == "fused")
        {
            implementation.codegen.prepare = prepare_acl_fused_direct;
            implementation.kernel.native = exec_fused_acl;
            implementation.kernel.compile = compile_acl_fused;
        }
        else if (name == "code")
            implementation.kernel.compile = compile_acl_code;
        else if (acl_ops.count(name))
        {
            implementation.codegen.prepare = prepare_acl_mapped_direct;
            implementation.kernel.compile = compile_acl_mapped;
            implementation.kernel.native = direct_acl_mapped_native(name);
        }
        else if (primitives.count(name))
        {
            implementation.codegen.prepare = prepare_acl_mapped_direct;
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
        // Random's CPU OpDef deliberately does not advertise every accelerator;
        // the composer cannot create an entry missing from its backend mask.
        // Keep the generated jit callback: Op::run_jit uses it to enter the
        // provider's compiler even when native execution is also available.
        auto implementation = get_op_definition("random")->implementations.at(BackendId::Cpu);
        implementation.codegen.prepare = prepare_acl_mapped_direct;
        implementation.kernel.native = exec_mapped_acl_ops;
        implementation.kernel.compile = compile_acl_mapped;
        implementation.kernel.fallback_only = false;
        register_op_implementation("random", BackendId::Acl, implementation,
            "acl-native-random-v1");
    }

} // jittor
