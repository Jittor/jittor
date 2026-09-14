#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <queue>
#include <set>
#include "core/common.h"
#include "core/op.h"
#include "acl_jittor.h"
#include "ops/composite/random_op.h"
#include "ops/reduce_op.h"
#include "ops/binary_op.h"
#include "ops/broadcast_to_op.h"
#include "ops/composite/transpose_op.h"
#include "ops/composite/array_op.h"
#include "ops/composite/code_op.h"
#include "core/fused_op.h"
#include "ops/unary_op.h"
#include "ops/ternary_op.h"
#include "core/executor.h"
#include "runtime/device.h"
#include "mem/allocator.h"
#include "codegen/op_compiler.h"
#include "ops/op_register.h"
#include "codegen/opt/tuner_manager.h"
#include "utils/str_utils.h"
#include "aclnn/aclnn.h"
#include "binary_op_acl.h"
#include "base_op.h"

namespace jittor
{
    namespace
    {
        // Scratch blocks are leased, never shared: acquire pops one and the
        // runner's destructor pushes the same one back. A runner built inside
        // another runner's executeOp therefore holds a different block.
        constexpr size_t kMaxPooledScratch = 32;

        std::vector<AclRunnerScratch *> &scratch_pool()
        {
            static thread_local std::vector<AclRunnerScratch *> pool;
            return pool;
        }
    }

    AclRunnerScratch *acl_scratch_acquire()
    {
        auto &pool = scratch_pool();
        if (pool.empty())
            return new AclRunnerScratch();
        auto *scratch = pool.back();
        pool.pop_back();
        return scratch;
    }

    void acl_scratch_release(AclRunnerScratch *scratch) noexcept
    {
        if (!scratch)
            return;
        // reset() keeps the vector buffers and drops their contents, which is
        // the whole point of the lease.
        scratch->reset();
        try
        {
            auto &pool = scratch_pool();
            if (pool.size() >= kMaxPooledScratch)
            {
                delete scratch;
                return;
            }
            pool.push_back(scratch);
        }
        catch (...)
        {
            delete scratch;
        }
    }

    // A var whose whole storage is a single element but whose logical shape is
    // larger. jittor materialises `x <op> scalar` this way: the broadcast node
    // is folded away and the operand becomes a stride-0 view of the four bytes
    // the `array` op wrote.
    static inline bool is_scalar_expansion(Var *v)
    {
        return v && v->num > 1 && !v->is_contiguous() &&
               v->storage_span_bytes() == v->dsize();
    }

    // Common functionality for adding input/output variables
    void BaseOpRunner::add(Var *v, bool is_input)
    {
        if (is_input)
        {
            in_.push_back(v);
        }
        else
        {
            out_.push_back(v);
        }
        return;
    }

    void BaseOpRunner::setupInputDesc()
    {
        auto input_num = in_.size();
        inputShapes.resize(input_num);
        // CANN infers the result shape from the operands, so collapsing *every*
        // operand to one element would make it infer a one-element result and
        // reject the real output. At least one operand therefore keeps its full
        // shape; when they are all one-element expansions nothing is collapsed
        // and the launch is exactly what it was before.
        bool collapse = false;
        if (collapsesScalarInputs())
        {
            size_t collapsible = 0;
            for (size_t i = 0; i < input_num; i++)
                if (is_scalar_expansion(in_[i])) collapsible++;
            collapse = collapsible && collapsible < input_num;
        }
        for (size_t input_idx = 0; input_idx < input_num; input_idx++)
        {
            // Built in place: the shape used to be assembled in a temporary
            // vector and then copied into inputShapes, two allocations per
            // input per launch.
            auto &shape = inputShapes[input_idx];
            Var *v = in_[input_idx];
            if (collapse && is_scalar_expansion(v))
            {
                // One element behind a stride-0 expansion: hand CANN the real
                // shape and let it broadcast (see collapsesScalarInputs).
                shape.assign(1, 1);
                continue;
            }
            const auto &var_shape = v->shape;
            shape.resize(var_shape.size());
            for (int j = 0; j < var_shape.size(); j++)
            {
                shape[j] = var_shape[j];
            }
        }

        inputTensors.resize(input_num, nullptr);
        for (size_t idx = 0; idx < input_num; idx++)
        {
            inputTensors[idx] = nullptr;
            // A collapsed input is described as a contiguous one-element
            // tensor, so it must not carry the expanded var as its storage --
            // apply_storage_strides would put the stride-0 view back.
            const bool collapsed = inputShapes[idx].size() == 1 && inputShapes[idx][0] == 1
                                   && in_[idx]->num > 1;
            auto ret = AcquireAclTensor(scratch->descriptors, inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw, collapsed ? nullptr : in_[idx]);
            if (ret != ACL_SUCCESS) LOGf << name << ": input tensor creation failed. ERROR:" << ret;
        }
    }

    void BaseOpRunner::cleanupDesc()
    {
        auto input_num = in_.size();
        auto output_num = out_.size();
        // Clearing the slot as it goes keeps the failure path in
        // AclExecutionRunner::run from destroying a descriptor that is already
        // back in the pool.
        for (int idx = 0; idx < input_num; idx++)
        {
            RecycleAclTensor(scratch->descriptors, inputTensors[idx]);
            inputTensors[idx] = nullptr;
        }
        for (int idx = 0; idx < output_num; idx++)
        {
            RecycleAclTensor(scratch->descriptors, outputTensors[idx]);
            outputTensors[idx] = nullptr;
        }
    }

    void BaseOpRunner::setupOutputDesc()
    {
        auto output_num = out_.size();

        outputShapes.resize(output_num);
        for (size_t output_idx = 0; output_idx < output_num; output_idx++)
        {
            auto &shape = outputShapes[output_idx];
            const auto &var_shape = out_[output_idx]->shape;
            shape.resize(var_shape.size());
            for (int j = 0; j < var_shape.size(); j++)
            {
                shape[j] = var_shape[j];
            }
        }

        outputTensors.resize(output_num, nullptr);
        for (size_t idx = 0; idx < output_num; idx++)
        {
            outputTensors[idx] = nullptr;
            auto ret = AcquireAclTensor(scratch->descriptors, outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw, out_[idx]);
            if (ret != ACL_SUCCESS) LOGf << name << ": output tensor creation failed. ERROR:" << ret;
        }
    }

    void BaseOpRunner::syncRun()
    {
        if (!runtime_device_state().sync_run)
            return;
        auto sync_ret = aclrtSynchronizeStream(aclstream);
        if (sync_ret != ACL_SUCCESS)
        {
            LOGf << "ACL operator" << name
                 << ": aclrtSynchronizeStream failed, return code"
                 << sync_ret << acl_error_to_string(sync_ret);
        }
    }

    void BaseOpRunner::checkRet(aclnnStatus ret)
    {
        if (ret == ACL_SUCCESS)
            return;
        const char *recent_error = aclGetRecentErrMsg();
        LOGf << "ACL operator" << name
             << ": aclnn workspace-size query failed, return code" << ret
             << acl_error_to_string(ret) << "recent error:"
             << (recent_error == nullptr ? "unavailable" : recent_error);
    }

    void BaseOpRunner::launch(aclnnStatus workspace_ret,
                              const AclExecuteLauncher &launcher,
                              bool synchronize)
    {
        checkRet(workspace_ret);
        if (!launcher)
            LOGf << "ACL operator has an empty execute launcher:" << name;
        if (workspaceSize > 0)
            mallocWorkSpace(workspaceSize);
        auto launch_ret = launcher(
            workspaceAddr, workspaceSize, executor, aclstream);
        if (launch_ret != ACL_SUCCESS)
            LOGf << "ACL operator" << name
                 << ": execute launcher failed, return code" << launch_ret
                 << acl_error_to_string(launch_ret);
        if (synchronize)
            syncRun();
    }

    // Base run method with common operator lookup logic
    void BaseOpRunner::run()
    {
        if (is_group_op)
        {
            auto it = acl_op_registry().find(name);
            if (it == acl_op_registry().end())
            {
                LOGf << "ACL operator has no registered launcher:" << name;
            }
            setupInputDesc();
            setupOutputDesc();
            executeOp(it);
            cleanupDesc();
        }
        else
        {
            auto it = acl_op_registry().find(name);
            if (it == acl_op_registry().end())
            {
                LOGf << "ACL operator has no registered launcher:" << name;
            }
            setupInputDesc();
            setupOutputDesc();
            executeOp(it);
            cleanupDesc();
        }
    }

}
