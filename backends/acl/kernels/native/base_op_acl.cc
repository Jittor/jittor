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
        for (size_t input_idx = 0; input_idx < input_num; input_idx++)
        {
            // Built in place: the shape used to be assembled in a temporary
            // vector and then copied into inputShapes, two allocations per
            // input per launch.
            auto &shape = inputShapes[input_idx];
            const auto &var_shape = in_[input_idx]->shape;
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
            auto ret = AcquireAclTensor(scratch->descriptors, inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw, in_[idx]);
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
