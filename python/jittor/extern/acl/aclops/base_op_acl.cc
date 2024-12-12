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
#include "binary_op_acl.h"
#include "base_op.h"

namespace jittor
{
    extern int sync_run;
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
        for (int input_idx = 0; input_idx < input_num; input_idx++)
        {
            std::vector<int64_t> shape;
            for (int j = 0; j < in_[input_idx]->shape.size(); j++)
            {
                shape.push_back(in_[input_idx]->shape[j]);
            }
            inputShapes.push_back(shape);
        }

        for (int idx = 0; idx < input_num; idx++)
        {
            inputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    void BaseOpRunner::cleanupDesc()
    {
        auto input_num = in_.size();
        auto output_num = out_.size();
        for (int idx = 0; idx < input_num; idx++)
        {
            aclDestroyTensor(inputTensors[idx]);
        }
        for (int idx = 0; idx < output_num; idx++)
        {
            aclDestroyTensor(outputTensors[idx]);
        }
    }

    void BaseOpRunner::setupOutputDesc()
    {
        auto output_num = out_.size();

        for (int output_idx = 0; output_idx < output_num; output_idx++)
        {
            std::vector<int64_t> shape;
            for (int j = 0; j < out_[output_idx]->shape.size(); j++)
            {
                shape.push_back(out_[output_idx]->shape[j]);
            }
            outputShapes.push_back(shape);
        }

        for (int idx = 0; idx < output_num; idx++)
        {
            outputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    void BaseOpRunner::syncRun()
    {
        if(sync_run) {
            ret = aclrtSynchronizeStream(aclstream);
            CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclrtSynchronizeStream failed. ERROR: %d\n", name.c_str(), ret); return);
        }
    }

    
    void BaseOpRunner::checkRet(aclnnStatus ret)
    {
        if (ret != ACL_SUCCESS)
        {
            auto tmp_err_msg = aclGetRecentErrMsg();
            LOGir << name << ", " << tmp_err_msg;
        }

        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxxGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
    }

    // Base run method with common operator lookup logic
    void BaseOpRunner::run()
    {
        if (is_group_op)
        {
            auto it = aclOpFuncMap.find(name);
            if (it == aclOpFuncMap.end())
            {
                LOGir << "aclOpFuncMap Not supported op: " << name;
                throw std::runtime_error("Unsupported operation type.");
            }
            setupInputDesc();
            setupOutputDesc();
            executeOp(it);
            cleanupDesc();
        }
        else
        {
            auto it = aclOpFuncMap.find(name);
            setupInputDesc();
            setupOutputDesc();
            executeOp(it);
            cleanupDesc();
        }
    }

}
