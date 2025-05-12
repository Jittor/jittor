#pragma once
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
#include "concat_op_acl.h"

namespace jittor
{
    ConcatOpRunner::ConcatOpRunner() : BaseOpRunner("Concat")
    {
    }

    void ConcatOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto input_num = in_.size();
        std::vector<aclTensor *> concatTensorList = {};
        for (int i = 0; i < input_num; i++)
        {
            concatTensorList.push_back(inputTensors[i]);
        }
        auto concatTensorListInput = aclCreateTensorList(&concatTensorList[0], input_num);
        auto attr = dynamic_cast<ConcatAttr *>(op_attr.get());
        ret = aclnnCatGetWorkspaceSize(concatTensorListInput, attr->dim, outputTensors[0], &workspaceSize, &executor);
        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnCat(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnCat failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

    SplitWithSizeOpRunner::SplitWithSizeOpRunner() : BaseOpRunner("SplitWithSize")
    {
    }

    void SplitWithSizeOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto output_num = out_.size();
        auto attr = dynamic_cast<SplitWithSizeAttr *>(op_attr.get());
        auto splitSize = aclCreateIntArray(attr->splitSize.data(), attr->splitSize.size());
        auto tensorList = aclCreateTensorList(&outputTensors[0], output_num);
        ret = aclnnSplitWithSizeGetWorkspaceSize(inputTensors[0], splitSize, attr->dim, tensorList, &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnSplitWithSize(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnSplitWithSize failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

}