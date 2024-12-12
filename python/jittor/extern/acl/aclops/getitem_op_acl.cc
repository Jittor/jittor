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
#include "getitem_op_acl.h"

namespace jittor
{
    MaskedSelectOpRunner::MaskedSelectOpRunner() : BaseOpRunner("MaskedSelect")
    {
    }

    void MaskedSelectOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        ret = aclnnMaskedSelectGetWorkspaceSize(inputTensors[0], inputTensors[1], outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnMaskedSelect(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnMaskedSelect failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }


    IndexOpRunner::IndexOpRunner() : BaseOpRunner("Index")
    {
    }
        
    void IndexOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto input_num = in_.size();
        auto indexTensorList = aclCreateTensorList(&inputTensors[1], input_num - 1);
        ret = aclnnIndexGetWorkspaceSize(inputTensors[0], indexTensorList, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnIndex(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnIndex failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

    SliceV2OpRunner::SliceV2OpRunner() : BaseOpRunner("SliceV2")
    {
    }
        
    void SliceV2OpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<StrideAttr *>(op_attr.get());
        auto begins = aclCreateIntArray(attr->begins.data(), attr->begins.size());
        auto ends = aclCreateIntArray(attr->ends.data(), attr->ends.size());
        auto steps = aclCreateIntArray(attr->steps.data(), attr->steps.size());
        auto axes = aclCreateIntArray(attr->axes.data(), attr->axes.size());
        ret = aclnnSliceV2GetWorkspaceSize(inputTensors[0], begins, ends, axes, steps, outputTensors[0], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnSliceV2(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnSliceV2 failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        return;
    }


    IndexPutImplAccumulateOpRunner::IndexPutImplAccumulateOpRunner() : BaseOpRunner("IndexPutImplAccumulate")
    {
    }
        
    void IndexPutImplAccumulateOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto input_num = in_.size();
        std::vector<aclTensor *> indexTensorList = {};
        for (int i = 1; i < input_num; i++)
        {
            indexTensorList.push_back(inputTensors[i]);
        }
        auto indexTensorListInput = aclCreateTensorList(&indexTensorList[0], input_num - 1);
        ret = aclnnIndexPutImplGetWorkspaceSize(outputTensors[0], indexTensorListInput, inputTensors[0], true, true, &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnIndexPutImpl(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnIndexPutImpl failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        return;
    }


    StridedSliceAssignV2OpRunner::StridedSliceAssignV2OpRunner() : BaseOpRunner("StridedSliceAssignV2")
    {
    }
        

    void StridedSliceAssignV2OpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<StrideAttr *>(op_attr.get());
        auto begins = aclCreateIntArray(attr->begins.data(), attr->begins.size());
        auto ends = aclCreateIntArray(attr->ends.data(), attr->ends.size());
        auto steps = aclCreateIntArray(attr->steps.data(), attr->steps.size());
        auto axes = aclCreateIntArray(attr->axes.data(), attr->axes.size());
        ret = aclnnStridedSliceAssignV2GetWorkspaceSize(outputTensors[0], inputTensors[0], begins, ends, steps, axes, &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnStridedSliceAssignV2(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnStridedSliceAssignV2 failed. ERROR: %d\n", name.c_str(), ret); return);

        // syncRun(); 



        return;
    }
}