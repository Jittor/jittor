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
#include "flashattention_op_acl.h"

namespace jittor
{
    FlashAttentionOpRunner::FlashAttentionOpRunner() : BaseOpRunner("FlashAttention")
    {
    }

    void FlashAttentionOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<FlashAttentionAttr *>(op_attr.get());
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> prefix(
            aclCreateIntArray(attr->prefix.data(), attr->prefix.size()),
            aclDestroyIntArray);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> qstart(
            aclCreateIntArray(attr->qStartIdx.data(), attr->qStartIdx.size()),
            aclDestroyIntArray);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> kvstart(
            aclCreateIntArray(attr->kvStartIdx.data(), attr->kvStartIdx.size()),
            aclDestroyIntArray);
        char *layout = const_cast<char *>(attr->inputLayout.data());
        ret = aclnnFlashAttentionScoreV2GetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], attr->hasRealshift ? inputTensors[3] : nullptr, attr->hasDropmask ? inputTensors[4] : nullptr, nullptr, attr->hasAttentmask ? inputTensors[6] : nullptr, prefix.get(), qstart.get(), kvstart.get(), attr->scale, attr->keepProb, attr->preToken, attr->nextToken, attr->headNum, layout, attr->innerPrecise, attr->sparseMode, attr->psetype, outputTensors[0], outputTensors[1], nullptr, outputTensors[2], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnFlashAttentionScoreV2(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnFlashAttentionScoreV2 failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

    FlashAttentionBackwardOpRunner::FlashAttentionBackwardOpRunner() : BaseOpRunner("FlashAttentionBackward")
    {
    }

    void FlashAttentionBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<FlashAttentionAttr *>(op_attr.get());
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> prefix(
            aclCreateIntArray(attr->prefix.data(), attr->prefix.size()),
            aclDestroyIntArray);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> qstart(
            aclCreateIntArray(attr->qStartIdx.data(), attr->qStartIdx.size()),
            aclDestroyIntArray);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> kvstart(
            aclCreateIntArray(attr->kvStartIdx.data(), attr->kvStartIdx.size()),
            aclDestroyIntArray);
        char *layout = const_cast<char *>(attr->inputLayout.data());
        ret = aclnnFlashAttentionScoreGradV2GetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], inputTensors[3], attr->hasRealshift ? inputTensors[4] : nullptr, attr->hasDropmask ? inputTensors[5] : nullptr, nullptr, attr->hasAttentmask ? inputTensors[7] : nullptr, inputTensors[8], inputTensors[9], nullptr, inputTensors[10], prefix.get(), qstart.get(), kvstart.get(), attr->scale, attr->keepProb, attr->preToken, attr->nextToken, attr->headNum, layout, attr->innerPrecise, attr->sparseMode, attr->psetype, outputTensors[0], outputTensors[1], outputTensors[2], nullptr, &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnFlashAttentionScoreGradV2(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnFlashAttentionScoreGradV2 failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

    IncreFlashAttentionOpRunner::IncreFlashAttentionOpRunner() : BaseOpRunner("IncreFlashAttention")
    {
    }

    void IncreFlashAttentionOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<IncreFlashAttentionAttr *>(op_attr.get());
        // The executor retains these lists through the execute call.
        auto key = aclCreateTensorList(&inputTensors[1], 1);
        auto value = aclCreateTensorList(&inputTensors[2], 1);
        char *layout = const_cast<char *>(attr->inputLayout.data());
        ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(
            inputTensors[0], key, value, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, attr->headNum, attr->scale, layout,
            attr->keyValueHeadNum, 0, attr->innerPrecise, outputTensors[0],
            &workspaceSize, &executor);
        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnIncreFlashAttentionV4(
            workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnIncreFlashAttentionV4 failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();
        return;
    }

}
