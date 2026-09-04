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

        launch(ret, aclnnFlashAttentionScoreV2, true);
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

        launch(ret, aclnnFlashAttentionScoreGradV2, true);
        return;
    }

    IncreFlashAttentionOpRunner::IncreFlashAttentionOpRunner() : BaseOpRunner("IncreFlashAttention")
    {
    }

    void IncreFlashAttentionOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<IncreFlashAttentionAttr *>(op_attr.get());
        aclTensor *keyTensor = nullptr;
        aclTensor *valueTensor = nullptr;
        aclTensor *blockTable = nullptr;
        aclTensor *keyView = nullptr;
        aclTensor *valueView = nullptr;
        if (attr->hasBlockTable)
        {
            CHECK(inputShapes.size() == 3);
            CHECK(inputShapes[1].size() == 5);
            CHECK(inputShapes[1][1] == 2);
            auto &cacheShape = inputShapes[1];
            int64_t blocks = cacheShape[0];
            int64_t blockSize = cacheShape[2];
            int64_t width = cacheShape[3] * cacheShape[4];
            std::vector<int64_t> viewDims{blocks, blockSize, width};
            std::vector<int64_t> viewStrides{2 * blockSize * width, width, 1};
            int64_t valueOffset = blockSize * width;
            keyView = aclCreateTensor(
                viewDims.data(), viewDims.size(), get_dtype(in_[1]->dtype()),
                viewStrides.data(), 0, aclFormat::ACL_FORMAT_ND,
                cacheShape.data(), cacheShape.size(), in_[1]->mem_ptr);
            valueView = aclCreateTensor(
                viewDims.data(), viewDims.size(), get_dtype(in_[1]->dtype()),
                viewStrides.data(), valueOffset, aclFormat::ACL_FORMAT_ND,
                cacheShape.data(), cacheShape.size(), in_[1]->mem_ptr);
            CHECK(keyView != nullptr);
            CHECK(valueView != nullptr);
            keyTensor = keyView;
            valueTensor = valueView;
            blockTable = inputTensors[2];
        }
        else
        {
            keyTensor = inputTensors[1];
            valueTensor = inputTensors[2];
        }
        // The executor retains these lists through the execute call.
        auto key = aclCreateTensorList(&keyTensor, 1);
        auto value = aclCreateTensorList(&valueTensor, 1);
        aclIntArray *actualSeqLengths = nullptr;
        if (!attr->actualSeqLengths.empty())
        {
            actualSeqLengths = aclCreateIntArray(
                attr->actualSeqLengths.data(), attr->actualSeqLengths.size());
        }
        char *layout = const_cast<char *>(attr->inputLayout.data());
        ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(
            inputTensors[0], key, value, nullptr, nullptr, actualSeqLengths,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            blockTable, nullptr, attr->headNum, attr->scale, layout,
            attr->keyValueHeadNum, attr->blockSize, attr->innerPrecise, outputTensors[0],
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
        if (actualSeqLengths != nullptr)
            aclDestroyIntArray(actualSeqLengths);
        if (keyView != nullptr)
            aclDestroyTensor(keyView);
        if (valueView != nullptr)
            aclDestroyTensor(valueView);
        return;
    }

    KVCacheMemcpyOpRunner::KVCacheMemcpyOpRunner() : BaseOpRunner("KVCacheMemcpy")
    {
    }

    void KVCacheMemcpyOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<KVCacheMemcpyAttr *>(op_attr.get());
        CHECK(in_.size() == 2);
        CHECK(out_.size() == 1);
        CHECK(inputShapes[0].size() == 3);
        CHECK(inputShapes[1] == inputShapes[0]);
        CHECK(outputShapes[0].size() == 5);
        CHECK(outputShapes[0][1] == 2);
        CHECK(outputShapes[0][2] == attr->blockSize);
        CHECK(outputShapes[0][3] == inputShapes[0][1]);
        CHECK(outputShapes[0][4] == inputShapes[0][2]);
        CHECK(attr->slots.size() >= size_t(inputShapes[0][0]));
        CHECK(in_[0]->size == in_[1]->size);

        int64_t tokens = inputShapes[0][0];
        int64_t tokenBytes = tokens > 0 ? in_[0]->size / tokens : 0;
        int64_t capacity = outputShapes[0][0] * attr->blockSize;
        auto *cache = static_cast<char *>(out_[0]->mem_ptr);
        auto *key = static_cast<char *>(in_[0]->mem_ptr);
        auto *value = static_cast<char *>(in_[1]->mem_ptr);
        for (int64_t token = 0; token < tokens; ++token)
        {
            int64_t slot = attr->slots[token];
            if (slot < 0 || slot >= capacity)
                continue;
            int64_t block = slot / attr->blockSize;
            int64_t offset = slot % attr->blockSize;
            int64_t keyRow = block * 2 * attr->blockSize + offset;
            int64_t valueRow = keyRow + attr->blockSize;
            int64_t keyOffset = keyRow * tokenBytes;
            int64_t valueOffset = valueRow * tokenBytes;
            int64_t sourceOffset = token * tokenBytes;
            ret = aclrtMemcpyAsync(
                cache + keyOffset, out_[0]->size - keyOffset,
                key + sourceOffset, tokenBytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, return);
            ret = aclrtMemcpyAsync(
                cache + valueOffset, out_[0]->size - valueOffset,
                value + sourceOffset, tokenBytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

}
