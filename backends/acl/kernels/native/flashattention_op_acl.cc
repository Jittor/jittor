#pragma once
#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <Python.h>
#include <pystate.h>
#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
#include "flashattention_op_acl.h"

namespace jittor
{
    FlashAttentionOpRunner::FlashAttentionOpRunner() : BaseOpRunner("FlashAttention")
    {
    }

    void FlashAttentionOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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

    void FlashAttentionBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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

    void IncreFlashAttentionOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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
        // The paged IncreFlashAttention kernel and vLLM-Ascend's fused infer
        // kernel can round BF16 decode differently on real model inputs. Keep
        // the old kernel as the default and as the fallback for other shapes.
        const char *fusedDecode = std::getenv("JITTOR_NPU_PAGED_DECODE_FIA");
        const bool useFusedDecode = fusedDecode != nullptr &&
            std::strcmp(fusedDecode, "1") == 0 && attr->hasBlockTable &&
            attr->inputLayout == "BNSD" &&
            inputShapes.size() >= 3 && inputShapes[0].size() == 4 &&
            inputShapes[1].size() == 5 && outputShapes.size() == 1 &&
            outputShapes[0] == inputShapes[0] &&
            inputShapes[0][0] == 1 && inputShapes[0][1] == attr->headNum &&
            inputShapes[0][2] == 1 && inputShapes[0][3] == 128 &&
            inputShapes[1][2] == 128 && inputShapes[1][4] == 128 &&
            inputShapes[1][3] == attr->keyValueHeadNum &&
            attr->blockSize == 128 && attr->actualSeqLengths.size() == 1 &&
            get_dtype(in_[0]->dtype()) == ACL_BF16 &&
            get_dtype(in_[1]->dtype()) == ACL_BF16;
        if (std::getenv("JITTOR_NPU_PAGED_DECODE_FIA_DEBUG") != nullptr)
        {
            static std::atomic<int> debugCount{0};
            if (debugCount.fetch_add(1) < 4)
                std::fprintf(stderr,
                    "paged decode FIA dispatch=%d q=[%lld,%lld,%lld,%lld] "
                    "cache=[%lld,%lld,%lld,%lld,%lld] heads=%lld kvheads=%lld "
                    "block=%lld seq_count=%zu layout=%s dtype=%d\n",
                    static_cast<int>(useFusedDecode),
                    static_cast<long long>(inputShapes[0].size() > 0 ? inputShapes[0][0] : -1),
                    static_cast<long long>(inputShapes[0].size() > 1 ? inputShapes[0][1] : -1),
                    static_cast<long long>(inputShapes[0].size() > 2 ? inputShapes[0][2] : -1),
                    static_cast<long long>(inputShapes[0].size() > 3 ? inputShapes[0][3] : -1),
                    static_cast<long long>(inputShapes[1].size() > 0 ? inputShapes[1][0] : -1),
                    static_cast<long long>(inputShapes[1].size() > 1 ? inputShapes[1][1] : -1),
                    static_cast<long long>(inputShapes[1].size() > 2 ? inputShapes[1][2] : -1),
                    static_cast<long long>(inputShapes[1].size() > 3 ? inputShapes[1][3] : -1),
                    static_cast<long long>(inputShapes[1].size() > 4 ? inputShapes[1][4] : -1),
                    static_cast<long long>(attr->headNum),
                    static_cast<long long>(attr->keyValueHeadNum),
                    static_cast<long long>(attr->blockSize),
                    attr->actualSeqLengths.size(), attr->inputLayout.c_str(),
                    static_cast<int>(get_dtype(in_[0]->dtype())));
        }
        if (useFusedDecode)
        {
            // FIA expects TND for this single-token case. These are only
            // descriptor views; query/output buffers retain their addresses.
            const std::vector<int64_t> viewDims{1, attr->headNum, 128};
            const std::vector<int64_t> viewStrides{attr->headNum * 128, 128, 1};
            std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> queryView(
                aclCreateTensor(viewDims.data(), viewDims.size(),
                    get_dtype(in_[0]->dtype()), viewStrides.data(), 0,
                    aclFormat::ACL_FORMAT_ND, inputShapes[0].data(),
                    inputShapes[0].size(), in_[0]->mem_ptr), aclDestroyTensor);
            std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> outputView(
                aclCreateTensor(viewDims.data(), viewDims.size(),
                    get_dtype(out_[0]->dtype()), viewStrides.data(), 0,
                    aclFormat::ACL_FORMAT_ND, outputShapes[0].data(),
                    outputShapes[0].size(), out_[0]->mem_ptr), aclDestroyTensor);
            const int64_t one = 1;
            std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> qLengths(
                aclCreateIntArray(&one, 1), aclDestroyIntArray);
            CHECK(queryView != nullptr && outputView != nullptr &&
                  qLengths != nullptr && actualSeqLengths != nullptr);
            char fusedLayout[] = "TND";
            ret = aclnnFusedInferAttentionScoreGetWorkspaceSize(
                queryView.get(), key, value, nullptr, nullptr,
                qLengths.get(), actualSeqLengths,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                blockTable, nullptr, nullptr,
                attr->headNum, attr->scale, 2147483647, 2147483647,
                fusedLayout, attr->keyValueHeadNum, 0, 0, attr->blockSize,
                0, false, outputView.get(), nullptr, &workspaceSize, &executor);
            launch(ret, aclnnFusedInferAttentionScore, true);
        }
        else
        {
            ret = aclnnIncreFlashAttentionV4GetWorkspaceSize(
                inputTensors[0], key, value, nullptr, nullptr, actualSeqLengths,
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                blockTable, nullptr, attr->headNum, attr->scale, layout,
                attr->keyValueHeadNum, attr->blockSize, attr->innerPrecise,
                outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnIncreFlashAttentionV4, true);
        }
        if (actualSeqLengths != nullptr)
            aclDestroyIntArray(actualSeqLengths);
        if (keyView != nullptr)
            aclDestroyTensor(keyView);
        if (valueView != nullptr)
            aclDestroyTensor(valueView);
        return;
    }

    FusedInferPrefillOpRunner::FusedInferPrefillOpRunner()
        : BaseOpRunner("FusedInferPrefill")
    {
    }

    void FusedInferPrefillOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<FusedInferPrefillAttr *>(op_attr.get());
        CHECK(attr != nullptr);
        CHECK(inputShapes.size() == 4);
        CHECK(outputShapes.size() == 1);
        CHECK(inputShapes[0].size() == 3);
        CHECK(inputShapes[1].size() == 3);
        CHECK(inputShapes[2].size() == 3);
        CHECK(inputShapes[0][0] == inputShapes[1][0]);
        CHECK(inputShapes[0][0] == inputShapes[2][0]);
        CHECK(inputShapes[1] == inputShapes[2]);
        CHECK(inputShapes[0][2] == inputShapes[1][2]);
        CHECK(inputShapes[3].size() == 2);
        CHECK(inputShapes[3][0] == 2048 && inputShapes[3][1] == 2048);
        CHECK(outputShapes[0] == inputShapes[0]);
        const int64_t tokens = inputShapes[0][0];
        const int64_t num_heads = inputShapes[0][1];
        const int64_t num_kv_heads = inputShapes[1][1];
        CHECK(tokens > 1 && tokens <= 2048);
        CHECK(num_kv_heads > 0 && num_heads % num_kv_heads == 0);
        CHECK(!attr->actualSeqLengths.empty());
        CHECK(attr->actualSeqLengths.back() == tokens);
        int64_t previous = 0;
        for (int64_t end : attr->actualSeqLengths)
        {
            CHECK(end > previous && end - previous > 1);
            previous = end;
        }

        aclTensor *key_tensor = inputTensors[1];
        aclTensor *value_tensor = inputTensors[2];
        // Match the existing IncreFlashAttention list lifetime: CANN's
        // executor holds these descriptors through launch, while BaseOpRunner
        // retains and cleans the underlying input aclTensor descriptors.
        auto key = aclCreateTensorList(&key_tensor, 1);
        auto value = aclCreateTensorList(&value_tensor, 1);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> lengths(
            aclCreateIntArray(attr->actualSeqLengths.data(),
                              attr->actualSeqLengths.size()),
            aclDestroyIntArray);
        CHECK(key != nullptr && value != nullptr && lengths != nullptr);
        char layout[] = "TND";
        ret = aclnnFusedInferAttentionScoreGetWorkspaceSize(
            inputTensors[0], key, value, nullptr, inputTensors[3],
            lengths.get(), lengths.get(),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr,
            num_heads, attr->scale, 2147483647, 2147483647, layout,
            num_kv_heads, 3, 0, 128, 0, false,
            outputTensors[0], nullptr, &workspaceSize, &executor);
        launch(ret, aclnnFusedInferAttentionScore, true);
    }

    KVCacheMemcpyOpRunner::KVCacheMemcpyOpRunner()
        : BaseOpRunner("KVCacheMemcpy")
    {
    }

    void KVCacheMemcpyOpRunner::run()
    {
        // Cache writes use aclrtMemcpyAsync and have no aclnn workspace or
        // launcher. Keep this exception local to the concrete copy runner;
        // BaseOpRunner still rejects every unregistered operator.
        auto entry = acl_op_registry().end();
        try
        {
            setupInputDesc();
            setupOutputDesc();
            executeOp(entry);
            cleanupDesc();
        }
        catch (...)
        {
            // A failed setup may have constructed only part of either vector.
            // Drain queued copies before releasing descriptors and preserve
            // the original exception, including the original copy error.
            aclrtSynchronizeStream(aclstream);
            for (auto *&tensor : inputTensors)
                if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
            for (auto *&tensor : outputTensors)
                if (tensor) { aclDestroyTensor(tensor); tensor = nullptr; }
            throw;
        }
    }

    void KVCacheMemcpyOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<KVCacheMemcpyAttr *>(op_attr.get());
        CHECK(in_.size() == 3);
        CHECK(out_.size() == 1);
        // The input edge materializes the old cache before this partial write.
        // outputs= requests sharing, but a materialized broadcast initializer
        // can have a different buffer. Preserve its untouched rows explicitly.
        CHECK(inputShapes[2] == outputShapes[0]);
        CHECK(in_[2]->is_contiguous());
        CHECK(in_[2]->size == out_[0]->size);
        if (in_[2]->mem_ptr != out_[0]->mem_ptr)
        {
            ret = aclrtMemcpyAsync(out_[0]->mem_ptr, out_[0]->size,
                in_[2]->mem_ptr, in_[2]->size,
                ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
            if (ret != ACL_SUCCESS)
                LOGf << name << ": previous cache copy failed. ERROR:" << ret;
        }
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
            if (ret != ACL_SUCCESS) LOGf << name << ": key cache memcpy failed. ERROR:" << ret;
            ret = aclrtMemcpyAsync(
                cache + valueOffset, out_[0]->size - valueOffset,
                value + sourceOffset, tokenBytes,
                ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
            if (ret != ACL_SUCCESS) LOGf << name << ": value cache memcpy failed. ERROR:" << ret;
        }
    }

}
