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
#include "aclnnop/aclnn_prod.h"
#include "reduce_op_acl.h"

namespace jittor
{
    ReduceOpRunner::ReduceOpRunner() : BaseOpRunner("reduce")
    {
        use_nchw = false;
    }

    ReduceOpRunner::~ReduceOpRunner()
    {
        if (dim != nullptr)
            aclDestroyIntArray(dim);
    }

    void ReduceOpRunner::setupInputDesc()
    {
        auto input_num = in_.size();
        for (int input_idx = 0; input_idx < input_num; input_idx++)
        {
            std::vector<int64_t> shape;
            for (int j = 0; j < in_[input_idx]->shape.size(); j++)
            {
                shape.push_back(in_[input_idx]->shape[j]);
            }
            // ACL reduce kernels (aclnnAmax/Amin/...) reject 1-D inputs with
            // ERROR 161002. Pad a 1-D input to 2-D (1, N); axes are shifted by
            // +1 in setupOutputDesc so the reduction result is unchanged.
            if (input_idx == 0 && shape.size() == 1)
            {
                shape.insert(shape.begin(), 1);
                input_padded_1d = true;
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

    void ReduceOpRunner::setupOutputDesc()
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

        attr = dynamic_cast<ReduceAttr *>(op_attr.get());
        keepdims = attr->keepdims;
        // A 1-D reduce is always a full reduce producing a single element.
        // When the input was padded to (1, N) we reduce over all padded axes
        // and force a scalar output (keepdims off): ACL rejects the (1,)/keepdims
        // combination with ERROR 161002 but accepts a true scalar.
        shifted_axes_.assign(attr->axes.begin(), attr->axes.end());
        if (input_padded_1d)
        {
            shifted_axes_.clear();
            for (int64_t ax = 0; ax < (int64_t)inputShapes[0].size(); ax++)
                shifted_axes_.push_back(ax);
            keepdims = false;
        }
        dim = aclCreateIntArray(shifted_axes_.data(), shifted_axes_.size());

        if (op_idx <= 13)
        {
            if (input_padded_1d || attr->axes.size() == in_[0]->shape.size())
                outputShapes[0] = {};
        }

        for (int idx = 0; idx < output_num; idx++)
        {
            outputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    void ReduceOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        switch (op_idx)
        {
        case 9:
        {
            ret = aclnnReduceSumGetWorkspaceSize(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnReduceSum, true);
            break;
        }
        case 10:
        {
            ret = aclnnMeanGetWorkspaceSize(inputTensors[0], dim, keepdims, get_dtype(out_[0]->dtype()), outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnMean, true);
            break;
        }
        case 11:
        {
            ret = aclnnAmaxGetWorkspaceSize(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnAmax, true);
            break;
        }
        case 12:
        {
            ret = aclnnAminGetWorkspaceSize(inputTensors[0], dim, keepdims, outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnAmin, true);
            break;
        }
        case 13:
        {
            const bool reduce_all = attr->axes.size() == in_[0]->shape.size();
            if (input_padded_1d || reduce_all)
            {
                ret = aclnnProdGetWorkspaceSize(
                    inputTensors[0], get_dtype(out_[0]->dtype()),
                    outputTensors[0], &workspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnProdGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            }
            else if (shifted_axes_.size() == 1)
            {
                ret = aclnnProdDimGetWorkspaceSize(
                    inputTensors[0], shifted_axes_[0], keepdims,
                    get_dtype(out_[0]->dtype()), outputTensors[0],
                    &workspaceSize, &executor);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnProdDimGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
            }
            else
            {
                std::vector<int64_t> axes = shifted_axes_;
                if (!keepdims)
                    std::sort(axes.rbegin(), axes.rend());

                std::vector<int64_t> current_shape = inputShapes[0];
                aclTensor *current_tensor = inputTensors[0];
                std::vector<aclTensor *> intermediate_tensors;
                std::vector<void *> intermediate_buffers;

                for (size_t index = 0; index < axes.size(); index++)
                {
                    const int64_t axis = axes[index];
                    std::vector<int64_t> next_shape = current_shape;
                    if (keepdims)
                        next_shape[axis] = 1;
                    else
                        next_shape.erase(next_shape.begin() + axis);

                    const bool is_last = index + 1 == axes.size();
                    aclTensor *next_tensor = outputTensors[0];
                    if (!is_last)
                    {
                        uint64_t buffer_size = out_[0]->dsize();
                        for (int64_t extent : next_shape)
                            buffer_size *= extent;

                        void *buffer = nullptr;
                        ret = aclrtMalloc(
                            &buffer, buffer_size, ACL_MEM_MALLOC_HUGE_FIRST);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: product intermediate allocation failed. ERROR: %d\n", name.c_str(), ret); return);
                        intermediate_buffers.push_back(buffer);

                        next_tensor = nullptr;
                        ret = CreateAclTensor(
                            next_shape, buffer, buffer_size,
                            get_dtype(out_[0]->dtype()), &next_tensor, use_nchw);
                        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: product intermediate tensor creation failed. ERROR: %d\n", name.c_str(), ret); return);
                        intermediate_tensors.push_back(next_tensor);
                    }

                    ret = aclnnProdDimGetWorkspaceSize(
                        current_tensor, axis, keepdims,
                        get_dtype(out_[0]->dtype()), next_tensor,
                        &workspaceSize, &executor);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnProdDimGetWorkspaceSize failed. ERROR: %d\n", name.c_str(), ret); return);
                    if (workspaceSize > 0)
                        mallocWorkSpace(workspaceSize);
                    ret = aclnnProdDim(
                        workspaceAddr, workspaceSize, executor, aclstream);
                    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnProdDim failed. ERROR: %d\n", name.c_str(), ret); return);

                    current_shape = std::move(next_shape);
                    current_tensor = next_tensor;
                }

                ret = aclrtSynchronizeStream(aclstream);
                CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: product intermediate synchronization failed. ERROR: %d\n", name.c_str(), ret); return);
                for (aclTensor *tensor : intermediate_tensors)
                    aclDestroyTensor(tensor);
                for (void *buffer : intermediate_buffers)
                    aclrtFree(buffer);
                break;
            }
            if (workspaceSize > 0)
            {
                mallocWorkSpace(workspaceSize);
            }
            ret = input_padded_1d || reduce_all
                ? aclnnProd(workspaceAddr, workspaceSize, executor, aclstream)
                : aclnnProdDim(workspaceAddr, workspaceSize, executor, aclstream);
            break;
        }
        default:
        {
            LOGir << "no such reduce!!";
            exit(-1);
        }
        }
        syncRun();
        return;
    }
}
