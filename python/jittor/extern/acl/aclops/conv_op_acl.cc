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
#include "conv_op_acl.h"

namespace jittor
{
    Conv2dOpRunner::Conv2dOpRunner() : BaseOpRunner("Conv2d")
    {
        use_nchw = true;
    }

    void Conv2dOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        // for conv
        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *outPads = nullptr;
        aclIntArray *dilations = nullptr;
        auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
        strides = aclCreateIntArray(attr->convStrides.data(), 2);
        pads = aclCreateIntArray(attr->convPads.data(), 2);
        outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
        dilations = aclCreateIntArray(attr->convDilations.data(), 2);

        aclTensor *bias = nullptr;

        auto input_num = in_.size();
        if (input_num == 3)
            bias = inputTensors[2];

        ret = aclnnConvolutionGetWorkspaceSize(inputTensors[0], inputTensors[1], bias, strides, pads, dilations, false, outPads, attr->group, outputTensors[0], 0, &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnConvolution(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnConvolution failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(outPads);
        aclDestroyIntArray(dilations);
        return;
    }


    Conv2dBackwardOpRunner::Conv2dBackwardOpRunner() : BaseOpRunner("Conv2dBackward")
    {
        use_nchw = true;
    }

    void Conv2dBackwardOpRunner::setupOutputDesc()
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

        for (int idx = 0; idx < 2; idx++)
        {
            outputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
        // biasgrad nd format
        {
            outputTensors.push_back(nullptr);
            auto ret = CreateAclTensor(outputShapes[2], out_[2]->mem_ptr, out_[2]->size, get_dtype(out_[2]->dtype()), &outputTensors[2], false);
            CHECK_RET(ret == ACL_SUCCESS, return);
        }
    }

    void Conv2dBackwardOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        // for conv
        aclIntArray *strides = nullptr;
        aclIntArray *pads = nullptr;
        aclIntArray *outPads = nullptr;
        aclIntArray *dilations = nullptr;
        auto attr = dynamic_cast<ConvAttr *>(op_attr.get());
        strides = aclCreateIntArray(attr->convStrides.data(), 2);
        pads = aclCreateIntArray(attr->convPads.data(), 2);
        outPads = aclCreateIntArray(attr->convOutPads.data(), 2);
        dilations = aclCreateIntArray(attr->convDilations.data(), 2);
        bool outputMask[3] = {true, true, true};
        auto input_num = in_.size();
        if (input_num == 3)
        {
            outputMask[2] = false;
        }
        aclBoolArray *outMask = aclCreateBoolArray(outputMask, 3);
        auto biasSizes = aclCreateIntArray(&outputShapes[2][0], outputShapes[2].size());
        ret = aclnnConvolutionBackwardGetWorkspaceSize(inputTensors[0], inputTensors[1], inputTensors[2], biasSizes, strides, pads, dilations, false, outPads, attr->group, outMask, 0, outputTensors[0], outputTensors[1], outputTensors[2], &workspaceSize, &executor);

        checkRet(ret);

        if (workspaceSize > 0)
        {
            mallocWorkSpace(workspaceSize);
        }

        ret = aclnnConvolutionBackward(workspaceAddr, workspaceSize, executor, aclstream);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnConvolutionBackward failed. ERROR: %d\n", name.c_str(), ret); return);

        syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(outPads);
        aclDestroyIntArray(dilations);
        return;
    }
}