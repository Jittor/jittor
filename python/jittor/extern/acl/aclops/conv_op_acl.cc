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
    ConvOpRunner::ConvOpRunner() : BaseOpRunner("Conv2d")
    {
        use_nchw = true;
    }
        
    void ConvOpRunner::executeOp(std::unordered_map<string, AclOpFunctions>::iterator &it)
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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("%s: aclnnxxx failed. ERROR: %d\n", name.c_str(), ret); return);

        // syncRun();

        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(outPads);
        aclDestroyIntArray(dilations);
        return;
    }
}