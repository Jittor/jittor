#pragma once
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
#include "bmm_op_acl.h"

namespace jittor
{
    BatchMatMulOpRunner::BatchMatMulOpRunner() : BaseOpRunner("BatchMatMul")
    {
    }
    // CANN's batched matmul is rank-3 only, while the attention shapes every
    // transformer writes are rank-4 ([batch, heads, tokens, width]). The extra
    // axes are pure batching over a contiguous buffer, so fold them into the
    // leading dimension when building the descriptor. Doing it here rather
    // than in the Python builder keeps the reshape out of the graph entirely.
    static void collapse_batch_dims(std::vector<int64_t> &shape)
    {
        if (shape.size() <= 3) return;
        int64_t batch = 1;
        for (size_t i = 0; i + 2 < shape.size(); i++) batch *= shape[i];
        shape[0] = batch;
        shape[1] = shape[shape.size() - 2];
        shape[2] = shape[shape.size() - 1];
        shape.resize(3);
    }

    void BatchMatMulOpRunner::setupInputDesc()
    {
        const size_t input_num = in_.size();
        inputShapes.resize(input_num);
        inputTensors.assign(input_num, nullptr);
        for (size_t idx = 0; idx < input_num; idx++)
        {
            Var *v = in_[idx];
            auto &shape = inputShapes[idx];
            shape.resize(v->shape.size());
            for (int j = 0; j < v->shape.size(); j++)
                shape[j] = v->shape[j];
            collapse_batch_dims(shape);

            const bool trans = (jt_name == "bmm_trans_1" && idx == 1) || (jt_name == "bmm_trans_0" && idx == 0);
            if (!v->is_contiguous())
            {
                const int r = v->shape.size();
                int64_t batch = 1, batch_stride = 0;
                for (int i = r - 3; i >= 0; --i)
                {
                    const int64_t d = v->shape[i];
                    if (d > 1)
                    {
                        if (batch == 1) batch_stride = v->storage_stride(i);
                        batch *= d;
                    }
                }
                int64_t dims[3] = {batch, v->shape[r - 2], v->shape[r - 1]};
                if (batch == 1)
                    batch_stride = std::max<int64_t>(1, dims[1] * dims[2]);
                int64_t strides[3] = {batch_stride, v->storage_stride(r - 2), v->storage_stride(r - 1)};
                if (trans)
                {
                    std::swap(dims[1], dims[2]);
                    std::swap(strides[1], strides[2]);
                }
                const int64_t packed = v->storage_span_bytes() / v->dsize();
                auto st = AcquireRawAclTensor(
                    scratch->descriptors, dims, 3, get_dtype(v->dtype()), strides,
                    &packed, 1, v->mem_ptr, &inputTensors[idx]);
                if (st != ACL_SUCCESS || !inputTensors[idx])
                    LOGf << name << ": strided input tensor creation failed";
            }
            else if (trans)
            {
                auto ret = AcquireFakeTransAclTensor(scratch->descriptors, shape, v->mem_ptr, v->size, get_dtype(v->dtype()), &inputTensors[idx], use_nchw, v);
                if (ret != ACL_SUCCESS) LOGf << name << ": transposed input tensor creation failed. ERROR:" << ret;
            }
            else
            {
                auto ret = AcquireAclTensor(scratch->descriptors, shape, v->mem_ptr, v->size, get_dtype(v->dtype()), &inputTensors[idx], use_nchw, v);
                if (ret != ACL_SUCCESS) LOGf << name << ": input tensor creation failed. ERROR:" << ret;
            }
        }
    }
    void BatchMatMulOpRunner::setupOutputDesc()
    {
        const size_t output_num = out_.size();
        outputShapes.resize(output_num);
        outputTensors.assign(output_num, nullptr);
        for (size_t idx = 0; idx < output_num; idx++)
        {
            Var *v = out_[idx];
            auto &shape = outputShapes[idx];
            shape.resize(v->shape.size());
            for (int j = 0; j < v->shape.size(); j++)
                shape[j] = v->shape[j];
            collapse_batch_dims(shape);
            auto ret = AcquireAclTensor(scratch->descriptors, shape, v->mem_ptr, v->size,
                                        get_dtype(v->dtype()), &outputTensors[idx],
                                        use_nchw, v);
            if (ret != ACL_SUCCESS) LOGf << name << ": output tensor creation failed. ERROR:" << ret;
        }
    }

    void BatchMatMulOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        acl_check_fp32_cube(cube_math_type, in_[0]->dtype(), "BatchMatMul");
        ret = aclnnBatchMatMulGetWorkspaceSize(inputTensors[0], inputTensors[1], outputTensors[0], cube_math_type, &workspaceSize, &executor);
        launch(ret, aclnnBatchMatMul, true);
    }
}
