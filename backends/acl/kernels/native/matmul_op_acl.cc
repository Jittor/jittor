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
#include "matmul_op_acl.h"

namespace jittor
{
    MatMulOpRunner::MatMulOpRunner() : BaseOpRunner("MatMul")
    {
    }
    void MatMulOpRunner::setupInputDesc()
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
            if ((jt_name == "matmul_trans_1" && idx == 1) || (jt_name == "matmul_trans_0" && idx == 0))
            {
                auto ret = CreateFakeTransAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw, in_[idx]);
                if (ret != ACL_SUCCESS) LOGf << name << ": transposed input tensor creation failed. ERROR:" << ret;
            }
            else
            {
                auto ret = CreateAclTensor(inputShapes[idx], in_[idx]->mem_ptr, in_[idx]->size, get_dtype(in_[idx]->dtype()), &inputTensors[idx], use_nchw, in_[idx]);
                if (ret != ACL_SUCCESS) LOGf << name << ": input tensor creation failed. ERROR:" << ret;
            }
        }
    }
    void MatMulOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        acl_check_fp32_cube(cube_math_type, in_[0]->dtype(), "MatMul");
        ret = aclnnMatmulGetWorkspaceSize(inputTensors[0], inputTensors[1], outputTensors[0], cube_math_type, &workspaceSize, &executor);
        launch(ret, aclnnMatmul, true);
    }

    PackedMatMulOpRunner::PackedMatMulOpRunner() : BaseOpRunner("PackedMatMul") {}

    void PackedMatMulOpRunner::setupInputDesc()
    {
        CHECK(in_.size() == 2 && out_.size() == 1);
        auto *x = in_[0], *weight = in_[1], *y = out_[0];
        CHECK(x->shape.size() >= 1 && y->shape.size() == x->shape.size());
        CHECK(weight->shape.size() == 4);
        CHECK(x->dtype() == ns_float16 && weight->dtype() == ns_float16
              && y->dtype() == ns_float16);
        const int64 k = x->shape[x->shape.size()-1], n = y->shape[y->shape.size()-1];
        CHECK(k > 0 && n > 0);
        CHECK(x->is_contiguous() && weight->is_contiguous() && y->is_contiguous());
        CHECK(x->num / k == y->num / n);
        CHECK(weight->shape[0] == (k - 1) / 16 + 1
              && weight->shape[1] == (n - 1) / 16 + 1
              && weight->shape[2] == 16 && weight->shape[3] == 16);
        inputShapes.resize(2);
        // Flatten only in the CANN descriptor. The graph retains its ND
        // shape, and its dense-input boundary has materialized storage views.
        inputShapes[0] = {x->num / k, k};
        inputShapes[1] = {k, n};
        inputTensors.resize(2, nullptr);
        auto status = CreateAclTensor(inputShapes[0], x->mem_ptr, x->size,
            ACL_FLOAT16, &inputTensors[0], false, x);
        CHECK(status == ACL_SUCCESS) << "PackedMatMul input descriptor failed: " << status;
        // NZ packs the original weight [N,K]; [1,K] describes its logical
        // transpose without transposing or repacking the stored coefficients.
        std::vector<int64_t> strides{1, k};
        std::vector<int64_t> storage{weight->shape[0], weight->shape[1], 16, 16};
        inputTensors[1] = aclCreateTensor(inputShapes[1].data(), 2, ACL_FLOAT16,
            strides.data(), 0, ACL_FORMAT_FRACTAL_NZ,
            storage.data(), storage.size(), weight->mem_ptr);
        CHECK(inputTensors[1] != nullptr) << "PackedMatMul NZ descriptor failed";
    }

    void PackedMatMulOpRunner::setupOutputDesc()
    {
        auto *y = out_[0];
        const int64 n = y->shape[y->shape.size()-1];
        outputShapes.resize(1);
        outputShapes[0] = {y->num / n, n};
        outputTensors.resize(1, nullptr);
        auto status = AcquireAclTensor(scratch->descriptors, outputShapes[0],
            y->mem_ptr, y->size, ACL_FLOAT16, &outputTensors[0], false, y);
        CHECK(status == ACL_SUCCESS) << "PackedMatMul output descriptor failed: " << status;
    }

    void PackedMatMulOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        ret = aclnnMatmulGetWorkspaceSize(inputTensors[0], inputTensors[1],
            outputTensors[0], 0, &workspaceSize, &executor);
        launch(ret, aclnnMatmul, true);
    }
}
