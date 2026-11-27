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
#include <aclnnop/aclnn_grouped_matmul_v5.h>
#include <aclnnop/aclnn_addmm.h>
#include "matmul_op_acl.h"

namespace jittor
{
    MatMulOpRunner::MatMulOpRunner() : BaseOpRunner("MatMul")
    {
    }
    void MatMulOpRunner::setupInputDesc()
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
            if ((jt_name == "matmul_trans_1" && idx == 1) || (jt_name == "matmul_trans_0" && idx == 0))
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

    GroupedMatMulOpRunner::GroupedMatMulOpRunner() : BaseOpRunner("GroupedMatMul") {}

    void GroupedMatMulOpRunner::setupInputDesc()
    {
        CHECK(in_.size() == 3 && out_.size() == 1);
        inputShapes.resize(3);
        inputTensors.assign(3, nullptr);
        for (size_t idx = 0; idx < 3; ++idx)
        {
            Var *v = in_[idx];
            auto &shape = inputShapes[idx];
            shape.resize(v->shape.size());
            for (int j = 0; j < v->shape.size(); ++j)
                shape[j] = v->shape[j];
            // CreateAclTensor (non-pooled) for x & weight since aclDestroyTensorList
            // owns and destroys its element descriptors on teardown.
            auto status = CreateAclTensor(
                shape, v->mem_ptr, v->size, get_dtype(v->dtype()),
                &inputTensors[idx], false, v);
            if (status != ACL_SUCCESS)
                LOGf << name << ": input descriptor " << idx << " failed. ERROR:" << status;
        }
    }

    void GroupedMatMulOpRunner::setupOutputDesc()
    {
        outputShapes.resize(1);
        outputTensors.assign(1, nullptr);
        Var *y = out_[0];
        outputShapes[0].resize(y->shape.size());
        for (int j = 0; j < y->shape.size(); ++j)
            outputShapes[0][j] = y->shape[j];
        auto status = CreateAclTensor(
            outputShapes[0], y->mem_ptr, y->size, get_dtype(y->dtype()),
            &outputTensors[0], false, y);
        if (status != ACL_SUCCESS)
            LOGf << name << ": output descriptor failed. ERROR:" << status;
    }

    void GroupedMatMulOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        aclTensor *x_t = inputTensors[0];
        aclTensor *w_t = inputTensors[1];
        aclTensor *o_t = outputTensors[0];
        // Transfer descriptor ownership to the tensor lists so cleanupDesc
        // does not recycle or double-destroy them.
        inputTensors[0] = nullptr;
        inputTensors[1] = nullptr;
        outputTensors[0] = nullptr;
        std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> x_list(
            aclCreateTensorList(&x_t, 1), aclDestroyTensorList);
        std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> w_list(
            aclCreateTensorList(&w_t, 1), aclDestroyTensorList);
        std::unique_ptr<aclTensorList, decltype(&aclDestroyTensorList)> out_list(
            aclCreateTensorList(&o_t, 1), aclDestroyTensorList);
        ret = aclnnGroupedMatmulV5GetWorkspaceSize(
            x_list.get(), w_list.get(),
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            inputTensors[2],
            nullptr, nullptr, nullptr,
            /*splitItem=*/3, /*groupType=*/0, /*groupListType=*/0, /*actType=*/0,
            /*tuningConfigOptional=*/nullptr,
            out_list.get(), nullptr, nullptr,
            &workspaceSize, &executor);
        launch(ret, aclnnGroupedMatmulV5, true);
    }

    AddmmOpRunner::AddmmOpRunner() : BaseOpRunner("Addmm") {}

    void AddmmOpRunner::setupInputDesc()
    {
        CHECK(in_.size() == 3 && out_.size() == 1);
        // in_[0] = bias [N], in_[1] = x [..., K], in_[2] = weight [N, K] (transposed to [K, N])
        Var *bias = in_[0], *x = in_[1], *w = in_[2];
        const int64 k = x->shape[x->shape.size() - 1];
        const int64 rows = x->num / k;
        inputShapes.resize(3);
        inputShapes[0] = {bias->shape[0]};
        inputShapes[1] = {rows, k};
        inputShapes[2] = {w->shape[0], w->shape[1]};
        inputTensors.assign(3, nullptr);
        auto st0 = AcquireAclTensor(scratch->descriptors, inputShapes[0], bias->mem_ptr, bias->size, get_dtype(bias->dtype()), &inputTensors[0], false, nullptr);
        auto st1 = AcquireAclTensor(scratch->descriptors, inputShapes[1], x->mem_ptr, x->size, get_dtype(x->dtype()), &inputTensors[1], false, nullptr);
        auto st2 = AcquireFakeTransAclTensor(scratch->descriptors, inputShapes[2], w->mem_ptr, w->size, get_dtype(w->dtype()), &inputTensors[2], false, w);
        if (st0 != ACL_SUCCESS || st1 != ACL_SUCCESS || st2 != ACL_SUCCESS)
            LOGf << name << ": addmm input descriptor creation failed. ERROR:" << st0 << st1 << st2;
    }

    void AddmmOpRunner::setupOutputDesc()
    {
        Var *y = out_[0];
        const int64 n = y->shape[y->shape.size() - 1];
        const int64 rows = y->num / n;
        outputShapes.resize(1);
        outputShapes[0] = {rows, n};
        outputTensors.assign(1, nullptr);
        auto st = AcquireAclTensor(scratch->descriptors, outputShapes[0], y->mem_ptr, y->size, get_dtype(y->dtype()), &outputTensors[0], false, nullptr);
        if (st != ACL_SUCCESS)
            LOGf << name << ": addmm output descriptor creation failed. ERROR:" << st;
    }

    void AddmmOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        acl_check_fp32_cube(cube_math_type, in_[1]->dtype(), "Addmm");
        float one = 1.0f;
        aclScalar *beta = aclCreateScalar(&one, ACL_FLOAT);
        aclScalar *alpha = aclCreateScalar(&one, ACL_FLOAT);
        ret = aclnnAddmmGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            beta, alpha, outputTensors[0],
            static_cast<int8_t>(cube_math_type),
            &workspaceSize, &executor);
        launch(ret, aclnnAddmm, true);
        aclDestroyScalar(beta);
        aclDestroyScalar(alpha);
    }
}
