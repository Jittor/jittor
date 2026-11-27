#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include "core/common.h"
#include "core/op.h"
#include "core/var.h"
#include "acl_jittor.h"
#include "aclnn/aclnn.h"
#include "ops/composite/mapped_unary_op.h"
#include "direct_mapped_op_acl.h"

namespace jittor
{
    namespace
    {
        static void setupDirectNormDescs(
            const string &name,
            std::vector<aclTensor *> &pool,
            const std::vector<Var *> &vars,
            std::vector<std::vector<int64_t>> &shapes,
            std::vector<aclTensor *> &tensors,
            size_t nchw_prefix)
        {
            const size_t n = vars.size();
            shapes.resize(n);
            for (size_t i = 0; i < n; ++i)
            {
                auto &s = shapes[i];
                const auto &vs = vars[i]->shape;
                s.resize(vs.size());
                for (size_t j = 0; j < vs.size(); ++j)
                    s[j] = vs[j];
            }
            tensors.resize(n, nullptr);
            for (size_t i = 0; i < n; ++i)
            {
                tensors[i] = nullptr;
                auto ret = AcquireAclTensor(
                    pool, shapes[i], vars[i]->mem_ptr, vars[i]->size,
                    get_dtype(vars[i]->dtype()), &tensors[i],
                    i < nchw_prefix, vars[i]);
                if (ret != ACL_SUCCESS)
                    LOGf << name << ": normalization tensor creation failed. ERROR:" << ret;
            }
        }
    } // namespace

    DirectConv2dOpRunner::DirectConv2dOpRunner() : BaseOpRunner("Conv2d")
    {
        use_nchw = true;
    }

    void DirectConv2dOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        int64_t stride_arr[2] = {stride_h, stride_w};
        int64_t pad_arr[2] = {pad_h, pad_w};
        int64_t dil_arr[2] = {dil_h, dil_w};
        int64_t out_pad_arr[2] = {0, 0};
        aclIntArray *strides = aclCreateIntArray(stride_arr, 2);
        aclIntArray *pads = aclCreateIntArray(pad_arr, 2);
        aclIntArray *dilations = aclCreateIntArray(dil_arr, 2);
        aclIntArray *out_pads = aclCreateIntArray(out_pad_arr, 2);
        aclTensor *bias = in_.size() == 3 ? inputTensors[2] : nullptr;
        acl_check_fp32_cube(cube_math_type, in_[0]->dtype(), "Conv2d");
        ret = aclnnConvolutionGetWorkspaceSize(
            inputTensors[0], inputTensors[1], bias,
            strides, pads, dilations, false, out_pads, groups,
            outputTensors[0], cube_math_type, &workspaceSize, &executor);
        launch(ret, aclnnConvolution, true);
        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(dilations);
        aclDestroyIntArray(out_pads);
    }

    DirectConv2dBackwardOpRunner::DirectConv2dBackwardOpRunner()
        : BaseOpRunner("Conv2dBackward")
    {
        use_nchw = true;
    }

    void DirectConv2dBackwardOpRunner::setupOutputDesc()
    {
        const size_t output_num = out_.size();
        outputShapes.resize(output_num);
        for (size_t i = 0; i < output_num; ++i)
        {
            auto &s = outputShapes[i];
            const auto &vs = out_[i]->shape;
            s.resize(vs.size());
            for (size_t j = 0; j < vs.size(); ++j)
                s[j] = vs[j];
        }
        outputTensors.resize(3, nullptr);
        for (size_t i = 0; i < 2; ++i)
        {
            outputTensors[i] = nullptr;
            auto st = AcquireAclTensor(
                scratch->descriptors, outputShapes[i],
                out_[i]->mem_ptr, out_[i]->size,
                get_dtype(out_[i]->dtype()), &outputTensors[i],
                use_nchw, out_[i]);
            if (st != ACL_SUCCESS)
                LOGf << name << ": convolution gradient tensor creation failed. ERROR:" << st;
        }
        if (output_num == 3)
        {
            outputTensors[2] = nullptr;
            auto st = AcquireAclTensor(
                scratch->descriptors, outputShapes[2],
                out_[2]->mem_ptr, out_[2]->size,
                get_dtype(out_[2]->dtype()), &outputTensors[2],
                false, out_[2]);
            if (st != ACL_SUCCESS)
                LOGf << name << ": bias gradient tensor creation failed. ERROR:" << st;
        }
        else
        {
            outputTensors[2] = nullptr;
        }
    }

    void DirectConv2dBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        int64_t stride_arr[2] = {stride_h, stride_w};
        int64_t pad_arr[2] = {pad_h, pad_w};
        int64_t dil_arr[2] = {dil_h, dil_w};
        int64_t out_pad_arr[2] = {0, 0};
        aclIntArray *strides = aclCreateIntArray(stride_arr, 2);
        aclIntArray *pads = aclCreateIntArray(pad_arr, 2);
        aclIntArray *dilations = aclCreateIntArray(dil_arr, 2);
        aclIntArray *out_pads = aclCreateIntArray(out_pad_arr, 2);
        const bool has_bias = out_.size() == 3;
        bool output_mask[3] = {out_[0]->is_result_needed(), true, has_bias};
        aclBoolArray *out_mask = aclCreateBoolArray(output_mask, 3);
        int64_t bias_len = in_[2]->shape[0];
        aclIntArray *bias_sizes = aclCreateIntArray(&bias_len, 1);
        acl_check_fp32_cube(cube_math_type, in_[0]->dtype(), "Conv2dBackward");
        ret = aclnnConvolutionBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            bias_sizes, strides, pads, dilations, false, out_pads, groups,
            out_mask, cube_math_type,
            outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        launch(ret, aclnnConvolutionBackward, true);
        aclDestroyIntArray(strides);
        aclDestroyIntArray(pads);
        aclDestroyIntArray(dilations);
        aclDestroyIntArray(out_pads);
        aclDestroyIntArray(bias_sizes);
        aclDestroyBoolArray(out_mask);
    }

    DirectBatchNormOpRunner::DirectBatchNormOpRunner() : BaseOpRunner("BatchNorm")
    {
    }

    void DirectBatchNormOpRunner::setupInputDesc()
    {
        setupDirectNormDescs(name, scratch->descriptors, in_, inputShapes, inputTensors, 1);
    }

    void DirectBatchNormOpRunner::setupOutputDesc()
    {
        setupDirectNormDescs(name, scratch->descriptors, out_, outputShapes, outputTensors, 1);
    }

    void DirectBatchNormOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        aclTensor *running_mean_tensor = inputTensors[3];
        aclTensor *running_var_tensor = inputTensors[4];
        if (outputTensors.size() >= 5)
        {
            if (out_[3]->mem_ptr != in_[3]->mem_ptr && out_[3]->size > 0)
            {
                auto st = aclrtMemcpyAsync(
                    out_[3]->mem_ptr, out_[3]->size, in_[3]->mem_ptr, out_[3]->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << name << ": running_mean D2D copy failed. ERROR:" << st;
            }
            if (out_[4]->mem_ptr != in_[4]->mem_ptr && out_[4]->size > 0)
            {
                auto st = aclrtMemcpyAsync(
                    out_[4]->mem_ptr, out_[4]->size, in_[4]->mem_ptr, out_[4]->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (st != ACL_SUCCESS)
                    LOGf << name << ": running_var D2D copy failed. ERROR:" << st;
            }
            running_mean_tensor = outputTensors[3];
            running_var_tensor = outputTensors[4];
        }
        ret = aclnnBatchNormGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            running_mean_tensor, running_var_tensor,
            is_train, momentum, eps,
            outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        launch(ret, aclnnBatchNorm, true);
    }

    DirectBatchNormBackwardOpRunner::DirectBatchNormBackwardOpRunner()
        : BaseOpRunner("BatchNormBackward")
    {
    }

    void DirectBatchNormBackwardOpRunner::setupInputDesc()
    {
        setupDirectNormDescs(name, scratch->descriptors, in_, inputShapes, inputTensors, 2);
    }

    void DirectBatchNormBackwardOpRunner::setupOutputDesc()
    {
        setupDirectNormDescs(name, scratch->descriptors, out_, outputShapes, outputTensors, 1);
    }

    void DirectBatchNormBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        bool output_mask[3] = {true, true, true};
        aclBoolArray *out_mask = aclCreateBoolArray(output_mask, 3);
        ret = aclnnBatchNormBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], inputTensors[2],
            inputTensors[3], inputTensors[4], inputTensors[5], inputTensors[6],
            is_train, eps, out_mask,
            outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        launch(ret, aclnnBatchNormBackward, true);
        aclDestroyBoolArray(out_mask);
    }

    DirectLayerNormOpRunner::DirectLayerNormOpRunner() : BaseOpRunner("LayerNorm")
    {
    }

    void DirectLayerNormOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        const auto &shape = in_[0]->shape;
        const int rank = static_cast<int>(shape.size());
        int64_t norm_dims[16];
        for (int i = 0; i < normalized_ndim; ++i)
            norm_dims[i] = shape[rank - normalized_ndim + i];
        aclIntArray *norm_shape = aclCreateIntArray(norm_dims, normalized_ndim);
        ret = aclnnLayerNormGetWorkspaceSize(
            inputTensors[0], norm_shape, inputTensors[1], inputTensors[2], eps,
            outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        launch(ret, aclnnLayerNorm, true);
        aclDestroyIntArray(norm_shape);
    }

    DirectLayerNormBackwardOpRunner::DirectLayerNormBackwardOpRunner()
        : BaseOpRunner("LayerNormBackward")
    {
    }

    void DirectLayerNormBackwardOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // inputs: dout(0), x(1), mean(2), rstd(3), weight(4), bias(5)
        // outputs: dx(0), dw(1), db(2)
        const auto &shape = in_[1]->shape;
        const int rank = static_cast<int>(shape.size());
        int64_t norm_dims[16];
        for (int i = 0; i < normalized_ndim; ++i)
            norm_dims[i] = shape[rank - normalized_ndim + i];
        aclIntArray *norm_shape = aclCreateIntArray(norm_dims, normalized_ndim);
        bool output_mask[3] = {true, true, true};
        aclBoolArray *out_mask = aclCreateBoolArray(output_mask, 3);
        ret = aclnnLayerNormBackwardGetWorkspaceSize(
            inputTensors[0], inputTensors[1], norm_shape,
            inputTensors[2], inputTensors[3], inputTensors[4], inputTensors[5],
            out_mask, outputTensors[0], outputTensors[1], outputTensors[2],
            &workspaceSize, &executor);
        launch(ret, aclnnLayerNormBackward, true);
        aclDestroyIntArray(norm_shape);
        aclDestroyBoolArray(out_mask);
    }

    DirectUnaryGradOpRunner::DirectUnaryGradOpRunner()
        : BaseOpRunner("LeakyReLUBackward")
    {
    }

    void DirectUnaryGradOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        if (kind == MappedUnaryOp::kRelu)
        {
            float slope_val = 0.0f;
            aclScalar *slope = aclCreateScalar(&slope_val, aclDataType::ACL_FLOAT);
            ret = aclnnLeakyReluBackwardGetWorkspaceSize(
                inputTensors[0], inputTensors[1], slope, false,
                outputTensors[0], &workspaceSize, &executor);
            launch(ret, aclnnLeakyReluBackward, true);
            aclDestroyScalar(slope);
            return;
        }
        static char exact[] = "none";
        static char tanh_mode[] = "tanh";
        char *approx = approximate ? tanh_mode : exact;
        ret = aclnnGeluBackwardV2GetWorkspaceSize(
            inputTensors[0], inputTensors[1], approx,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnGeluBackwardV2, true);
    }
}
