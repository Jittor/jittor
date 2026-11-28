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
#include "aclnnop/aclnn_prod.h"
#include <aclnnop/aclnn_histc.h>
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
        const size_t input_num = in_.size();
        inputShapes.resize(input_num);
        inputTensors.assign(input_num, nullptr);
        for (size_t idx = 0; idx < input_num; idx++)
        {
            Var *v = in_[idx];
            auto &shape = inputShapes[idx];
            if (idx == 0 && v->shape.size() == 1)
            {
                // ACL reduce kernels (aclnnAmax/Amin/...) reject 1-D inputs with
                // ERROR 161002. Pad a 1-D input to 2-D (1, N); axes are shifted by
                // +1 in setupOutputDesc so the reduction result is unchanged.
                shape.resize(2);
                shape[0] = 1;
                shape[1] = v->shape[0];
                input_padded_1d = true;
            }
            else
            {
                shape.resize(v->shape.size());
                for (int j = 0; j < v->shape.size(); j++)
                    shape[j] = v->shape[j];
            }
            auto ret = AcquireAclTensor(scratch->descriptors, shape, v->mem_ptr, v->size, get_dtype(v->dtype()), &inputTensors[idx], use_nchw, v);
            if (ret != ACL_SUCCESS) LOGf << name << ": input tensor creation failed. ERROR:" << ret;
        }
    }

    void ReduceOpRunner::setupOutputDesc()
    {
        const size_t output_num = out_.size();
        outputShapes.resize(output_num);
        outputTensors.assign(output_num, nullptr);
        for (size_t output_idx = 0; output_idx < output_num; output_idx++)
        {
            auto &shape = outputShapes[output_idx];
            const auto &vshape = out_[output_idx]->shape;
            shape.resize(vshape.size());
            for (int j = 0; j < vshape.size(); j++)
                shape[j] = vshape[j];
        }

        size_t orig_axes_size = 0;
        if (use_inline_attr)
        {
            attr = nullptr;
            keepdims = inline_keepdims;
            shifted_axes_.assign(inline_axes, inline_axes + inline_axes_len);
            orig_axes_size = static_cast<size_t>(inline_axes_len);
        }
        else
        {
            attr = dynamic_cast<ReduceAttr *>(op_attr.get());
            keepdims = attr->keepdims;
            shifted_axes_.assign(attr->axes.begin(), attr->axes.end());
            orig_axes_size = attr->axes.size();
        }
        // A 1-D reduce is always a full reduce producing a single element.
        // When the input was padded to (1, N) we reduce over all padded axes
        // and force a scalar output (keepdims off): ACL rejects the (1,)/keepdims
        // combination with ERROR 161002 but accepts a true scalar.
        if (input_padded_1d)
        {
            shifted_axes_.clear();
            for (int64_t ax = 0; ax < (int64_t)inputShapes[0].size(); ax++)
                shifted_axes_.push_back(ax);
            keepdims = false;
        }
        dim = aclCreateIntArray(shifted_axes_.data(), shifted_axes_.size());

        // Jittor's broadcast backward may keep only some reduced axes. CANN
        // accepts one keepdims boolean, so describe its canonical result over
        // the same flat output storage; retained singleton axes change no data.
        outputShapes[0].clear();
        for (size_t axis = 0; axis < inputShapes[0].size(); ++axis)
        {
            const bool reduced = std::find(shifted_axes_.begin(),
                shifted_axes_.end(), axis) != shifted_axes_.end();
            if (!reduced)
                outputShapes[0].push_back(inputShapes[0][axis]);
            else if (keepdims)
                outputShapes[0].push_back(1);
        }

        if (op_idx <= 13)
        {
            // Sum/mean/extrema require the retained singleton axes when
            // keepdims is true. The full-product API has no keepdims argument
            // and always consumes a scalar descriptor over the same storage.
            if (input_padded_1d ||
                (orig_axes_size == static_cast<size_t>(in_[0]->shape.size()) &&
                 (!keepdims || op_idx == 13)))
                outputShapes[0] = {};
        }

        for (size_t idx = 0; idx < output_num; idx++)
        {
            auto ret = AcquireAclTensor(scratch->descriptors, outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size, get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw, out_[idx]);
            if (ret != ACL_SUCCESS) LOGf << name << ": output tensor creation failed. ERROR:" << ret;
        }
    }

    void ReduceOpRunner::executeOp(AclOpRegistry::const_iterator &it)
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
            const bool reduce_all = (use_inline_attr ? static_cast<size_t>(inline_axes_len) : attr->axes.size()) == static_cast<size_t>(in_[0]->shape.size());
            if (input_padded_1d || reduce_all)
            {
                ret = aclnnProdGetWorkspaceSize(
                    inputTensors[0], get_dtype(out_[0]->dtype()),
                    outputTensors[0], &workspaceSize, &executor);
                launch(ret, aclnnProd, true);
            }
            else if (shifted_axes_.size() == 1)
            {
                ret = aclnnProdDimGetWorkspaceSize(
                    inputTensors[0], shifted_axes_[0], keepdims,
                    get_dtype(out_[0]->dtype()), outputTensors[0],
                    &workspaceSize, &executor);
                launch(ret, aclnnProdDim, true);
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
                        if (ret != ACL_SUCCESS) LOGf << name << ": product intermediate allocation failed. ERROR:" << ret;
                        intermediate_buffers.push_back(buffer);

                        next_tensor = nullptr;
                        ret = CreateAclTensor(
                            next_shape, buffer, buffer_size,
                            get_dtype(out_[0]->dtype()), &next_tensor, use_nchw);
                        if (ret != ACL_SUCCESS) LOGf << name << ": product intermediate tensor creation failed. ERROR:" << ret;
                        intermediate_tensors.push_back(next_tensor);
                    }

                    ret = aclnnProdDimGetWorkspaceSize(
                        current_tensor, axis, keepdims,
                        get_dtype(out_[0]->dtype()), next_tensor,
                        &workspaceSize, &executor);
                    launch(ret, aclnnProdDim, false);

                    current_shape = std::move(next_shape);
                    current_tensor = next_tensor;
                }

                ret = aclrtSynchronizeStream(aclstream);
                if (ret != ACL_SUCCESS) LOGf << name << ": product intermediate synchronization failed. ERROR:" << ret;
                for (aclTensor *tensor : intermediate_tensors)
                    aclDestroyTensor(tensor);
                for (void *buffer : intermediate_buffers)
                    aclrtFree(buffer);
                // The staged path already synchronised unconditionally above,
                // before the intermediates were freed. This keeps the extra
                // diagnostic synchronisation the hand-rolled tail also ran here.
                syncRun();
            }
            break;
        }
        default:
        {
            LOGir << "no such reduce!!";
            exit(-1);
        }
        }
        return;
    }

    HistcOpRunner::HistcOpRunner() : BaseOpRunner("Histc")
    {
    }

    void HistcOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        float fmin = static_cast<float>(min_val);
        float fmax = static_cast<float>(max_val);
        aclScalar *min_scalar = aclCreateScalar(&fmin, ACL_FLOAT);
        aclScalar *max_scalar = aclCreateScalar(&fmax, ACL_FLOAT);
        ret = aclnnHistcGetWorkspaceSize(
            inputTensors[0], bins, min_scalar, max_scalar,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnHistc, true);
        aclDestroyScalar(min_scalar);
        aclDestroyScalar(max_scalar);
    }
}
