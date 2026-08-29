#include <acl/acl.h>
#include <acl/acl_op_compiler.h>
#include <memory>
#include <stdexcept>
#include "acl_jittor.h"
#include "aclnn/aclnn.h"
#include "truth_reduce_op_acl.h"
#include "var.h"

namespace jittor
{
    TruthReduceOpRunner::TruthReduceOpRunner(bool reduce_all)
        : BaseOpRunner(reduce_all ? "All" : "Any"),
          reduce_all(reduce_all),
          attr(nullptr)
    {
        use_nchw = false;
    }

    void TruthReduceOpRunner::setupOutputDesc()
    {
        attr = dynamic_cast<ReduceAttr *>(op_attr.get());
        if (!attr)
            throw std::runtime_error("truth reduction requires ReduceAttr");

        for (auto *output : out_)
        {
            std::vector<int64_t> shape;
            for (int axis = 0; axis < output->shape.size(); ++axis)
                shape.push_back(output->shape[axis]);
            // Jittor exposes a one-element Var for a full reduction, whereas
            // aclnnAll/aclnnAny require a scalar descriptor when keepdim=false.
            if (!attr->keepdims && attr->axes.size() == in_[0]->shape.size())
                shape.clear();
            outputShapes.push_back(shape);
        }

        for (int idx = 0; idx < out_.size(); ++idx)
        {
            outputTensors.push_back(nullptr);
            auto status = CreateAclTensor(
                outputShapes[idx], out_[idx]->mem_ptr, out_[idx]->size,
                get_dtype(out_[idx]->dtype()), &outputTensors[idx], use_nchw);
            if (status != ACL_SUCCESS)
                throw std::runtime_error("failed to create truth reduction output");
        }
    }

    void TruthReduceOpRunner::executeOp(
        std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> dim(
            aclCreateIntArray(attr->axes.data(), attr->axes.size()),
            aclDestroyIntArray);
        if (!dim)
            throw std::runtime_error("failed to create truth reduction axes");

        ret = reduce_all
            ? aclnnAllGetWorkspaceSize(
                  inputTensors[0], dim.get(), attr->keepdims, outputTensors[0],
                  &workspaceSize, &executor)
            : aclnnAnyGetWorkspaceSize(
                  inputTensors[0], dim.get(), attr->keepdims, outputTensors[0],
                  &workspaceSize, &executor);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error(
                reduce_all ? "aclnnAllGetWorkspaceSize failed"
                           : "aclnnAnyGetWorkspaceSize failed");

        if (workspaceSize > 0)
            mallocWorkSpace(workspaceSize);
        ret = reduce_all
            ? aclnnAll(workspaceAddr, workspaceSize, executor, aclstream)
            : aclnnAny(workspaceAddr, workspaceSize, executor, aclstream);
        if (ret != ACL_SUCCESS)
            throw std::runtime_error(
                reduce_all ? "aclnnAll execution failed"
                           : "aclnnAny execution failed");
        syncRun();
    }
}
