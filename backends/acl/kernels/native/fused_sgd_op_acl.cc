#include <aclnnop/aclnn_fused_sgd.h>

#include <vector>

#include "acl_jittor.h"
#include "fused_sgd_op_acl.h"
#include "core/var.h"

namespace jittor
{
    FusedSgdOpRunner::FusedSgdOpRunner() : BaseOpRunner("FusedSgd")
    {
    }

    void FusedSgdOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<FusedSgdAttr *>(op_attr.get());
        CHECK(attr != nullptr);
        const int64_t count = attr->tensorCount;
        CHECK(count > 0);
        // inputs: parameters, velocities, gradients; outputs: new parameters,
        // new velocities. The outputs share storage with their inputs, so the
        // copies below are normally skipped.
        CHECK(inputTensors.size() == size_t(count * 3));
        CHECK(outputTensors.size() == size_t(count * 2));

        for (int64_t slot = 0; slot < count * 2; ++slot)
        {
            CHECK(in_[slot]->size == out_[slot]->size);
            if (in_[slot]->mem_ptr == out_[slot]->mem_ptr)
                continue;
            ret = aclrtMemcpyAsync(out_[slot]->mem_ptr, out_[slot]->size,
                                   in_[slot]->mem_ptr, in_[slot]->size,
                                   ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
            if (ret != ACL_SUCCESS)
                throw std::runtime_error("fused SGD D2D copy failed: " + acl_error_to_string(ret));
        }

        std::vector<aclTensor *> params(outputTensors.begin(), outputTensors.begin() + count);
        std::vector<aclTensor *> velocities(outputTensors.begin() + count, outputTensors.end());
        std::vector<aclTensor *> grads(inputTensors.begin() + count * 2, inputTensors.end());

        aclTensorList *paramList = aclCreateTensorList(params.data(), params.size());
        aclTensorList *velocityList = aclCreateTensorList(velocities.data(), velocities.size());
        aclTensorList *gradList = aclCreateTensorList(grads.data(), grads.size());
        if (!paramList || !velocityList || !gradList)
        {
            if (paramList) aclDestroyTensorList(paramList);
            if (velocityList) aclDestroyTensorList(velocityList);
            if (gradList) aclDestroyTensorList(gradList);
            LOGf << name << ": fused SGD tensor list creation failed";
        }

        ret = aclnnFusedSgdGetWorkspaceSize(
            paramList, gradList, velocityList, nullptr,
            attr->weightDecay, attr->momentum, attr->lr, attr->dampening,
            attr->nesterov, attr->maximize, attr->isFirstStep,
            &workspaceSize, &executor);
        launch(ret, aclnnFusedSgd, true);

        aclDestroyTensorList(paramList);
        aclDestroyTensorList(velocityList);
        aclDestroyTensorList(gradList);
    }
}
