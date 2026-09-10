#include <aclnnop/aclnn_apply_adam_w_v2.h>
#include <memory>

#include "acl_jittor.h"
#include "adamw_op_acl.h"
#include "core/var.h"

namespace jittor
{
    AdamWListOpRunner::AdamWListOpRunner() : BaseOpRunner("AdamWList", Dispatch::Direct)
    {
    }

    void AdamWListOpRunner::executeOp(
        AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<AdamWAttr *>(op_attr.get());
        CHECK(attr != nullptr);
        int64_t count = attr->tensorCount;
        CHECK(count > 0);
        CHECK(inputTensors.size() == count * 4 + 1);
        CHECK(outputTensors.size() == count * 3);

        // CANN's AdamW tiling requires a one-element vector. Native scalar
        // Vars have rank zero, even though they contain the same single value.
        auto* step = in_[count * 4];
        int64_t one = 1;
        std::unique_ptr<aclTensor, decltype(&aclDestroyTensor)> stepTensor(
            aclCreateTensor(&one, 1, get_dtype(step->dtype()), &one, 0,
                ACL_FORMAT_ND, &one, 1, step->mem_ptr), aclDestroyTensor);
        CHECK(stepTensor != nullptr);

        for (int64_t index = 0; index < count; ++index)
        {
            for (int64_t tensor = 0; tensor < 3; ++tensor)
            {
                int64_t position = tensor * count + index;
                CHECK(in_[position]->size == out_[position]->size);
                if (in_[position]->mem_ptr == out_[position]->mem_ptr)
                    continue;
                ret = aclrtMemcpyAsync(
                    out_[position]->mem_ptr, out_[position]->size,
                    in_[position]->mem_ptr, in_[position]->size,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, aclstream);
                if (ret != ACL_SUCCESS)
                    throw std::runtime_error(
                        "fused AdamW D2D copy failed: " +
                        acl_error_to_string(ret));
            }

            ret = aclnnApplyAdamWV2GetWorkspaceSize(
                outputTensors[index], outputTensors[count + index],
                outputTensors[count * 2 + index], nullptr,
                inputTensors[count * 3 + index], stepTensor.get(),
                attr->lr, attr->beta1, attr->beta2, attr->weightDecay,
                attr->eps, false, false, &workspaceSize, &executor);
            launch(ret, aclnnApplyAdamWV2, false);
        }
        syncRun();
    }
}
