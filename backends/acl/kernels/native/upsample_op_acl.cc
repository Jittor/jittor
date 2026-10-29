#include "upsample_op_acl.h"
#include <aclnnop/aclnn_upsample_bicubic_2d.h>

namespace jittor
{
    UpsampleNearest2dOpRunner::UpsampleNearest2dOpRunner()
        : BaseOpRunner("UpsampleNearest2d")
    {
        use_nchw = true;
    }

    void UpsampleNearest2dOpRunner::executeOp(
        AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<UpsampleNearest2dAttr *>(op_attr.get());
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> outputSize(
            aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size()),
            aclDestroyIntArray);
        if (!outputSize) LOGf << name << ": aclCreateIntArray failed";
        ret = aclnnUpsampleNearest2dGetWorkspaceSize(
            inputTensors[0], outputSize.get(), outputTensors[0],
            &workspaceSize, &executor);
        launch(ret, aclnnUpsampleNearest2d, true);
    }

    UpsampleBicubic2dOpRunner::UpsampleBicubic2dOpRunner()
        : BaseOpRunner("UpsampleBicubic2d")
    {
        use_nchw = true;
    }

    void UpsampleBicubic2dOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        // The geometry is shared with nearest resize; this runner admits only
        // align_corners=false, size-based sampling and no antialiasing.
        auto attr = dynamic_cast<UpsampleNearest2dAttr *>(op_attr.get());
        USER_CHECK(attr && inputShapes[0].size() == 4 && attr->outputSize.size() == 2)
            << "ACL bicubic resize requires NCHW input and two output dimensions";
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> outputSize(
            aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size()),
            aclDestroyIntArray);
        if (!outputSize) LOGf << name << ": aclCreateIntArray failed";
        ret = aclnnUpsampleBicubic2dGetWorkspaceSize(
            inputTensors[0], outputSize.get(), false, 0.0, 0.0, outputTensors[0],
            &workspaceSize, &executor);
        launch(ret, aclnnUpsampleBicubic2d, true);
    }

    UpsampleBilinear2dOpRunner::UpsampleBilinear2dOpRunner()
        : BaseOpRunner("UpsampleBilinear2d")
    {
        use_nchw = true;
    }

    void UpsampleBilinear2dOpRunner::executeOp(AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<UpsampleBilinear2dAttr *>(op_attr.get());
        USER_CHECK(attr && inputShapes[0].size() == 4 && attr->outputSize.size() == 2)
            << "ACL bilinear resize requires NCHW input and two output dimensions";
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> outputSize(
            aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size()),
            aclDestroyIntArray);
        if (!outputSize) LOGf << name << ": aclCreateIntArray failed";
        ret = aclnnUpsampleBilinear2dGetWorkspaceSize(
            inputTensors[0], outputSize.get(), attr->alignCorners, 0.0, 0.0,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnUpsampleBilinear2d, true);
    }

    UpsampleNearest2dBackwardOpRunner::UpsampleNearest2dBackwardOpRunner()
        : BaseOpRunner("UpsampleNearest2dBackward")
    {
        use_nchw = true;
    }

    void UpsampleNearest2dBackwardOpRunner::executeOp(
        AclOpRegistry::const_iterator &it)
    {
        auto attr = dynamic_cast<UpsampleNearest2dAttr *>(op_attr.get());
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> outputSize(
            aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size()),
            aclDestroyIntArray);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> inputSize(
            aclCreateIntArray(attr->inputSize.data(), attr->inputSize.size()),
            aclDestroyIntArray);
        if (!outputSize || !inputSize) LOGf << name << ": aclCreateIntArray failed";
        ret = aclnnUpsampleNearest2dBackwardGetWorkspaceSize(
            inputTensors[0], outputSize.get(), inputSize.get(), 0.0, 0.0,
            outputTensors[0], &workspaceSize, &executor);
        launch(ret, aclnnUpsampleNearest2dBackward, true);
    }
}
