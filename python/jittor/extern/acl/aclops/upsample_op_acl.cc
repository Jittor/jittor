#include "upsample_op_acl.h"

namespace jittor
{
    UpsampleNearest2dOpRunner::UpsampleNearest2dOpRunner()
        : BaseOpRunner("UpsampleNearest2d")
    {
        use_nchw = true;
    }

    void UpsampleNearest2dOpRunner::executeOp(
        std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<UpsampleNearest2dAttr *>(op_attr.get());
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> outputSize(
            aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size()),
            aclDestroyIntArray);
        CHECK_RET(outputSize != nullptr,
                  LOG_PRINT("%s: aclCreateIntArray failed.\n", name.c_str()); return);
        ret = aclnnUpsampleNearest2dGetWorkspaceSize(
            inputTensors[0], outputSize.get(), outputTensors[0],
            &workspaceSize, &executor);
        checkRet(ret);
        if (ret != ACL_SUCCESS)
            return;

        launch(ret, aclnnUpsampleNearest2d, true);
    }

    UpsampleNearest2dBackwardOpRunner::UpsampleNearest2dBackwardOpRunner()
        : BaseOpRunner("UpsampleNearest2dBackward")
    {
        use_nchw = true;
    }

    void UpsampleNearest2dBackwardOpRunner::executeOp(
        std::unordered_map<string, AclOpFunctions>::iterator &it)
    {
        auto attr = dynamic_cast<UpsampleNearest2dAttr *>(op_attr.get());
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> outputSize(
            aclCreateIntArray(attr->outputSize.data(), attr->outputSize.size()),
            aclDestroyIntArray);
        std::unique_ptr<aclIntArray, decltype(&aclDestroyIntArray)> inputSize(
            aclCreateIntArray(attr->inputSize.data(), attr->inputSize.size()),
            aclDestroyIntArray);
        CHECK_RET(outputSize != nullptr && inputSize != nullptr,
                  LOG_PRINT("%s: aclCreateIntArray failed.\n", name.c_str()); return);
        ret = aclnnUpsampleNearest2dBackwardGetWorkspaceSize(
            inputTensors[0], outputSize.get(), inputSize.get(), 0.0, 0.0,
            outputTensors[0], &workspaceSize, &executor);
        checkRet(ret);
        if (ret != ACL_SUCCESS)
            return;

        launch(ret, aclnnUpsampleNearest2dBackward, true);
    }
}
