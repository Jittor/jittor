#include <aclnnop/aclnn_foreach_add_list.h>
#include <aclnnop/aclnn_foreach_add_scalar.h>
#include <aclnnop/aclnn_foreach_addcdiv_scalar.h>
#include <aclnnop/aclnn_foreach_addcmul_scalar.h>
#include <aclnnop/aclnn_foreach_mul_scalar.h>
#include <aclnnop/aclnn_foreach_sqrt.h>

#include "acl_foreach_coefficients.h"
#include "acl_jittor.h"
#include "foreach_op_acl.h"

namespace jittor
{
    // The in-class initialiser is not a definition, and std::min takes its
    // arguments by reference.
    const int64_t ForeachOpRunner::list_limit;

    ForeachOpRunner::~ForeachOpRunner()
    {
        for (auto *list : lists)
            aclDestroyTensorList(list);
        for (auto *value : coefficients)
            aclDestroyTensor(value);
    }

    void ForeachOpRunner::stageCoefficients(const std::vector<float> &values)
    {
        coefficients.assign(values.size(), nullptr);
        acl_stage_foreach_coefficients(values.data(), int(values.size()),
                                       coefficients.data());
    }

    const aclTensor *ForeachOpRunner::coefficient(size_t index) const
    {
        if (index >= coefficients.size())
            LOGf << name << ": foreach coefficient" << index
                 << "was never staged;" << coefficients.size() << "are available";
        return coefficients[index];
    }

    const aclTensorList *ForeachOpRunner::makeList(
        std::vector<aclTensor *> &source, size_t begin, size_t count)
    {
        if (begin + count > source.size())
            LOGf << name << ": foreach tensor list [" << begin << "," << begin + count
                 << ") runs past the" << source.size() << "descriptors the runner holds";
        // A tensor list owns the descriptors it is handed: aclDestroyTensorList
        // frees them, so leaving them in place for cleanupDesc to free as well
        // is a double free. Hand ownership over and clear the slots; a cleared
        // slot is what cleanupDesc and the failure path both skip.
        auto *list = aclCreateTensorList(source.data() + begin, count);
        if (!list)
            LOGf << name << ": foreach tensor list creation failed";
        for (size_t index = begin; index < begin + count; ++index)
            source[index] = nullptr;
        lists.push_back(list);
        return list;
    }

    const aclTensorList *ForeachOpRunner::inputList(size_t begin, size_t count)
    {
        return makeList(inputTensors, begin, count);
    }

    const aclTensorList *ForeachOpRunner::outputList(size_t begin, size_t count)
    {
        return makeList(outputTensors, begin, count);
    }

    void ForeachOpRunner::foreachMulScalar(
        const aclTensorList *x, const aclTensor *scalar,
        const aclTensorList *out, bool synchronize)
    {
        ret = aclnnForeachMulScalarGetWorkspaceSize(x, scalar, out,
                                                    &workspaceSize, &executor);
        launch(ret, aclnnForeachMulScalar, synchronize);
    }

    void ForeachOpRunner::foreachAddList(
        const aclTensorList *x1, const aclTensorList *x2, const aclTensor *alpha,
        const aclTensorList *out, bool synchronize)
    {
        ret = aclnnForeachAddListGetWorkspaceSize(x1, x2, alpha, out,
                                                  &workspaceSize, &executor);
        launch(ret, aclnnForeachAddList, synchronize);
    }

    void ForeachOpRunner::foreachAddScalar(
        const aclTensorList *x, const aclTensor *scalar,
        const aclTensorList *out, bool synchronize)
    {
        ret = aclnnForeachAddScalarGetWorkspaceSize(x, scalar, out,
                                                    &workspaceSize, &executor);
        launch(ret, aclnnForeachAddScalar, synchronize);
    }

    void ForeachOpRunner::foreachSqrt(
        const aclTensorList *x, const aclTensorList *out, bool synchronize)
    {
        ret = aclnnForeachSqrtGetWorkspaceSize(x, out, &workspaceSize, &executor);
        launch(ret, aclnnForeachSqrt, synchronize);
    }

    void ForeachOpRunner::foreachAddcmulScalar(
        const aclTensorList *x1, const aclTensorList *x2, const aclTensorList *x3,
        const aclTensor *scalar, const aclTensorList *out, bool synchronize)
    {
        ret = aclnnForeachAddcmulScalarGetWorkspaceSize(x1, x2, x3, scalar, out,
                                                        &workspaceSize, &executor);
        launch(ret, aclnnForeachAddcmulScalar, synchronize);
    }

    void ForeachOpRunner::foreachAddcdivScalar(
        const aclTensorList *x1, const aclTensorList *x2, const aclTensorList *x3,
        const aclTensor *scalar, const aclTensorList *out, bool synchronize)
    {
        ret = aclnnForeachAddcdivScalarGetWorkspaceSize(x1, x2, x3, scalar, out,
                                                        &workspaceSize, &executor);
        launch(ret, aclnnForeachAddcdivScalar, synchronize);
    }
}
