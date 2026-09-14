#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    // CANN's foreach family applies one elementwise rule to a whole tensor list
    // per launch, which is what an optimizer update wants: a fixed number of
    // launches for the whole parameter list instead of a launch per parameter.
    //
    // Two things about the family on this SoC shape the interface. The
    // `...Inplace` spellings are declared but their OpDef does not list
    // Ascend950, so the usable forms are out-of-place -- they do accept an
    // output list that aliases an input list, which is how an update writes
    // back. And they take their coefficient as a one-element device tensor;
    // the `...V2` spellings that take a host `aclScalar` are available but cost
    // measurably more per launch, so this base stages the coefficients instead.
    class ForeachOpRunner : public BaseOpRunner
    {
    public:
        explicit ForeachOpRunner(const string &name) : BaseOpRunner(name) {}
        ~ForeachOpRunner() override;

    protected:
        // One foreach launch covers at most this many tensors. The tiling
        // function on this SoC refuses a longer list -- "Base tiling failed"
        // out of foreach_regbase_tiling.cpp -- and 50 is measured to be where
        // it stops, so a longer parameter list is split. The launch count then
        // grows with ceil(n / 50) instead of with n.
        static const int64_t list_limit = 50;

        // One stream-ordered upload for every coefficient the launch sequence
        // needs, read back in the order they were staged.
        void stageCoefficients(const std::vector<float> &values);
        const aclTensor *coefficient(size_t index) const;

        // Tensor lists over a contiguous run of the runner's descriptors. The
        // list takes those descriptors over, so a run can be listed once.
        const aclTensorList *inputList(size_t begin, size_t count);
        const aclTensorList *outputList(size_t begin, size_t count);

        // out[i] <- x[i] * scalar
        void foreachMulScalar(const aclTensorList *x, const aclTensor *scalar,
                              const aclTensorList *out, bool synchronize);
        // out[i] <- x1[i] + alpha * x2[i]
        void foreachAddList(const aclTensorList *x1, const aclTensorList *x2,
                            const aclTensor *alpha, const aclTensorList *out,
                            bool synchronize);
        // out[i] <- x[i] + scalar
        void foreachAddScalar(const aclTensorList *x, const aclTensor *scalar,
                              const aclTensorList *out, bool synchronize);
        // out[i] <- sqrt(x[i])
        void foreachSqrt(const aclTensorList *x, const aclTensorList *out,
                         bool synchronize);
        // out[i] <- x1[i] + scalar * x2[i] * x3[i]
        void foreachAddcmulScalar(const aclTensorList *x1, const aclTensorList *x2,
                                  const aclTensorList *x3, const aclTensor *scalar,
                                  const aclTensorList *out, bool synchronize);
        // out[i] <- x1[i] + scalar * x2[i] / x3[i]
        void foreachAddcdivScalar(const aclTensorList *x1, const aclTensorList *x2,
                                  const aclTensorList *x3, const aclTensor *scalar,
                                  const aclTensorList *out, bool synchronize);

    private:
        // A tensor list takes ownership of the descriptors it is given, so
        // this clears the slots it consumed out of the runner's own vector.
        const aclTensorList *makeList(std::vector<aclTensor *> &source,
                                      size_t begin, size_t count);

        std::vector<aclTensor *> coefficients;
        std::vector<const aclTensorList *> lists;
    };
}
