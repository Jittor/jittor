#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner();

    protected:
        // Every CANN binary broadcasts its operands, so a one-element operand
        // may be handed over as a one-element tensor.
        bool collapsesScalarInputs() const override { return true; }

        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
