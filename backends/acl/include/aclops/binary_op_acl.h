#pragma once
#include "utils.h"
#include "base_op.h"

namespace jittor
{
    struct BinaryOpRunner : public BaseOpRunner
    {
        BinaryOpRunner();

    protected:
        bool allowsFlattenedElementwise() const override { return true; }
        // Every CANN binary broadcasts its operands, so a one-element operand
        // may be handed over as a one-element tensor.
        bool collapsesScalarInputs() const override { return true; }

        void executeOp(AclOpRegistry::const_iterator &it) override;
    };

    // Binary tensor-with-host-scalar launcher (aclnnAdds/Subs/Muls/Divs), used
    // when a fused scalar ArrayOp (+ optional BroadcastToOp) is elided to avoid
    // device scratch allocation and an H2D DMA per scalar arithmetic op.
    struct BinaryScalarOpRunner : public BaseOpRunner
    {
        BinaryScalarOpRunner();
        const void *scalar_data = nullptr;
        NanoString scalar_dtype;

    protected:
        bool allowsFlattenedElementwise() const override { return true; }

        void executeOp(AclOpRegistry::const_iterator &it) override;
    };
}
