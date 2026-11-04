#pragma once
#include "base_op.h"

namespace jittor {
class LinalgVectorNormOpRunner : public BaseOpRunner {
public:
    LinalgVectorNormOpRunner();

protected:
    void executeOp(AclOpRegistry::const_iterator &it) override;
};
}
