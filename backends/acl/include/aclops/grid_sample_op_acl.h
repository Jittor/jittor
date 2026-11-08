#pragma once
#include "base_op.h"

namespace jittor {
class GridSampler2DOpRunner : public BaseOpRunner {
public:
    GridSampler2DOpRunner();

protected:
    void executeOp(AclOpRegistry::const_iterator &it) override;
};
}
