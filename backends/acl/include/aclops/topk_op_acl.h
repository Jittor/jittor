#pragma once
#include "base_op.h"

namespace jittor {
class TopkOpRunner : public BaseOpRunner {
public:
    TopkOpRunner();

protected:
    void executeOp(AclOpRegistry::const_iterator &it) override;
};
}
