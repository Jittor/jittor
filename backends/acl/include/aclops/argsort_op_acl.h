#pragma once
#include "base_op.h"

namespace jittor {
class ArgsortOpRunner : public BaseOpRunner {
    int64_t dim;
    bool descending;
protected:
    void executeOp(AclOpRegistry::const_iterator &it) override;
public:
    ArgsortOpRunner(int64_t dim, bool descending);
};
} // namespace jittor
