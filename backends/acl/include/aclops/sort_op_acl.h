#pragma once
#include "base_op.h"

namespace jittor {
struct SortOpRunner : public BaseOpRunner {
    SortOpRunner(bool stable, int64_t dim, bool descending);
protected:
    bool stable;
    int64_t dim;
    bool descending;
    void executeOp(AclOpRegistry::const_iterator &it) override;
};
}
