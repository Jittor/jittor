#pragma once
#include "base_op.h"

namespace jittor {
struct MultinomialOpRunner : public BaseOpRunner {
    MultinomialOpRunner(int64_t num_samples, bool replacement);
protected:
    int64_t num_samples;
    bool replacement;
    void executeOp(AclOpRegistry::const_iterator &it) override;
};
}
