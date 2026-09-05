#pragma once
#include "common.h"

namespace jittor {

class StartupConfigState {
public:
    void seal() { sealed_ = true; }
    bool sealed() const { return sealed_; }

private:
    bool sealed_ = false;
};

EXTERN_LIB StartupConfigState& runtime_startup_config();
EXTERN_LIB void check_startup_config_write(const char* name);

// @pyjt(seal_startup_config)
void seal_startup_config();

} // namespace jittor
