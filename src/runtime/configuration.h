#pragma once
#include "core/common.h"

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

// Every flag whose value the environment supplied, as
// ``[flag, variable, raw value, "prefixed"|"legacy"]`` rows.
//
// The core reads these during static initialization, i.e. before any Python
// code can observe it, and used to say so with one ``LOGi`` line per flag --
// a level the default ``log_v=0`` and ``log_silent`` both hide. The bootstrap
// turns these rows into one summary line and one DeprecationWarning for the
// unprefixed names; see ``_runtime/env_report.py``.
// @pyjt(env_flag_sources)
vector<vector<string>> env_flag_source_rows();

} // namespace jittor
