#include "runtime/configuration.h"
#include <stdexcept>

namespace jittor {

void check_startup_config_write(const char* name) {
    if (runtime_startup_config().sealed())
        throw std::runtime_error(string(name)
            + " is immutable startup configuration; set it before import jittor");
}

void seal_startup_config() {
    runtime_startup_config().seal();
}

} // namespace jittor
