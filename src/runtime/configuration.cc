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

vector<vector<string>> env_flag_source_rows() {
    vector<vector<string>> rows;
    for (auto& source : env_flag_sources())
        rows.push_back({source.flag, source.env_name, source.value,
                        source.legacy ? "legacy" : "prefixed"});
    return rows;
}

} // namespace jittor
