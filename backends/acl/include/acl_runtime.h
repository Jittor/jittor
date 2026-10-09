#pragma once
#include "core/common.h"
#include <acl/acl.h>

namespace jittor {

EXTERN_LIB int acl_runtime_current_device();
EXTERN_LIB aclrtStream acl_current_stream();
EXTERN_LIB void shutdown_acl_backend() noexcept;

// Recorded device graphs (aclmdlRI) alive or being recorded. A recording
// re-issues the addresses it saw -- the operator workspace, the host scalars
// its copies read -- so while one lives a block that would be replaced is
// kept instead of freed, and no stream is waited on for it (the stream may be
// the one being recorded). Kept blocks go when the last graph does.
EXTERN_LIB int acl_live_graphs();
EXTERN_LIB void acl_scalar_cache_release_retired();

} // namespace jittor
