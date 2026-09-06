// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// Maintainers: Dun Liang <randonlang@gmail.com>.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
// ACL runtime support is provided by acl_runtime/acl_workspace and the
// registered aclops translation units. Source-to-source CUDA rewriting was
// intentionally removed: unsupported generated kernels are rejected by the
// ACL backend instead of being silently rewritten here.
#include "common.h"
#include "acl_jittor.h"

namespace jittor {
// Keep a translation unit for build manifests and SDK-specific ACL globals.
}
