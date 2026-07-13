# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Per-domain OpInfo definition modules (mirrors ``torch.../opinfo/definitions``).

Each module here defines a list named ``op_db`` of :class:`OpInfo` objects for one
operator domain (elementwise, reductions, nn, indexing, shape, linalg, ...). The
top-level ``common_methods_invocations`` auto-discovers every module in this package
and concatenates their ``op_db`` lists, so adding a domain is just dropping a new
file here -- no central registry to edit (and no merge conflicts when several are
added at once).

Shared numpy references and sample-input builders live in :mod:`._refs`.
"""
