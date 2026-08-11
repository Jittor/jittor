# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""``op_db`` -- the jittor operator test registry (aggregator).

The jittor analogue of ``torch.testing._internal.common_methods_invocations``: it
auto-discovers every module under ``opinfo/definitions/`` and concatenates their
``op_db`` lists into a single ``op_db`` consumed by ``test_ops.py``. Each definition
module pairs a jittor operator with an INDEPENDENT numpy reference and a sample-input
generator; ``test_ops.py`` then generates, for every op, forward-vs-reference tests
across devices/dtypes plus a ``gradcheck`` of the backward.

Adding ops is dropping a file in ``definitions/`` (with an ``op_db`` list) -- it is
picked up here automatically, and every op gets the full fwd+bwd+multi-device
battery for free. This is how the suite makes that coverage the default instead of
something each test file re-implements (the audit's central finding).
"""
import importlib
import pkgutil

from . import definitions as _definitions_pkg
from .core import (  # re-exported for convenience
    OpInfo, UnaryUfuncInfo, BinaryUfuncInfo, ReductionOpInfo, SampleInput,
    DecorateInfo, skip, xfail,
)

op_db = []
_loaded_modules = []


def _discover():
    """Import every non-private module under ``definitions/`` and collect op_db."""
    for info in pkgutil.iter_modules(_definitions_pkg.__path__):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{_definitions_pkg.__name__}.{info.name}")
        mod_ops = getattr(mod, "op_db", None)
        if mod_ops:
            op_db.extend(mod_ops)
            _loaded_modules.append(info.name)


_discover()

# Guard: every op's full_name must be unique. The generic test templates key their
# generated methods by full_name (setattr), so a duplicate would SILENTLY overwrite
# the other op's test -- only one would ever run. Make that a loud error instead
# (the suite's own "loud crash > silent wrong" rule), so adding a colliding OpInfo
# fails fast rather than quietly dropping coverage.
def _check_unique_names():
    seen = {}
    for o in op_db:
        if o.full_name in seen:
            raise RuntimeError(
                f"duplicate OpInfo full_name '{o.full_name}' "
                f"(from '{seen[o.full_name]}' and '{o.name}'); use variant_test_name "
                f"to disambiguate -- a collision would silently drop one op's tests")
        seen[o.full_name] = o.name


_check_unique_names()

# domain subsets (handy for templates that want only some ops)
unary_ufuncs = [o for o in op_db if isinstance(o, UnaryUfuncInfo)]
binary_ufuncs = [o for o in op_db if isinstance(o, BinaryUfuncInfo)]
reduction_ops = [o for o in op_db if isinstance(o, ReductionOpInfo)]


def op_names():
    return sorted(o.full_name for o in op_db)
