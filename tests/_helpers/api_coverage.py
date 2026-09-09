# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Which public entry points the suite actually calls.

``tests/structure/public_api_manifest.json`` pins the 1294 names the public
surface has; it proves they still *resolve*. It cannot say whether anything
ever *called* them, and that is the number that matters: a `bitwise_not` that
returns True for every bool input sat behind an OpInfo entry the whole time,
because the entry declared `dtypes=_INT` and the samples forced integers.

Counting textual mentions does not answer it either. Every one of the 1294
names appears somewhere under ``tests/`` -- in an import, a comment, an
unrelated identifier -- so that measure reports ~96% and means nothing. The
only honest measure is dynamic: wrap each entry point, run the suite, and see
which wrappers fired.

Enabled by ``JITTOR_API_COVERAGE=1`` so a normal run pays nothing. The wrapper
is a thin ``__call__`` shim that records the qualified name and delegates; it
does not change arguments, return values or exceptions, and a name it cannot
wrap (a non-callable attribute, a slot that refuses assignment) is recorded as
unwrappable rather than silently dropped -- an entry point missing from the
denominator would flatter the result, which is the failure mode this file
exists to prevent.
"""

import functools
import json
import os
import pathlib


MANIFEST = pathlib.Path(__file__).resolve().parents[1] / "structure" / "public_api_manifest.json"

#: Qualified names observed during the session.
CALLED = set()
#: Qualified names the manifest lists but this process could not wrap.
UNWRAPPABLE = set()
#: Qualified names successfully wrapped -- the denominator.
WRAPPED = set()


def enabled():
    return os.environ.get("JITTOR_API_COVERAGE") == "1"


def _load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _owner(module_key, jt):
    """Resolve a manifest key such as ``jt.nn.functional`` or ``Var``."""
    if module_key == "Var":
        return jt.Var
    parts = module_key.split(".")
    if parts[0] != "jt":
        return None
    obj = jt
    for part in parts[1:]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _wrap(qualified, func):
    @functools.wraps(func)
    def recorder(*args, **kwargs):
        CALLED.add(qualified)
        return func(*args, **kwargs)
    recorder._jittor_api_coverage_wrapped = True
    return recorder


def install():
    """Wrap every manifest entry. Returns ``(wrapped, unwrappable)`` counts."""
    import jittor as jt

    for module_key, names in _load_manifest().items():
        owner = _owner(module_key, jt)
        if owner is None:
            for name in names:
                UNWRAPPABLE.add("%s.%s" % (module_key, name))
            continue
        for name in names:
            qualified = "%s.%s" % (module_key, name)
            try:
                attr = getattr(owner, name)
            except AttributeError:
                UNWRAPPABLE.add(qualified)
                continue
            if not callable(attr) or isinstance(attr, type):
                # A class is an entry point too, but wrapping it would replace
                # the type and break isinstance; record it as out of reach
                # rather than pretend it is covered.
                UNWRAPPABLE.add(qualified)
                continue
            if getattr(attr, "_jittor_api_coverage_wrapped", False):
                WRAPPED.add(qualified)
                continue
            try:
                setattr(owner, name, _wrap(qualified, attr))
            except (AttributeError, TypeError):
                UNWRAPPABLE.add(qualified)
                continue
            WRAPPED.add(qualified)
    return len(WRAPPED), len(UNWRAPPABLE)


def report():
    """A deterministic summary of what the session exercised."""
    covered = sorted(CALLED & WRAPPED)
    missed = sorted(WRAPPED - CALLED)
    return {
        "wrapped": len(WRAPPED),
        "unwrappable": len(UNWRAPPABLE),
        "called": len(covered),
        "uncalled": len(missed),
        "uncalled_names": missed,
        "unwrappable_names": sorted(UNWRAPPABLE),
    }


def write_report(path):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report(), indent=2, sort_keys=True), encoding="utf-8")
    return path
