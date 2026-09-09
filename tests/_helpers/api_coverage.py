# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Which public entry points the suite actually calls.

``tests/structure/public_api_manifest.json`` pins the 1294 names the native
public surface has; it proves they still *resolve*. It cannot say whether
anything ever *called* them, and that is the number that matters: a
`bitwise_not` that returns True for every bool input sat behind an OpInfo entry
the whole time, because the entry declared `dtypes=_INT` and the samples forced
integers.

Counting textual mentions does not answer it either. Every one of the 1294
names appears somewhere under ``tests/`` -- in an import, a comment, an
unrelated identifier -- so that measure reports ~96% and means nothing. The
only honest measure is dynamic: wrap each entry point, run the suite, and see
which wrappers fired.

Enabled by ``JITTOR_API_COVERAGE=1`` so a normal run pays nothing. The wrapper
is a thin ``__call__`` shim that records the qualified name and delegates; it
does not change arguments, return values or exceptions, and a name it cannot
wrap (a non-callable attribute, a class, a callable module, a slot that refuses
assignment) is recorded as unwrappable rather than silently dropped -- an entry
point missing from the denominator would flatter the result, which is the
failure mode this file exists to prevent.

Two surfaces, one measurement
-----------------------------
Torch compatibility mode is process-global, and the surface it publishes is a
different object graph: ``torch`` is a namespace view over the compatibility
owner, ``torch.Tensor`` is not ``jt.Var``, and ``torch.nn`` is composed rather
than re-exported. So there are two manifests and a run measures exactly one of
them -- the one its process mode owns. ``install`` therefore takes the surface
name from its caller instead of guessing: ``tests/_helpers/pytest_policy.py``
already decides the process mode from ``JITTOR_TORCH_SHIM`` and nothing else,
and a second copy of that decision here is a second thing to drift.

**The on/off comparison has to be run once per surface.** Passing on one is not
evidence about the other, and the reason is in the code below: the native
surface has no callable module, so wrapping every callable was harmless there
and stayed harmless through every native check -- while on the Torch surface
the same line replaced ``torch.random`` and broke the frontend's own namespace
ownership test. A diagnostic verified on one object graph has been verified on
one object graph.

The Torch surface also has one limit this wrapper cannot get past, and it is
recorded rather than hidden: the frontend contracts on the *identity* of what
it publishes and keys a fidelity registry by the object, so a wrapper that
replaces the object is visible to those cases. See ``IDENTITY_CONTRACT_FILES``
for the list, the measurements, and what a fix would have to look like.
"""

import functools
import importlib
import json
import os
import pathlib
import types


_STRUCTURE = pathlib.Path(__file__).resolve().parents[1] / "structure"

#: The measurable surfaces, by process mode.
#:
#: ``module`` is imported to reach the surface; ``alias`` is the prefix the
#: manifest keys use for it. ``extra_owners`` names manifest keys that are not
#: written under that prefix -- the native manifest records ``Var`` methods
#: under a bare ``Var`` -- and maps them to the attribute path to walk instead.
#: ``baseline`` is where ``tools/api_coverage_ratchet.py`` keeps the untested
#: set for this surface; it lives here so the tool, the gate and the wrapper
#: read one declaration rather than three copies of the same paths.
SURFACES = {
    "native": {
        "module": "jittor",
        "alias": "jt",
        "extra_owners": {"Var": ("Var",)},
        "manifest": _STRUCTURE / "public_api_manifest.json",
        "baseline": _STRUCTURE / "api_coverage_baseline.json",
    },
    "torch": {
        "module": "torch",
        "alias": "torch",
        "extra_owners": {},
        "manifest": _STRUCTURE / "torch_api_manifest.json",
        "baseline": _STRUCTURE / "torch_api_coverage_baseline.json",
    },
}

#: Qualified names observed during the session.
CALLED = set()
#: Qualified names the manifest lists but this process could not wrap.
UNWRAPPABLE = set()
#: Qualified names successfully wrapped -- the denominator.
WRAPPED = set()
#: Which surface ``install`` measured, so the report says what it counted.
SURFACE = None


def enabled():
    return os.environ.get("JITTOR_API_COVERAGE") == "1"


def declaration(surface):
    """The declaration for one surface, with a useful error for a typo.

    Named for what it returns rather than for the surface itself: the surface
    name is a parameter threaded through half this module, and a function
    called ``surface`` invites a local rebinding of it -- which is exactly the
    bug that shipped for one revision here, when unpacking an owner binding
    into ``name`` turned the surface argument into an attribute name.
    """
    try:
        return SURFACES[surface]
    except KeyError:
        raise KeyError("unknown API surface %r; declared surfaces are %s"
                       % (surface, ", ".join(sorted(SURFACES))))


def _load_manifest(surface):
    return json.loads(declaration(surface)["manifest"].read_text(encoding="utf-8"))


def _owner(module_key, surface, root):
    """Resolve a manifest key such as ``jt.nn.functional``, ``Var`` or ``torch.Tensor``.

    A key names whatever holds the entry points, which is a module for most of
    them and a class for ``Var`` and ``torch.Tensor``. The walk is the same
    either way; what the manifest records is the owner, not its kind.
    """
    spec = declaration(surface)
    path = spec["extra_owners"].get(module_key)
    if path is None:
        parts = module_key.split(".")
        if parts[0] != spec["alias"]:
            return None
        path = parts[1:]
    obj = root
    for part in path:
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


#: What the wrapper cannot hide from, measured rather than assumed.
#:
#: The Torch frontend contracts on the *identity* of what it publishes --
#: ``assertIs(torch.addmm, installers.numerical.addmm)`` -- and keys its
#: fidelity registry by the object itself. A wrapper replaces that object, so
#: those cases see it: 86 of them go red under ``JITTOR_API_COVERAGE=1``, all
#: in the files listed here, and all asserting some form of "nothing has
#: wrapped this". Two attempts to hide from them made it worse (rebinding the
#: defining module: 101 red; rebinding every alias: 97 red, with the failures
#: moving to the object-keyed registry), which is the evidence that a mutating
#: recorder cannot satisfy an identity contract. It is recorded here instead:
#: a Torch coverage run excludes these files, the exclusion is written into the
#: baseline, and the fix is a recorder that observes calls without replacing
#: anything (``sys.setprofile`` keyed by code object). The native surface
#: states no such contract and needs no exclusion.
IDENTITY_CONTRACT_FILES = (
    "compat/tests/torch/test_autograd_library_owners.py",
    "compat/tests/torch/test_core_misc_owner.py",
    "compat/tests/torch/test_division_remainder_family.py",
    "compat/tests/torch/test_explicit_native_api.py",
    "compat/tests/torch/test_family_api_owners.py",
    "compat/tests/torch/test_serialization_api_owners.py",
    "compat/tests/torch/test_torch_compat_cuda_streams.py",
    "compat/tests/torch/test_torch_compat_library.py",
    "compat/tests/torch/test_torch_compiler_fidelity.py",
    "compat/tests/torch/test_torch_cumulative_fidelity.py",
    "compat/tests/torch/test_torch_factory_fidelity.py",
    "compat/tests/torch/test_torch_numerical_fidelity.py",
    "compat/tests/torch/test_torch_ordering_fidelity.py",
    "compat/tests/torch/test_torch_reduction_owner_fidelity.py",
)


def _wrap(qualified, func):
    @functools.wraps(func)
    def recorder(*args, **kwargs):
        CALLED.add(qualified)
        return func(*args, **kwargs)
    recorder._jittor_api_coverage_wrapped = True
    return recorder


def install(surface):
    """Wrap every manifest entry of one surface. Returns ``(wrapped, unwrappable)``."""
    global SURFACE
    spec = declaration(surface)
    SURFACE = surface
    root = importlib.import_module(spec["module"])

    for module_key, names in _load_manifest(surface).items():
        owner = _owner(module_key, surface, root)
        if owner is None:
            for entry in names:
                UNWRAPPABLE.add("%s.%s" % (module_key, entry))
            continue
        for entry in names:
            qualified = "%s.%s" % (module_key, entry)
            try:
                attr = getattr(owner, entry)
            except AttributeError:
                UNWRAPPABLE.add(qualified)
                continue
            if not callable(attr) or isinstance(attr, (type, types.ModuleType)):
                # A class is an entry point too, but wrapping it would replace
                # the type and break isinstance; record it as out of reach
                # rather than pretend it is covered.
                #
                # A module is an owner, never an entry point -- and the Torch
                # frontend has callable ones: ``torch.random`` is a module
                # subclass with ``__call__``, so it passes ``callable`` while
                # also being the published ``torch.random`` namespace. Wrapping
                # it replaced that binding, and the frontend's own ownership
                # check (``_aliases.torch_namespace_owned``) then refused to
                # re-activate: the diagnostic broke the session it was
                # measuring. Accounted, like every other name out of reach.
                UNWRAPPABLE.add(qualified)
                continue
            if getattr(attr, "_jittor_api_coverage_wrapped", False):
                WRAPPED.add(qualified)
                continue
            recorder = _wrap(qualified, attr)
            try:
                setattr(owner, entry, recorder)
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
        "surface": SURFACE,
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
