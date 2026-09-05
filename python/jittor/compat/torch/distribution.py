"""Import-neutral manifest for the standalone ``torch.distributed`` package.

The compatibility installer still supplies the implementation, but package
layout is a publication concern.  Keeping the names here makes a future
``jittor-torch`` distribution able to build the same import graph without
importing CUDA/NCCL, FSDP, or the native Jittor runtime.
"""

from __future__ import annotations



DISTRIBUTION_ROOT = "torch.distributed"
# Packaging identity is deliberately separate from the import root.  The
# future wheel is named ``jittor-torch`` but publishes a ``torch`` namespace;
# keeping both values here prevents a builder from deriving one from the
# other (and accidentally naming the wheel ``torch``).
DISTRIBUTION_PROJECT = "jittor-torch"
DISTRIBUTION_SCHEMA_VERSION = 1
DISTRIBUTION_IMPORT_ROOT = "torch"

# This is the public import graph assembled by the FSDP2 installer.  It is a
# manifest rather than an installer dependency so packaging/bootstrap checks
# can validate it on CPU-only and NPU hosts alike.
DISTRIBUTION_MODULES = (
    "torch.distributed",
    "torch.distributed.tensor",
    "torch.distributed._tensor",
    "torch.distributed.tensor._api",
    "torch.distributed.tensor.placement_types",
    "torch.distributed.tensor._dtensor_spec",
    "torch.distributed.tensor._utils",
    "torch.distributed.tensor.parallel",
    "torch.distributed.tensor.parallel.api",
    "torch.distributed.tensor.parallel.style",
    "torch.distributed.tensor.parallel.loss",
    "torch.distributed.device_mesh",
    "torch.distributed._tensor.device_mesh",
    "torch.distributed.tensor.device_mesh",
    "torch.distributed.fsdp",
    "torch.distributed.fsdp.api",
    "torch.distributed.fsdp.fully_sharded_data_parallel",
    "torch.distributed.fsdp.wrap",
    "torch.distributed.fsdp._traversal_utils",
    "torch.distributed.fsdp._runtime_utils",
    "torch.distributed.fsdp._common_utils",
    "torch.distributed.fsdp._fsdp_state",
    "torch.distributed.fsdp.sharded_grad_scaler",
    "torch.distributed.fsdp._fully_shard",
    "torch.distributed.fsdp._fully_shard._fully_shard",
    "torch.distributed.fsdp._fully_shard._fsdp_api",
    "torch.distributed.fsdp._fully_shard._fsdp_common",
    "torch.distributed.fsdp._fully_shard._fsdp_init",
    "torch.distributed.fsdp._fully_shard._fsdp_state",
    "torch.distributed.fsdp._fully_shard._fsdp_param",
    "torch.distributed.fsdp._fully_shard._fsdp_collectives",
    "torch.distributed._composable",
    "torch.distributed._composable.fsdp",
    "torch.distributed._composable.fsdp.fully_shard",
    "torch.distributed._composable.fsdp._fsdp_api",
    "torch.distributed._functional_collectives",
    "torch.distributed.algorithms",
    "torch.distributed.algorithms._checkpoint",
    "torch.distributed.algorithms._checkpoint.checkpoint_wrapper",
)

# These are alternate package spellings, not implementation aliases.  The
# installer intentionally gives each spelling its own module object because
# both paths carry different submodule attributes in existing callers.
DISTRIBUTION_PACKAGE_ALIASES = (
    ("torch.distributed._tensor", "torch.distributed.tensor"),
    ("torch.distributed._tensor.device_mesh", "torch.distributed.tensor.device_mesh"),
)


class _DistributionManifest(dict):
    """Small import-free read-only mapping for package metadata."""

    @staticmethod
    def _readonly(*args, **kwargs):
        raise TypeError("distribution manifest is read-only")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _readonly

    def __ior__(self, other):
        self._readonly(other)
        return self


def distribution_manifest():
    """Return the immutable package boundary consumed by packagers.

    Keeping this snapshot import-neutral gives a wheel builder (or a host
    bootstrap) one object to inspect without importing the native runtime.
    Tuples are returned directly so callers cannot mutate the canonical graph.
    """

    return _DistributionManifest({
        "root": DISTRIBUTION_ROOT,
        "modules": DISTRIBUTION_MODULES,
        "packages": distribution_package_names(),
        "aliases": DISTRIBUTION_PACKAGE_ALIASES,
    })


def distribution_metadata():
    """Return the backend-neutral project metadata consumed by packagers.

    This is intentionally flat and immutable at the top level: a standalone
    wheel builder can inspect the project/import identities and the exact
    module graph without importing ``jittor``, CUDA, NCCL, or ACL.
    """

    return _DistributionManifest({
        "project": DISTRIBUTION_PROJECT,
        "schema_version": DISTRIBUTION_SCHEMA_VERSION,
        "import_root": DISTRIBUTION_IMPORT_ROOT,
        "root": DISTRIBUTION_ROOT,
        "modules": DISTRIBUTION_MODULES,
        "packages": distribution_package_names(),
        "aliases": DISTRIBUTION_PACKAGE_ALIASES,
    })


def validate_distribution_metadata(metadata=None):
    """Validate project identity and its exact import-graph manifest."""

    spec = distribution_metadata() if metadata is None else metadata
    if not hasattr(spec, "__getitem__"):
        raise TypeError("distribution metadata must be a mapping")
    try:
        project = spec["project"]
        schema_version = spec["schema_version"]
        import_root = spec["import_root"]
        manifest = {
            "root": spec["root"],
            "modules": spec["modules"],
            "packages": spec["packages"],
            "aliases": spec["aliases"],
        }
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError(
            "distribution metadata must provide project/schema_version/import_root/manifest"
        ) from exc
    if project != DISTRIBUTION_PROJECT:
        raise ValueError("distribution project must be %r" % DISTRIBUTION_PROJECT)
    if schema_version != DISTRIBUTION_SCHEMA_VERSION:
        raise ValueError("unsupported distribution metadata schema version: %r" % schema_version)
    if import_root != DISTRIBUTION_IMPORT_ROOT:
        raise ValueError("distribution import root must be %r" % DISTRIBUTION_IMPORT_ROOT)
    validate_distribution_manifest(manifest)
    return True


def validate_distribution_aliases(names=None, aliases=DISTRIBUTION_PACKAGE_ALIASES):
    """Validate alias schema/endpoints and reject ambiguous package aliases.

    Alias declarations are packaging input.  Validate their shape before
    unpacking so malformed wheel metadata fails with a stable contract error
    instead of leaking an implementation-level ``ValueError``/``TypeError``.
    Each alias must point directly at a canonical endpoint; allowing an alias
    target to be another alias source would make publication order observable
    and can create cycles when two independently generated manifests merge.
    """

    present = set(DISTRIBUTION_MODULES if names is None else names)
    seen = set()
    try:
        alias_entries = tuple(aliases)
    except TypeError as exc:
        raise ValueError("distribution aliases must be an iterable of pairs") from exc
    for entry in alias_entries:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("distribution alias must be a pair: %r" % (entry,))
        source, target = entry
        if not isinstance(source, str) or not isinstance(target, str):
            raise ValueError(
                "distribution alias endpoints must be strings: %r" % (entry,)
            )
        if source == target:
            raise ValueError("distribution alias cannot target itself: %r" % source)
        if source not in present or target not in present:
            raise ValueError(
                "distribution alias endpoint is missing: %r -> %r"
                % (source, target)
            )
        if source in seen:
            raise ValueError("distribution alias is declared more than once: %r" % source)
        seen.add(source)
    alias_sources = {source for source, _ in alias_entries}
    chained = sorted(
        (source, target)
        for source, target in alias_entries
        if target in alias_sources
    )
    if chained:
        source, target = chained[0]
        raise ValueError(
            "distribution alias target must be canonical, not another alias source: "
            "%r -> %r" % (source, target)
        )
    return True


def validate_distribution_manifest(manifest=None):
    """Validate the package graph contract before any module is imported.

    A wheel/bootstrap manifest is an input boundary, not trusted metadata.
    Validate its root, module names, package closure and package aliases in
    one backend-neutral step so malformed metadata cannot reach an installer.
    """

    spec = distribution_manifest() if manifest is None else manifest
    if not hasattr(spec, "__getitem__"):
        raise TypeError("distribution manifest must be a mapping")
    try:
        root = spec["root"]
        modules = tuple(spec["modules"])
        packages = tuple(spec["packages"])
        aliases = tuple(spec["aliases"])
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError(
            "distribution manifest must provide root/modules/packages/aliases"
        ) from exc

    if not isinstance(root, str) or not root:
        raise ValueError("distribution manifest root must be a non-empty string")
    if not modules or modules[0] != root:
        raise ValueError("distribution manifest must start with its root module")
    if len(set(modules)) != len(modules):
        raise ValueError("distribution manifest contains duplicate modules")
    for name in modules:
        if not isinstance(name, str) or not name or any(
            not part.isidentifier() for part in name.split(".")
        ):
            raise ValueError("distribution manifest contains invalid module name: %r" % (name,))
    if root not in modules:
        raise ValueError("distribution manifest root is not published")

    expected_packages = {root}
    for name in modules:
        parts = name.split(".")
        for index in range(3, len(parts)):
            expected_packages.add(".".join(parts[:index]))
    for package in packages:
        if not isinstance(package, str) or not package or any(
            not part.isidentifier() for part in package.split(".")
        ):
            raise ValueError(
                "distribution manifest contains invalid package name: %r" % (package,)
            )
    if len(set(packages)) != len(packages):
        raise ValueError("distribution manifest contains duplicate packages")
    if set(packages) != expected_packages:
        missing = sorted(expected_packages - set(packages))
        extra = sorted(set(packages) - expected_packages)
        raise ValueError(
            "distribution manifest package closure mismatch (missing=%r, extra=%r)"
            % (missing, extra)
        )
    if any(package not in modules for package in packages):
        raise ValueError("distribution manifest package is not a module")
    canonical_packages = tuple(
        sorted(expected_packages, key=lambda item: (item.count("."), item))
    )
    if packages != canonical_packages:
        raise ValueError(
            "distribution manifest packages must use canonical order"
        )

    validate_distribution_graph(modules, modules=modules, aliases=aliases)
    return True


def distribution_module_names():
    """Return a stable tuple suitable for packaging and import checks."""

    return DISTRIBUTION_MODULES


def distribution_package_names():
    """Return package nodes that must expose ``__path__``."""

    packages = {DISTRIBUTION_ROOT}
    for name in DISTRIBUTION_MODULES:
        parts = name.split(".")
        for index in range(3, len(parts)):
            packages.add(".".join(parts[:index]))
    return tuple(sorted(packages, key=lambda item: (item.count("."), item)))


def validate_distribution_graph(
    names, modules=DISTRIBUTION_MODULES, aliases=DISTRIBUTION_PACKAGE_ALIASES
):
    """Validate that a published module-name set has complete parent closure.

    The function deliberately accepts names only, so it is safe to run while
    building wheels or before a native backend has been selected.
    """

    try:
        published_names = tuple(names)
    except TypeError as exc:
        raise TypeError("distribution graph names must be an iterable") from exc
    for name in published_names:
        if not isinstance(name, str) or not name or any(
            not part.isidentifier() for part in name.split(".")
        ):
            raise ValueError(
                "distribution graph contains invalid module name: %r" % (name,)
            )
    present = set(published_names)
    expected = set(modules)
    missing = tuple(sorted(expected - present))
    if missing:
        raise ValueError("distribution graph is missing: %s" % ", ".join(missing))
    extra = tuple(sorted(present - expected))
    if extra:
        raise ValueError("distribution graph contains unpublished modules: %s" % ", ".join(extra))
    validate_distribution_aliases(present, aliases)
    for name in expected:
        if name == "torch":
            continue
        parent = name.rsplit(".", 1)[0]
        # The detached root is published by the Torch namespace boundary; the
        # distribution manifest starts at its child package by design.
        if parent == "torch":
            continue
        if parent not in present:
            raise ValueError(
                "distribution graph is missing parent %r for %r" % (parent, name)
            )
    return True


def validate_distribution_publication(published, manifest=None):
    """Validate the object graph emitted by an installer or bootstrap.

    ``validate_distribution_graph`` only deals with names, which is enough for
    a wheel manifest but not for a live installer registry.  This validator
    stays import-neutral and checks the four invariants a publisher must keep:
    every manifest node has the right module identity, package nodes expose a
    path, every child is attached to its published parent, and aliases point
    at published objects.  It deliberately never imports a backend or looks
    at ``sys.modules``.
    """

    if not hasattr(published, "__getitem__") or not hasattr(published, "keys"):
        raise TypeError("published distribution must be a mapping")
    spec = distribution_manifest() if manifest is None else manifest
    validate_distribution_manifest(spec)
    try:
        modules = tuple(spec["modules"])
        aliases = tuple(spec["aliases"])
        packages = set(spec.get("packages", ()))
    except (AttributeError, KeyError, TypeError) as exc:
        raise TypeError("distribution manifest must provide modules/packages/aliases") from exc

    validate_distribution_graph(published.keys(), modules=modules, aliases=aliases)
    for name in modules:
        try:
            module = published[name]
        except KeyError as exc:
            raise ValueError("published distribution is missing %r" % name) from exc
        if getattr(module, "__name__", None) != name:
            raise ValueError(
                "published module %r has wrong __name__ %r"
                % (name, getattr(module, "__name__", None))
            )
        if name in packages and getattr(module, "__path__", None) is None:
            raise ValueError("published package %r has no __path__" % name)
        if name == "torch.distributed":
            continue
        parent_name, attr = name.rsplit(".", 1)
        parent = published.get(parent_name)
        if parent is None:
            raise ValueError("published parent %r is missing for %r" % (parent_name, name))
        if getattr(parent, attr, None) is not module:
            raise ValueError(
                "published parent %r does not bind child %r" % (parent_name, name)
            )
    for source, target in aliases:
        if published[source] is published[target]:
            raise ValueError(
                "distribution alias %r and %r unexpectedly share an object"
                % (source, target)
            )
    return True


def validate_distribution_boundary(published=None, manifest=None):
    """Validate a standalone distribution before bootstrap mutates state.

    Wheel builders only have a manifest, while a live shim bootstrap also has
    the registry mapping that will be published.  Keep both checks behind one
    backend-neutral entry point so callers cannot accidentally validate only
    names or only live bindings.  ``published`` is optional for the packaging
    path and, when supplied, is validated against the exact manifest object.
    """

    spec = distribution_manifest() if manifest is None else manifest
    validate_distribution_manifest(spec)
    if published is not None:
        validate_distribution_publication(published, spec)
    return True


def validate_distribution_bootstrap(bootstrap):
    """Validate the stable bootstrap facade for standalone distribution use.

    The bootstrap module is deliberately a thin export surface.  A wrapper or
    copied constant there would make a wheel builder validate one contract
    while runtime activation consumes another.  Check object identity for the
    manifest constants and validators so the facade cannot drift silently.
    This function only inspects the supplied module-like object and never
    imports a backend or mutates process state.
    """

    if bootstrap is None:
        raise TypeError("distribution bootstrap must be a module-like object")
    constants = (
        "DISTRIBUTION_ROOT",
        "DISTRIBUTION_PROJECT",
        "DISTRIBUTION_SCHEMA_VERSION",
        "DISTRIBUTION_IMPORT_ROOT",
        "DISTRIBUTION_MODULES",
        "DISTRIBUTION_PACKAGE_ALIASES",
    )
    functions = (
        "distribution_manifest",
        "distribution_metadata",
        "distribution_module_names",
        "distribution_package_names",
        "validate_distribution_aliases",
        "validate_distribution_manifest",
        "validate_distribution_metadata",
        "validate_distribution_graph",
        "validate_distribution_publication",
        "validate_distribution_boundary",
        "validate_distribution_bootstrap",
    )
    for name in constants + functions:
        try:
            exported = getattr(bootstrap, name)
        except AttributeError as exc:
            raise ValueError("distribution bootstrap is missing %r" % name) from exc
        if exported is not globals()[name]:
            raise ValueError(
                "distribution bootstrap export %r is not the canonical object" % name
            )
    try:
        public_names = tuple(bootstrap.__all__)
    except (AttributeError, TypeError) as exc:
        raise ValueError(
            "distribution bootstrap must define a string __all__"
        ) from exc
    if any(not isinstance(name, str) for name in public_names):
        raise ValueError(
            "distribution bootstrap __all__ must contain only strings"
        )
    if len(set(public_names)) != len(public_names):
        raise ValueError(
            "distribution bootstrap __all__ contains duplicate names"
        )
    missing_public = tuple(
        name for name in constants + functions if name not in public_names
    )
    if missing_public:
        raise ValueError(
            "distribution bootstrap __all__ is missing %r" % (missing_public,)
        )
    validate_distribution_boundary()
    return True


__all__ = [
    "DISTRIBUTION_PROJECT",
    "DISTRIBUTION_SCHEMA_VERSION",
    "DISTRIBUTION_IMPORT_ROOT",
    "DISTRIBUTION_ROOT",
    "DISTRIBUTION_MODULES",
    "DISTRIBUTION_PACKAGE_ALIASES",
    "distribution_module_names",
    "distribution_package_names",
    "distribution_manifest",
    "distribution_metadata",
    "validate_distribution_metadata",
    "validate_distribution_aliases",
    "validate_distribution_manifest",
    "validate_distribution_graph",
    "validate_distribution_publication",
    "validate_distribution_boundary",
    "validate_distribution_bootstrap",
]
