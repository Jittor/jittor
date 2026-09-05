"""Import-neutral manifest for the standalone ``torch.distributed`` package.

The compatibility installer still supplies the implementation, but package
layout is a publication concern.  Keeping the names here makes a future
``jittor-torch`` distribution able to build the same import graph without
importing CUDA/NCCL, FSDP, or the native Jittor runtime.
"""

from __future__ import annotations


DISTRIBUTION_ROOT = "torch.distributed"

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


def validate_distribution_graph(names):
    """Validate that a published module-name set has complete parent closure.

    The function deliberately accepts names only, so it is safe to run while
    building wheels or before a native backend has been selected.
    """

    present = set(names)
    expected = set(DISTRIBUTION_MODULES)
    missing = tuple(sorted(expected - present))
    if missing:
        raise ValueError("distribution graph is missing: %s" % ", ".join(missing))
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


__all__ = [
    "DISTRIBUTION_ROOT",
    "DISTRIBUTION_MODULES",
    "DISTRIBUTION_PACKAGE_ALIASES",
    "distribution_module_names",
    "distribution_package_names",
    "validate_distribution_graph",
]
