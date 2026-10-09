"""Explicit, versioned DeepSpeed integration; importing this package is inert."""

SUPPORTED_VERSIONS = frozenset({"0.17.6"})


def activate(device="cpu"):
    from .activation import activate as install

    return install(device=device)


def deactivate():
    from .activation import deactivate as uninstall

    return uninstall()


def status():
    from .activation import status as describe

    return describe()


__all__ = ["SUPPORTED_VERSIONS", "activate", "deactivate", "status"]
