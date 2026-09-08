"""The installation boundary shared by all Torch API families.

Installers receive one context, publish into its target/registry, and resolve
native delegates through its backend. They must join the active transaction
for persistent writes. A normal return means the step completed; exceptions
are interpreted by InstallContext's required/optional policy, not swallowed
by individual installers. Return values are opaque to the coordinator.
"""
from typing import TYPE_CHECKING, Tuple
import inspect
import sys

if sys.version_info >= (3, 8):
    from typing import Protocol
else:  # Python 3.7, declared by the core distribution.
    from typing_extensions import Protocol

if TYPE_CHECKING:
    from .context import InstallContext


class Installer(Protocol):
    def __call__(self, __context: "InstallContext") -> object:
        """Bind a family to this installation, or raise with context."""
        ...


InstallStep = Tuple[str, Installer]


def validate_installer(installer: Installer, step: str) -> None:
    """Reject an invalid entry before it can mutate installation state."""
    if not callable(installer):
        raise TypeError("installer %r must be callable(context)" % step)
    try:
        inspect.signature(installer).bind(object())
    except (TypeError, ValueError) as error:
        raise TypeError("installer %r must accept one InstallContext argument" % step) from error
