"""Compatibility domains maintained by Jittor."""

from ._aliases import install_aliases as _install_aliases

# Importing this optional domain registers its historical spellings, without
# activating Torch or loading backend integrations.
_install_aliases()
del _install_aliases
