"""Build implementation namespace, loaded after standalone utilities bootstrap."""

from jittor.compat._aliases import install_aliases as _install_aliases

# Utilities retain a standalone import spelling so command-line build tools do
# not initialize Jittor. Canonical build aliases share all of that state.
_install_aliases()
del _install_aliases
