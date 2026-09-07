"""Locate backend resources in checkouts, installations, and converted trees."""

import os


def backend_root(jittor_path, name):
    if not name or not name.replace("_", "").isalnum():
        raise ValueError("backend name must be an identifier")
    package = os.path.abspath(os.fspath(jittor_path))
    parent = os.path.dirname(package)
    if os.path.basename(parent) == "python" and os.path.basename(package) == "jittor":
        checkout = os.path.join(os.path.dirname(parent), "backends", name)
        if os.path.isfile(os.path.join(checkout, "__init__.py")):
            return checkout
    installed = os.path.join(package, "backends", name)
    if os.path.isfile(os.path.join(installed, "__init__.py")):
        return installed
    raise FileNotFoundError(
        "backend resources for %s are missing from %s; check the source layout "
        "or reinstall the complete Jittor package" % (name, package))


def core_root(jittor_path):
    """Locate the C++ core sources, in a checkout or an installed tree.

    `4.15` moved them out of the Python package to the top level, so a checkout
    keeps them beside ``backends/`` while a wheel still ships them inside
    ``jittor/``. Same two-location shape as :func:`backend_root`, and the same
    reason: one place answers "where is it" so a layout change is one edit.

    The marker file is ``common.h``. It is the header every other core
    translation unit includes, so a directory that has it is the core and a
    directory that does not is not -- unlike a bare ``isdir``, which would
    happily accept an empty leftover directory and hand back a root whose
    ``-I`` finds nothing.
    """
    package = os.path.abspath(os.fspath(jittor_path))
    parent = os.path.dirname(package)
    if os.path.basename(parent) == "python" and os.path.basename(package) == "jittor":
        checkout = os.path.join(os.path.dirname(parent), "src")
        if os.path.isfile(os.path.join(checkout, "common.h")):
            return checkout
    installed = os.path.join(package, "src")
    if os.path.isfile(os.path.join(installed, "common.h")):
        return installed
    raise FileNotFoundError(
        "the C++ core sources are missing from %s; check the source layout or "
        "reinstall the complete Jittor package" % package)
