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
