"""Process-local ``torch`` distribution metadata for the in-process shim.

The deployed shim answers ``importlib.metadata.version("torch")`` with the
``torch-<version>.dist-info`` directory :mod:`jittor.compat.shim.deploy` copies
next to it.  The in-process shim (``JITTOR_TORCH_SHIM=1`` with the packaged
``torch`` entry on ``PYTHONPATH``) publishes the ``torch`` module but no
distribution, so a library that asks metadata instead of importing -- and
decides from that answer whether PyTorch is installed -- sees it as absent.

:func:`publish` registers a metadata-only finder that serves the packaged
``resources/torch_dist_info`` directory as the ``torch`` distribution.  It
writes nothing to disk, fabricates no other distribution, and is registered by
shim activation only: plain ``import jittor`` leaves ``importlib.metadata``
unchanged.  Activation records :func:`release` as the undo for a failed install,
and the successful finder stays for the lifetime of the process.
"""

from __future__ import absolute_import

import os
import pathlib
import sys

_NAME = "torch"

_finder = None


def resource_root():
    """The packaged ``torch`` distribution directory this finder serves."""

    return pathlib.Path(__file__).resolve().parent / "resources" / "torch_dist_info"


def version():
    """Read the advertised version from the packaged ``METADATA``.

    The deployment helper reads the same field; keeping one on-disk source of
    truth stops the two publication paths from advertising different versions.
    """

    path = resource_root() / "METADATA"
    with open(os.fspath(path), encoding="utf-8") as handle:
        for line in handle:
            key, separator, value = line.partition(":")
            if separator and key.strip().lower() == "version" and value.strip():
                return value.strip()
    raise RuntimeError("torch shim metadata has no Version field: %s" % path)


def _metadata_module():
    try:
        from importlib import metadata
    except ImportError:  # Python 3.7: only the backport exists.
        try:
            import importlib_metadata as metadata
        except ImportError:
            return None
    return metadata


def _name_matches(name):
    if name is None:
        return True
    return str(name).lower().replace("-", "_") == _NAME


def _path_matches(paths, root):
    # The process-local distribution is not part of any directory scan.  Answer
    # the default ``sys.path`` query, and a scan that explicitly includes the
    # packaged resource root; a scan of an unrelated directory (a deployment
    # target, another environment) must not see it.
    if paths is None or paths is sys.path:
        return True
    try:
        wanted = os.path.realpath(os.fspath(root))
        return any(os.path.realpath(os.fspath(entry)) == wanted for entry in paths)
    except (OSError, TypeError):
        return False


class TorchDistributionFinder(object):
    """Serve one ``torch`` distribution from the packaged shim resources.

    ``importlib.metadata`` discovers any finder that exposes
    ``find_distributions``.  ``find_spec`` declines every import explicitly:
    some CPython 3.11 builds fall back to the removed ``find_module`` protocol
    for a finder that has no ``find_spec``, and a missing method there raises
    instead of being skipped.
    """

    def __init__(self, root):
        self.root = pathlib.Path(root)

    def find_spec(self, name, path=None, target=None):
        return None

    def find_distributions(self, context=None):
        metadata = _metadata_module()
        if metadata is None:
            return iter(())
        name = getattr(context, "name", None)
        if not _name_matches(name):
            return iter(())
        paths = getattr(context, "path", None)
        if not _path_matches(paths, self.root.parent):
            return iter(())
        return iter((metadata.PathDistribution(self.root),))


def publish():
    """Publish the shim's ``torch`` distribution for this process.

    Idempotent: repeated activation returns the finder already installed.
    """

    global _finder
    if _finder is not None and _finder in sys.meta_path:
        return _finder
    root = resource_root()
    if not (root / "METADATA").is_file():
        raise RuntimeError(
            "the packaged torch shim distribution is missing: %s" % root)
    finder = TorchDistributionFinder(root)
    # First so this process-local answer wins over a stale torch dist-info that
    # happens to be on sys.path.  ``find_spec`` returns None, so ordinary
    # imports pass through to the real import finders.
    sys.meta_path.insert(0, finder)
    _finder = finder
    return finder


def release(finder=None):
    """Remove a finder published by :func:`publish`; idempotent."""

    global _finder
    target = finder if finder is not None else _finder
    if target is None:
        return False
    removed = False
    while target in sys.meta_path:
        sys.meta_path.remove(target)
        removed = True
    if _finder is target:
        _finder = None
    return removed


def is_published():
    return _finder is not None and _finder in sys.meta_path
