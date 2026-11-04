"""Data-only ``torch.package`` archives for the Jittor shim.

Upstream ``torch.package`` is a hermetic code packager: it copies Python
source into a zip archive, rewrites imports through a private importer, and
serializes objects with a pickler that records every referenced module.  This
module implements the *data container* half of that API and refuses the
code-packaging half with an explicit error.

Supported
---------
* :class:`PackageExporter` / :class:`PackageImporter`, as context managers or
  plain objects, over a path or a binary file-like object.
* ``save_pickle`` / ``load_pickle``.  Objects are written with the shim's
  ``torch.save`` codec, so tensors keep their dtype and shape, and
  ``map_location`` moves them on load.
* ``save_text`` / ``load_text`` and ``save_binary`` / ``load_binary``.
* :class:`Importer`, :class:`OrderedImporter` and ``sys_importer``.  These are
  plain Python helpers and do not depend on packaged source.

Not supported
-------------
* Python source packaging.  ``intern``, ``mock``, ``extern``, ``deny``,
  ``save_source_file``, ``save_source_string``, ``save_module``,
  ``add_dependency``, the action hooks, and ``PackageImporter.import_module``
  raise :class:`NotImplementedError` with a message naming the missing
  behavior.  Archives are therefore not hermetic: classes and functions are
  resolved from the interpreting environment when a pickle is loaded.
* Binary compatibility with archives written by PyTorch's ``torch.package``.
  This module reads and writes its own documented zip layout; an archive
  without that layout is rejected with an :class:`ImportError` saying so.

Archive layout
--------------
The archive is a zip file with these records::

    version            -- "jittor-torch-package-1"
    manifest.json      -- {"format": ..., "python_version": ..., "entries": [...]}
    data/<pkg path>/<resource>

``<pkg path>`` replaces dots in the package name with ``/``.  Each manifest
entry records the package, resource, kind (``pickle``/``text``/``binary``),
record path, byte size and SHA-256 digest.  The manifest is written by
``close()``; an exporter abandoned without closing leaves a readable zip that
is rejected on import because the manifest is missing.

Security
--------
``load_pickle`` unpickles data; treat an archive like any other pickle file
and only load archives you trust.  ``torch.load`` runs with
``weights_only=False`` so arbitrary Python objects round-trip.
"""
from __future__ import absolute_import

import abc
import builtins
import collections
import collections.abc
import hashlib
import importlib
import io
import itertools
import json
import os
import sys
import weakref
import zipfile


__all__ = [
    "PackageExporter",
    "PackageImporter",
    "Importer",
    "OrderedImporter",
    "sys_importer",
    "ObjNotFoundError",
    "ObjMismatchError",
    "PackagingError",
    "EmptyMatchError",
]

_FORMAT = "jittor-torch-package-1"
_VERSION_RECORD = "version"
_MANIFEST_RECORD = "manifest.json"
_DATA_PREFIX = "data/"
_PICKLE_PROTOCOLS = (3, 4)
_IMPORTER_IDS = itertools.count()


class ObjNotFoundError(Exception):
    """An importer could not find the module or attribute naming an object."""


class ObjMismatchError(Exception):
    """The importer found a different object under the requested name."""


class PackagingError(Exception):
    """A package could not be assembled.

    Upstream's constructor takes a dependency graph and a debug flag; this
    implementation has no source dependency graph, so the class exists for
    ``except``-clause compatibility and carries a message.
    """


class EmptyMatchError(PackagingError):
    """A pattern matched no source modules.

    Source patterns are not implemented here; the class exists so callers that
    import it keep working.
    """


def _resolve_attr(module, name):
    value = module
    for atom in name.split("."):
        value = getattr(value, atom)
    return value


class Importer(abc.ABC):
    """An environment that can import modules and name objects by module path.

    The contract is the one upstream ``torch.package`` documents: the tuple
    returned by :meth:`get_name` imports back to the same object from this
    environment.  Implementations differ only in where modules come from.
    """

    modules = {}

    @abc.abstractmethod
    def import_module(self, module_name):
        """Import ``module_name`` from this environment."""

    def get_name(self, obj, name=None):
        """Return ``(module_name, attribute_name)`` that resolves to ``obj``."""
        if name is None:
            name = getattr(obj, "__qualname__", None)
        if name is None:
            name = getattr(obj, "__name__", None)
        if name is None:
            raise ObjNotFoundError("%r has no __qualname__ or __name__" % (obj,))
        module_name = self.whichmodule(obj, name)
        try:
            module = self.import_module(module_name)
            found = _resolve_attr(module, name)
        except (ImportError, KeyError, AttributeError):
            raise ObjNotFoundError(
                "%r was not found as %s.%s" % (obj, module_name, name)
            ) from None
        if found is not obj:
            raise ObjMismatchError(
                "%r was found at %s.%s, but that name resolves to a different "
                "object from another environment"
                % (obj, module_name, name))
        return module_name, name

    def whichmodule(self, obj, name):
        """Return the module name ``obj`` belongs to."""
        module_name = getattr(obj, "__module__", None)
        if module_name is not None:
            return module_name
        for candidate, module in list(self.modules.copy().items()):
            if candidate in ("__main__", "__mp_main__") or module is None:
                continue
            try:
                if _resolve_attr(module, name) is obj:
                    return candidate
            except AttributeError:
                continue
        return "__main__"


class _SysImporter(Importer):
    """The default environment: the ordinary Python import system."""

    def import_module(self, module_name):
        return importlib.import_module(module_name)


sys_importer = _SysImporter()
"""Importer that searches the ordinary Python environment."""


class OrderedImporter(Importer):
    """Try a sequence of importers in order; the first answer wins."""

    def __init__(self, *importers):
        if not importers:
            raise ValueError("OrderedImporter requires at least one importer")
        for importer in importers:
            if not isinstance(importer, Importer):
                raise TypeError(
                    "OrderedImporter arguments must be Importer instances, got %r"
                    % (type(importer),))
        self._importers = tuple(importers)

    def import_module(self, module_name):
        error = None
        for importer in self._importers:
            try:
                return importer.import_module(module_name)
            except ImportError as exc:
                if error is None:
                    error = exc
        raise error if error is not None else ImportError(module_name)

    def get_name(self, obj, name=None):
        error = None
        for importer in self._importers:
            try:
                return importer.get_name(obj, name)
            except (ObjNotFoundError, ObjMismatchError) as exc:
                if error is None:
                    error = exc
        raise error if error is not None else ObjNotFoundError(repr(obj))

    def whichmodule(self, obj, name):
        error = None
        for importer in self._importers:
            try:
                return importer.whichmodule(obj, name)
            except (ObjNotFoundError, ObjMismatchError) as exc:
                if error is None:
                    error = exc
        raise error if error is not None else ObjNotFoundError(repr(obj))


def _live_torch():
    """Return the installed Torch namespace without importing a foreign torch."""
    import jittor as jt
    from ..context import get_install_context

    return get_install_context(jt).target_namespace


def _validate_package(package):
    if not isinstance(package, str) or not package:
        raise ValueError("package must be a non-empty string, got %r" % (package,))
    for atom in package.split("."):
        if not atom or atom in (".", "..") or "/" in atom or "\\" in atom or "\0" in atom:
            raise ValueError(
                "invalid package %r: package names are dot-separated path-safe "
                "components" % (package,))
    return package


def _validate_resource(resource):
    if not isinstance(resource, str) or not resource:
        raise ValueError("resource must be a non-empty string, got %r" % (resource,))
    if "\\" in resource or "\0" in resource or resource.startswith("/"):
        raise ValueError(
            "invalid resource %r: use '/'-separated relative path components"
            % (resource,))
    if any(atom in ("", ".", "..") for atom in resource.split("/")):
        raise ValueError(
            "invalid resource %r: empty, '.' and '..' path components are "
            "not allowed" % (resource,))
    return resource


def _record_path(package, resource):
    return _DATA_PREFIX + package.replace(".", "/") + "/" + resource


class PackageExporter(object):
    """Write a data-only ``torch.package`` archive.

    The exporter is usable as a context manager; ``close()`` writes the
    manifest that makes the archive loadable.  If the ``with`` block raises,
    the zip is closed without a manifest so a half-written export can never be
    mistaken for a complete one.
    """

    def __init__(self, f, importer=sys_importer, debug=False):
        self.debug = bool(debug)
        if isinstance(importer, Importer):
            self.importer = importer
        elif isinstance(importer, collections.abc.Sequence):
            self.importer = OrderedImporter(*importer)
        else:
            raise TypeError(
                "importer must be an Importer or a sequence of Importers, got %r"
                % (type(importer),))
        if isinstance(f, (str, os.PathLike)):
            self.buffer = None
            target = os.fspath(f)
        elif hasattr(f, "write"):
            self.buffer = f
            target = f
        else:
            raise TypeError(
                "PackageExporter expects a filename or a binary file-like "
                "object, got %r" % (type(f),))
        self._zip = zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED)
        self._zip.writestr(_VERSION_RECORD, _FORMAT + "\n")
        self._entries = collections.OrderedDict()
        self._written = set()
        self._closed = False
        self._unique_id = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            self._close_without_manifest()
            return False
        self.close()
        return False

    def get_unique_id(self):
        """Return a process-unique ``<torch_package_N>`` style identifier."""
        self._unique_id += 1
        return "<torch_package_%d>" % (self._unique_id,)

    def save_pickle(self, package, resource, obj, dependencies=True,
                    pickle_protocol=3):
        """Serialize ``obj`` with ``torch.save`` into the archive.

        ``dependencies`` is accepted for signature compatibility but Python
        source dependency scanning is not implemented; loading resolves the
        object's classes from the interpreting environment.  ``pickle_protocol``
        must be 3 or 4, matching upstream.
        """
        if pickle_protocol not in _PICKLE_PROTOCOLS:
            raise AssertionError(
                "torch.package only supports pickle protocols 3 and 4, got %r"
                % (pickle_protocol,))
        buffer = io.BytesIO()
        _live_torch().save(obj, buffer, pickle_protocol=pickle_protocol)
        self._write(package, resource, buffer.getvalue(), "pickle")

    def save_text(self, package, resource, text):
        """Store UTF-8 text ``text`` under ``package.resource``."""
        if not isinstance(text, str):
            raise TypeError("text must be str, got %r" % (type(text),))
        self._write(package, resource, text.encode("utf-8"), "text")

    def save_binary(self, package, resource, binary):
        """Store raw bytes ``binary`` under ``package.resource``."""
        if not isinstance(binary, (bytes, bytearray, memoryview)):
            raise TypeError("binary must be bytes-like, got %r" % (type(binary),))
        self._write(package, resource, bytes(binary), "binary")

    def _write(self, package, resource, payload, kind):
        if self._closed:
            raise ValueError("cannot save into a closed PackageExporter")
        _validate_package(package)
        _validate_resource(resource)
        path = _record_path(package, resource)
        if path in self._written:
            raise ValueError(
                "resource %r in package %r resolves to %r, which was already "
                "written to this archive" % (resource, package, path))
        self._zip.writestr(path, payload)
        self._written.add(path)
        self._entries[(package, resource)] = {
            "package": package,
            "resource": resource,
            "kind": kind,
            "path": path,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _manifest(self):
        return {
            "format": _FORMAT,
            "python_version": sys.version,
            "entries": list(self._entries.values()),
        }

    def close(self):
        """Finish the archive.  Idempotent."""
        if self._closed:
            return
        manifest = json.dumps(self._manifest(), indent=1, sort_keys=True)
        self._zip.writestr(_MANIFEST_RECORD, manifest.encode("utf-8"))
        self._zip.close()
        self._closed = True
        if self.buffer is not None:
            self.buffer.flush()

    def _close_without_manifest(self):
        if self._closed:
            return
        self._zip.close()
        self._closed = True
        if self.buffer is not None:
            self.buffer.flush()

    def _unsupported(self, operation):
        raise NotImplementedError(
            "torch.package.PackageExporter.%s is not implemented by the Jittor "
            "shim: it would copy Python source into the archive and rewrite "
            "imports through a private importer. This implementation stores "
            "data resources only (save_pickle/save_text/save_binary)."
            % (operation,))

    def intern(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("intern")

    def mock(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("mock")

    def extern(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("extern")

    def deny(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("deny")

    def save_source_file(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("save_source_file")

    def save_source_string(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("save_source_string")

    def save_module(self, *args, **kwargs):
        """Unsupported: Python source packaging."""
        self._unsupported("save_module")

    def add_dependency(self, *args, **kwargs):
        """Unsupported: dependency scanning packages Python source."""
        self._unsupported("add_dependency")

    def register_extern_hook(self, *args, **kwargs):
        """Unsupported: hooks fire only while packaging source."""
        self._unsupported("register_extern_hook")

    def register_mock_hook(self, *args, **kwargs):
        """Unsupported: hooks fire only while packaging source."""
        self._unsupported("register_mock_hook")

    def register_intern_hook(self, *args, **kwargs):
        """Unsupported: hooks fire only while packaging source."""
        self._unsupported("register_intern_hook")


class PackageImporter(Importer):
    """Read a data-only ``torch.package`` archive written above.

    The importer knows no Python modules: ``import_module`` and
    ``import_module``-driven ``get_name`` raise :class:`NotImplementedError`
    because source packaging is unsupported.  Use ``load_pickle``,
    ``load_text`` and ``load_binary`` for the archive's contents.
    """

    def __init__(self, file_or_buffer, module_allowed=lambda module_name: True):
        if not callable(module_allowed):
            raise TypeError("module_allowed must be callable")
        if isinstance(file_or_buffer, (str, os.PathLike)):
            self.filename = os.fspath(file_or_buffer)
            if os.path.isdir(self.filename):
                raise NotImplementedError(
                    "PackageImporter does not read unpacked package "
                    "directories; pass the zip archive")
            source = self.filename
        elif isinstance(file_or_buffer, (bytes, bytearray, memoryview)):
            self.filename = "<binary>"
            source = io.BytesIO(bytes(file_or_buffer))
        elif hasattr(file_or_buffer, "read"):
            self.filename = "<binary>"
            source = file_or_buffer
        else:
            raise TypeError(
                "PackageImporter expects a filename, bytes or a binary "
                "file-like object, got %r" % (type(file_or_buffer),))
        try:
            self._zip = zipfile.ZipFile(source, "r")
        except zipfile.BadZipFile as exc:
            raise ImportError(
                "torch.package.PackageImporter: %r is not a zip archive, so it "
                "is not a torch.package archive" % (self.filename,)) from exc
        self.modules = {}
        self.extern_modules = []
        try:
            self._entries = self._read_manifest()
        except ImportError:
            self._zip.close()
            raise
        self._id = "<torch_package_%d>" % (next(_IMPORTER_IDS))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self):
        """Close the underlying zip file."""
        self._zip.close()

    def _read_manifest(self):
        try:
            raw = self._zip.read(_MANIFEST_RECORD)
        except KeyError:
            raise ImportError(
                "torch.package.PackageImporter: archive %r has no %s record. "
                "It was either written by an exporter that never reached "
                "close(), or it is a PyTorch torch.package archive, whose "
                "format this Jittor implementation does not read."
                % (self.filename, _MANIFEST_RECORD)) from None
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise ImportError(
                "torch.package.PackageImporter: archive %r has an unreadable "
                "%s record" % (self.filename, _MANIFEST_RECORD)) from exc
        if not isinstance(data, dict) or data.get("format") != _FORMAT:
            raise ImportError(
                "torch.package.PackageImporter: archive %r declares format %r; "
                "this Jittor implementation reads %r only"
                % (self.filename,
                   data.get("format") if isinstance(data, dict) else None,
                   _FORMAT))
        entries = {}
        record_paths = {}
        for item in data.get("entries", []):
            if not isinstance(item, dict):
                raise ImportError(
                    "torch.package.PackageImporter: archive %r has a malformed "
                    "manifest entry" % (self.filename,))
            try:
                package = _validate_package(item.get("package"))
                resource = _validate_resource(item.get("resource"))
            except ValueError as exc:
                raise ImportError(
                    "torch.package.PackageImporter: archive %r has a malformed "
                    "manifest entry: %s" % (self.filename, exc)) from exc
            expected = _record_path(package, resource)
            if item.get("path") != expected:
                raise ImportError(
                    "torch.package.PackageImporter: archive %r maps %s.%s to %r, "
                    "expected %r"
                    % (self.filename, package, resource, item.get("path"),
                       expected))
            if expected in record_paths:
                raise ImportError(
                    "torch.package.PackageImporter: archive %r records %r twice "
                    "(%s.%s and %s.%s)"
                    % (self.filename, expected, record_paths[expected][0],
                       record_paths[expected][1], package, resource))
            record_paths[expected] = (package, resource)
            entries[(package, resource)] = item
        self._python_version = data.get("python_version")
        return entries

    def _entry(self, package, resource):
        _validate_package(package)
        _validate_resource(resource)
        entry = self._entries.get((package, resource))
        if entry is None:
            raise KeyError(
                "resource %r in package %r is not present in this archive"
                % (resource, package))
        return entry

    def load_binary(self, package, resource):
        """Return the raw bytes stored under ``package.resource``."""
        entry = self._entry(package, resource)
        return self._zip.read(entry["path"])

    def load_text(self, package, resource, encoding="utf-8", errors="strict"):
        """Return the stored bytes decoded as text."""
        return self.load_binary(package, resource).decode(encoding, errors)

    def load_pickle(self, package, resource, map_location=None):
        """Unpickle ``package.resource`` with ``torch.load``.

        ``map_location`` is forwarded to ``torch.load``.  Unpickling executes
        the pickle's reconstruction code; only load trusted archives.
        """
        entry = self._entry(package, resource)
        if entry.get("kind") != "pickle":
            raise ValueError(
                "resource %r in package %r was saved as %r, not a pickle"
                % (resource, package, entry.get("kind")))
        return _live_torch().load(
            io.BytesIO(self._zip.read(entry["path"])),
            map_location=map_location,
            weights_only=False,
        )

    def id(self):
        """Return this importer's ``<torch_package_N>`` identifier."""
        return self._id

    def python_version(self):
        """Return the ``sys.version`` string recorded by the exporter."""
        return self._python_version

    def file_structure(self, **kwargs):
        """Unsupported: no directory tree API is implemented."""
        raise NotImplementedError(
            "torch.package.PackageImporter.file_structure is not implemented; "
            "the archive's manifest lists its data resources")

    def import_module(self, name, package=None):
        """Unsupported: archives contain data, not Python source modules."""
        raise NotImplementedError(
            "torch.package.PackageImporter.import_module is not implemented by "
            "the Jittor shim: torch.package code packaging "
            "(intern/save_source_string/save_module) is unsupported, so an "
            "archive contains no Python modules. Use load_pickle/load_text/"
            "load_binary for data resources.")

    def __import__(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Delegate external imports; relative imports need packaged source."""
        if level:
            raise NotImplementedError(
                "torch.package.PackageImporter.__import__ does not support "
                "relative imports: torch.package code packaging is unsupported")
        return builtins.__import__(name, globals, locals, fromlist, level)


#: Modules loaded from package importers, mirroring upstream's private
#: registry.  Source packaging is unsupported, so this stays empty; it exists
#: because torch._dynamo reads it when it encounters package-imported modules.
_package_imported_modules = weakref.WeakValueDictionary()
