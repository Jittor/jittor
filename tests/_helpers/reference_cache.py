"""Persist the *device-independent* half of a comparison between gate runs.

The device parity battery computes every result twice: once on the CPU, which is
the oracle, and once on the accelerator, which is the thing under test. The CPU
half is a pure function of (operator, inputs, the code that implements it) -- it
does not depend on which accelerator is present, or on there being one -- so
across two runs of the same checkout it recomputes an answer it already has.

This module stores that answer. It is deliberately paranoid, because a cache in
front of an oracle is a way to make a gate answer about *yesterday's* code
while looking exactly like a gate that answered about today's:

* **The key contains a content hash of the implementation** (:func:`source_fingerprint`
  over ``python/jittor``). Any edit to the tree -- Python or C++ -- misses every
  entry, so a cached oracle can only ever have been produced by the byte-identical
  source. This is why the first run after any commit pays full price, on purpose.
* **The key contains the inputs themselves**, hashed from the materialized bytes
  rather than from the seed that produced them. A change to sample generation
  therefore misses instead of silently comparing against the wrong inputs.
* **The entry re-states its own key material** and a reader that finds a
  mismatch treats the entry as absent, so a hash collision or a misplaced file
  cannot substitute one operator's oracle for another's.
* **Entries are written to a temporary file and renamed** (``os.replace``): an
  interrupted or concurrent write is never observed half-finished (9.20, 9.22).
* **Every read and write is counted** and the totals are reported, because
  "green" must not be able to mean "answered from a cache nobody noticed".

Location: under the Jittor build-configuration cache directory, which already
carries the compiler, Python, architecture and flag fingerprint (0.07), so
entries never cross build configurations. Nothing is written into the checkout.

Environment
-----------
=================================== ==========================================
``JITTOR_REFERENCE_CACHE=0``        recompute everything; write nothing
``JITTOR_REFERENCE_CACHE_DIR=DIR``  store entries under ``DIR`` instead
=================================== ==========================================
"""

import hashlib
import json
import os
from pathlib import Path
import zipfile

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bumped when the *meaning* of a payload changes (what is stored, or how the
#: value is produced). Old entries then miss rather than being reinterpreted.
SCHEMA_VERSION = "reference-cache-v1"

ENABLE_VARIABLE = "JITTOR_REFERENCE_CACHE"
DIRECTORY_VARIABLE = "JITTOR_REFERENCE_CACHE_DIR"

#: What counts as "the implementation" for the fingerprint. Everything that can
#: change a number the CPU produces: the Python layer, the C++ core, the kernel
#: sources and the headers they are generated from.
_FINGERPRINT_SUFFIXES = (
    ".py", ".cc", ".cu", ".cuh", ".h", ".hpp", ".inc", ".tcc",
)
_FINGERPRINT_SKIP_PARTS = ("__pycache__", ".git")

_fingerprint = None


def source_fingerprint(root=None):
    """A content hash of the implementation under ``python/jittor``.

    Content, not mtime: a checkout or a rebase rewrites mtimes without changing
    a single number, and an mtime-keyed cache would throw away every entry on
    every branch switch. Computed once per process (a few hundred files).
    """
    global _fingerprint
    if root is None and _fingerprint is not None:
        return _fingerprint
    base = Path(root) if root is not None else REPO_ROOT / "python" / "jittor"
    digest = hashlib.blake2b(digest_size=16)
    for path in sorted(base.rglob("*")):
        if not path.is_file() or path.suffix not in _FINGERPRINT_SUFFIXES:
            continue
        if any(part in _FINGERPRINT_SKIP_PARTS for part in path.parts):
            continue
        digest.update(path.relative_to(base).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    value = digest.hexdigest()
    if root is None:
        _fingerprint = value
    return value


def _contiguous(value):
    """``ascontiguousarray`` without its 0-d surprise.

    ``np.ascontiguousarray(np.array(7.0))`` returns shape ``(1,)``: a documented
    legacy behaviour, and a silent one. A full reduction's oracle is 0-d, so
    left alone it would be *stored* as a one-element vector, come back with the
    wrong shape, and be compared against a 0-d accelerator result -- which
    broadcasts, so the comparison would still pass. That is the exact failure
    mode this cache is not allowed to have.
    """
    array = np.asarray(value)
    if array.ndim == 0:
        return array if array.flags["C_CONTIGUOUS"] else array.copy()
    return np.ascontiguousarray(array)


def value_digest(value):
    """A stable description of one key component, arrays included by content.

    ``repr`` is not enough for an array: it truncates, and two different arrays
    can print the same. dtype and shape are part of the description because a
    reshaped or re-typed input is a different question.
    """
    if isinstance(value, np.ndarray):
        contiguous = _contiguous(value)
        return "ndarray(%s,%s,%s)" % (
            contiguous.dtype.str, contiguous.shape,
            hashlib.blake2b(contiguous.tobytes(), digest_size=16).hexdigest())
    if isinstance(value, (list, tuple)):
        return "[%s]" % ",".join(value_digest(item) for item in value)
    if isinstance(value, dict):
        return "{%s}" % ",".join(
            "%r:%s" % (name, value_digest(value[name])) for name in sorted(value))
    if hasattr(value, "numpy") and not isinstance(value, (str, bytes)):
        # A Var reaching a cache key would key on its identity; materialize it.
        return value_digest(np.asarray(value.numpy()))
    return repr(value)


def key_material(*parts):
    """The human-readable string a key hashes, stored alongside the entry."""
    return "|".join([SCHEMA_VERSION, source_fingerprint()]
                    + [value_digest(part) for part in parts])


def _cache_root():
    configured = os.environ.get(DIRECTORY_VARIABLE, "").strip()
    if configured:
        return Path(configured).expanduser()
    try:
        import jittor as jt
        base = jt.compiler.cache_path
    except (ImportError, AttributeError):
        # No jittor, or a build without a cache path: the cache is inactive and
        # every lookup misses. Never a failure -- see `load`.
        return None
    if not base:
        return None
    # cache_path already ends in the build-configuration fingerprint (0.07).
    return Path(base) / "reference_cache"


def enabled():
    return os.environ.get(ENABLE_VARIABLE, "1").strip() not in ("0", "off", "no")


#: Every cache built in this process, so the session can report what it reused.
#:
#: "Green" must not be able to mean "answered from a cache nobody mentioned":
#: ``tests/conftest.py`` prints these lines in the terminal summary, which is
#: the same discipline as the per-file execution accounting in 0.18.
_REGISTRY = []


def registry():
    return tuple(_REGISTRY)


class ReferenceCache:
    """Named store for one battery's device-independent results.

    ``disabled`` instances are usable and always miss, so a call site needs no
    branch of its own: the statistics still say what happened.
    """

    def __init__(self, name, root=None):
        self.name = name
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.rejected = 0
        self.errors = []
        if root is not None:
            self.directory = Path(root)
        else:
            base = _cache_root()
            self.directory = None if base is None else base / name
        self.active = bool(self.directory) and enabled()
        _REGISTRY.append(self)

    # -- paths -----------------------------------------------------------
    def _path(self, digest):
        # Two levels: 65k entries in one directory is slow to list and slow to
        # stat on some filesystems.
        return self.directory / digest[:2] / (digest[2:] + ".npz")

    @staticmethod
    def digest(material):
        return hashlib.blake2b(material.encode("utf-8"),
                                   digest_size=20).hexdigest()

    # -- read/write ------------------------------------------------------
    def load(self, material):
        """The stored arrays for ``material``, or ``None``.

        Returns ``(values, extras)`` where ``values`` is the list of arrays and
        ``extras`` the JSON-able side data handed to :meth:`store`.
        """
        if not self.active:
            self.misses += 1
            return None
        path = self._path(self.digest(material))
        if not path.is_file():
            self.misses += 1
            return None
        try:
            with np.load(str(path), allow_pickle=False) as data:
                stored = str(data["key_material"].item())
                if stored != material:
                    # Collision, or a file that belongs to another question.
                    # Recomputing is always correct; trusting this is not.
                    self.rejected += 1
                    self.misses += 1
                    return None
                extras = json.loads(str(data["extras"].item()))
                count = int(data["count"].item())
                values = [np.array(data["value%d" % index])
                          for index in range(count)]
        except (OSError, ValueError, KeyError, EOFError,
                zipfile.BadZipFile, json.JSONDecodeError) as error:
            # A truncated or foreign file is a miss, never a failure: the cache
            # is an optimisation and must not be able to fail a gate.
            self.rejected += 1
            self.misses += 1
            self.errors.append("%s: %s" % (path.name, error))
            return None
        self.hits += 1
        return values, extras

    def store(self, material, values, extras=None):
        if not self.active:
            return False
        path = self._path(self.digest(material))
        payload = {
            "key_material": np.array(material),
            "extras": np.array(json.dumps(extras if extras is not None else {})),
            "count": np.array(len(values)),
        }
        for index, value in enumerate(values):
            payload["value%d" % index] = _contiguous(value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(
                ".%s.%d.partial.npz" % (path.stem, os.getpid()))
            with open(str(temporary), "wb") as handle:
                np.savez(handle, **payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(str(temporary), str(path))
        except OSError as error:
            # Out of space, read-only cache directory: the run continues
            # uncached rather than failing on a missing optimisation.
            self.errors.append("write %s: %s" % (path.name, error))
            return False
        self.writes += 1
        return True

    # -- reporting -------------------------------------------------------
    def summary(self):
        """One line, for a run to state what it answered from where."""
        if not self.active:
            reason = ("disabled by %s" % ENABLE_VARIABLE if not enabled()
                      else "no cache directory available")
            return "reference cache [%s]: inactive (%s); %d oracle values computed" % (
                self.name, reason, self.misses)
        line = ("reference cache [%s]: %d reused, %d computed, %d stored, dir=%s"
                % (self.name, self.hits, self.misses, self.writes, self.directory))
        if self.rejected:
            line += " (%d entries rejected)" % self.rejected
        if self.errors:
            line += "; first problem: %s" % self.errors[0]
        return line
