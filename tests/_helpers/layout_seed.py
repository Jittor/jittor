"""Preserve sampled inputs across the reviewed 10.23 path-only migration.

This changes only the generator seed key, never pytest ids or selection. The
frozen map applies to pre-existing test methods; newly added methods/files use
their canonical new path. Parameterization and device suffixes are preserved.
"""

import json
from functools import lru_cache
from pathlib import Path

from _helpers.seed_case_identity import source_cases

_ROOT = Path(__file__).resolve().parents[2]
_MOVED = json.loads(Path(__file__).with_name("layout_seed_paths.json").read_text())


@lru_cache(maxsize=None)
def _explicit_current_cases(filename, modified_ns, size):
    return source_cases(Path(filename).read_text())[0]


def _known_case(previous, suffix, source):
    identity = suffix.split("[", 1)[0]
    owner, separator, method = identity.rpartition("::")
    original_owner = previous["device_classes"].get(owner, owner)
    canonical = original_owner + separator + method
    if canonical in previous["cases"] or identity in previous.get("dynamic_cases", ()):
        return True
    # A newly written concrete method must not inherit a former generator's
    # seed just because its spelling begins with that generator's name.
    try:
        stat = source.stat()
    except OSError:
        return False
    if canonical in _explicit_current_cases(str(source), stat.st_mtime_ns, stat.st_size):
        return False
    return any(canonical.startswith(family + "_") for family in previous["families"])


def seed_nodeid(nodeid, path=None):
    filename, separator, suffix = nodeid.partition("::")
    if path is not None:
        try:
            filename = Path(str(path)).resolve().relative_to(_ROOT).as_posix()
        except ValueError:
            pass
    previous = _MOVED.get(filename)
    if previous is not None and _known_case(previous, suffix, _ROOT / filename):
        filename = previous["path"]
    return filename + separator + suffix
