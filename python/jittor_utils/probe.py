# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
'''On-disk cache for the toolchain probes ``import jittor`` runs.

Importing Jittor used to shell out more than a dozen times before doing any
work of its own: ``g++ --version``, ``git branch``, ``python3-config
--includes`` and ``--extension-suffix``, ``nvcc --version`` once per CUDA
library (six times), ``gdb --version``, ``addr2line --version``, ``mpicc
--showme:*`` three times, ``hipcc --version``, ``nvidia-smi -q -u``, and a
whole second interpreter to read the GPUs' compute capabilities. None of the
answers were remembered, so every process paid for all of them -- and every one
of them was a place the import could fail or hang. The compute-capability child
is the one that once deadlocked against the build lock its own parent was
holding.

Those answers change only when the tools do. This module keeps them in
``<jittor home>/.cache/jittor/probe.json``, keyed by the probe and invalidated
by the size and mtime of the files the probe asks about. A probe whose files
cannot be stamped is never cached, so "the answer might change for reasons we
cannot see" degrades to today's behaviour rather than to a wrong answer.

The file deliberately sits at the *root* of the cache rather than inside
``cache_path``: the compiler version and the git branch are inputs to the
computation of ``cache_path`` itself, so they have to be answerable before it
exists.

Set ``JT_PROBE_CACHE=0`` to bypass it entirely.
'''

import json
import os

VERSION = 1

# Probes this process actually had to run. Tests read it to assert that a warm
# import spawns nothing.
MISSES = 0

_entries = None


def disabled():
    return os.environ.get("JT_PROBE_CACHE", "1") == "0"


def cache_file():
    from jittor_utils import home
    return os.path.join(home(), ".cache", "jittor", "probe.json")


def _stamp(paths):
    '''(path, size, mtime) for each file a probe's answer depends on.

    A path that does not exist stamps as nulls, which is itself a state: a
    compiler that appears or disappears re-runs the probe.
    '''
    stamps = []
    for path in paths:
        try:
            st = os.stat(path)
            stamps.append([path, st.st_size, st.st_mtime_ns])
        except OSError:
            stamps.append([path, None, None])
    return stamps


def _read_file():
    try:
        with open(cache_file()) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict) or data.get("version") != VERSION:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def _load():
    global _entries
    if _entries is None:
        _entries = _read_file()
    return _entries


def _store(key, record):
    '''Merge one entry into the file on disk and replace it atomically.

    Re-reading first means two processes probing different tools at the same
    time keep both answers; the worst a lost update costs is one re-probe.
    '''
    path = cache_file()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except OSError:
        return
    entries = _read_file()
    entries[key] = record
    temporary = "%s.%d.tmp" % (path, os.getpid())
    try:
        with open(temporary, "w") as f:
            json.dump({"version": VERSION, "entries": entries}, f)
        os.replace(temporary, path)
    except OSError:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def cached(key, stamp_paths, compute, extra=None):
    '''Return ``compute()``, remembering it against the state of ``stamp_paths``.

    ``extra`` is any additional JSON-able value that must match for the cached
    answer to still apply -- a driver version string, say, for a probe that
    asks the driver rather than a file.

    The value must be JSON-able: this cache is shared between processes, not a
    memo table.
    '''
    global MISSES
    if disabled():
        MISSES += 1
        return compute()
    stamp = _stamp(stamp_paths)
    entries = _load()
    hit = entries.get(key)
    if isinstance(hit, dict) and "value" in hit \
            and hit.get("stamp") == stamp and hit.get("extra") == extra:
        return hit["value"]
    MISSES += 1
    record = {"stamp": stamp, "extra": extra, "value": compute()}
    entries[key] = record
    _store(key, record)
    return record["value"]


def forget():
    '''Drop the cache, in this process and on disk. For tests and clean_cache.'''
    global _entries
    _entries = None
    try:
        os.unlink(cache_file())
    except OSError:
        pass
