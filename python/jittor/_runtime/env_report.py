# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""What the environment configured, said once, at startup.

A process whose behaviour was changed by a variable in its environment had no
way to find that out. The core printed one ``LOGi`` line per flag it read
(``"Load log_v: 1"``), which the default ``log_v=0`` and ``log_silent`` both
hide, and the Python build variables printed nothing at all. So "jittor behaves
differently on this machine" was an unanswerable question, and
``export debug=1`` -- a word any CI system might export for its own reasons --
changed the whole framework's build silently.

Four things get reported here, once each, after every static initializer and
every build variable has been read:

* one summary line naming every setting the environment supplied, and which
  variable each came from;
* one ``DeprecationWarning`` listing the unprefixed names that were used, with
  the ``JT_``/``JT_BUILD_`` name to move to;
* one ``DeprecationWarning`` for unprefixed names that were **ignored** because
  they are ordinary words (``debug``), so that a user who did mean the jittor
  switch is told which name to use rather than silently getting the default;
* one ``UserWarning`` for ``JT_*`` variables jittor does not read at all --
  ``JT_CACHE_NAME`` for ``JT_BUILD_CACHE_NAME`` used to do nothing, silently.

Kept out of ``jittor/__init__.py`` so it can be tested without importing the
core: :func:`format_report` and :func:`format_legacy_warning` are pure
functions of the rows the core hands over.
"""

import difflib
import os
import warnings
from typing import Dict, Tuple

from jittor_utils import env_config

from .flag_policy import STARTUP_FLAGS


#: ``JT_*`` variables read by name somewhere other than the native flag table
#: and ``env_config``: backend rendezvous, lock tuning, build macros taken from
#: the environment. Together with those two they are every ``JT_*`` name
#: jittor reads; ``tests/structure/runtime/test_env_variable_registry.py``
#: checks the source tree against this list.
DIRECT_VARIABLES = frozenset((
    "JT_BACKEND", "JT_CHECK_NAN", "JT_DEFAULT_PARA_OPT_LEVEL",
    "JT_GRAPH_BUILD_PROFILE", "JT_HAS_HALF_SIMD",
    "JT_HCCL", "JT_HCCL_COLLECTIVE_SYNC", "JT_HCCL_LOCAL_RANK", "JT_HCCL_NO_MPI",
    "JT_HCCL_RANK", "JT_HCCL_ROOTINFO_FILE", "JT_HCCL_WORLD_SIZE",
    "JT_LOCK_REPORT_AFTER", "JT_LOCK_TIMEOUT", "JT_MIN_FREE_SPACE_MB",
    "JT_MPI", "JT_MPI_HALF_SIMD",
    "JT_NCCL", "JT_NCCL_LOCAL_RANK", "JT_NCCL_NO_MPI", "JT_NCCL_RANK",
    "JT_NCCL_ROOTINFO_FILE", "JT_NCCL_WATCHDOG_GRACE_S",
    "JT_NCCL_WATCHDOG_INTERVAL_S", "JT_NCCL_WATCHDOG_STALE_S",
    "JT_NCCL_WORLD_SIZE", "JT_NO_SIGNAL_HANDLER", "JT_ONEDNN_BUILD_JOBS",
    "JT_PROBE_CACHE", "JT_RENDEZVOUS_TIMEOUT_S", "JT_SAVE_MEM", "JT_SYNC",
    "JT_TRITON_DEBUG", "JT_TRITON_PTRTRACE",
))

#: Reserved for test harnesses handing parameters to child processes.
TEST_PREFIX = "JT_TEST_"


def known_variables(flag_names):
    """Every ``JT_*`` variable jittor reads, given the native flag names.

    A flag is listed under both prefixes: the core reads startup flags as
    ``JT_BUILD_<NAME>`` and the rest as ``JT_<NAME>``, and which one a flag is
    belongs to ``flag_policy``, not to this check.
    """
    known = set(DIRECT_VARIABLES)
    for name in flag_names:
        known.add(env_config.prefixed_name(name, "build"))
        known.add(env_config.prefixed_name(name, "runtime"))
    for name in env_config.PYTHON_BUILD_VARS:
        known.add(env_config.prefixed_name(name, "build"))
    for name in env_config.PYTHON_RUNTIME_VARS:
        known.add(env_config.prefixed_name(name, "runtime"))
    return known


def unread_variables(known, environ=None):
    """``[(variable, suggestion or None)]`` for ``JT_*`` variables nobody reads."""
    environ = os.environ if environ is None else environ
    found = []
    for variable in sorted(environ):
        if (not variable.startswith(env_config.RUNTIME_PREFIX)
                or variable.startswith(TEST_PREFIX) or variable in known):
            continue
        # The wrong namespace is the common mistake: JT_CACHE_NAME for
        # JT_BUILD_CACHE_NAME, or the other way round.
        rest = variable[len(env_config.RUNTIME_PREFIX):]
        if rest.startswith("BUILD_"):
            swapped = env_config.RUNTIME_PREFIX + rest[len("BUILD_"):]
        else:
            swapped = env_config.BUILD_PREFIX + rest
        if swapped in known:
            found.append((variable, swapped))
            continue
        close = difflib.get_close_matches(variable, sorted(known), n=1, cutoff=0.8)
        found.append((variable, close[0] if close else None))
    return found


def format_unread_warning(unread):
    """The warning text for ``JT_*`` variables nobody reads, or ``""``."""
    if not unread:
        return ""
    parts = ["%s (did you mean %s?)" % pair if pair[1] else pair[0]
             for pair in unread]
    return ("jittor does not read %d environment variable(s) that use its JT_ "
            "prefix, so they have no effect: %s" % (len(unread), ", ".join(parts)))


def canonical_variable(setting):
    """The ``JT_``/``JT_BUILD_`` variable ``setting`` should be given under.

    A native flag's namespace is the one ``flag_policy`` already assigns it:
    startup configuration is build, everything else is runtime. Settings that
    are not native flags come from ``env_config``'s own tables.
    """
    if setting in STARTUP_FLAGS or setting in env_config.PYTHON_BUILD_VARS:
        namespace = "build"
    else:
        namespace = "runtime"
    return env_config.prefixed_name(setting, namespace)


def merge_rows(*row_groups):
    """One row per setting, in name order.

    The core and the Python build driver both report what they read, and they
    overlap: ``log_v`` is a native flag *and* something ``jittor_utils`` reads
    to decide its own verbosity, so a naive concatenation reported it twice and
    made the count wrong.
    """
    merged: Dict[str, Tuple[str, str, object, str]] = {}
    for rows in row_groups:
        for name, variable, value, kind in rows:
            merged.setdefault(name, (name, variable, value, kind))
    return [merged[name] for name in sorted(merged)]


def format_report(rows, build_rows=()):
    """One line naming every setting the environment supplied.

    ``rows`` are the core's ``[flag, variable, value, "legacy"|"prefixed"]``
    rows (``jittor_core.env_flag_sources()``); ``build_rows`` are the same shape
    for the settings only the Python build driver reads.
    """
    entries = merge_rows(rows, build_rows)
    if not entries:
        return ""
    parts = []
    for name, variable, value, kind in entries:
        shown = value if len(value) <= 60 else value[:57] + "..."
        suffix = " [deprecated name]" if kind == "legacy" else ""
        parts.append("%s=%r from %s%s" % (name, shown, variable, suffix))
    return "environment set %d setting(s): %s" % (len(entries), ", ".join(parts))


def format_legacy_warning(rows, build_rows=()):
    """The deprecation text for unprefixed names that were used, or ``""``."""
    legacy = [name for name, _variable, _value, kind
              in merge_rows(rows, build_rows) if kind == "legacy"]
    if not legacy:
        return ""
    moves = ", ".join("%s -> %s" % (name, canonical_variable(name))
                      for name in legacy)
    return ("jittor read %d setting(s) from unprefixed environment variable(s), "
            "which are deprecated because they have no namespace and collide "
            "with unrelated shell variables: %s" % (len(legacy), moves))


def format_ignored_warning(ignored):
    """The warning text for unprefixed names deliberately not read, or ``""``."""
    if not ignored:
        return ""
    moves = ", ".join("%s (use %s)" % pair for pair in ignored)
    return ("jittor ignored %d environment variable(s) whose name is an "
            "ordinary word rather than a namespaced setting, because such a "
            "name in a shell is far more likely to belong to something else: "
            "%s" % (len(ignored), moves))


def python_variable_rows():
    """What the Python build driver read from the environment, as report rows.

    The core reports the flags *it* read; these are the settings only Python
    reads -- the build variables that are not native flags at all
    (``cache_name``, ``debug``, ``use_mkl`` ...) and the native build flags whose
    value ``compiler.py`` computes and then assigns, which the core deliberately
    does not read from the environment (see ``compiler_owned_flag_names`` in
    ``src/utils/log.cc``). Neither group appears in ``env_flag_sources()``, and
    ``nvcc_path`` -- the single most common thing anyone sets -- is in the second.
    """
    return [(name, variable, value, kind) for name, (variable, value, kind)
            in sorted(env_config.resolved().items())]


def report(core, log, environ=None):
    """Emit the startup summary and the deprecation warnings, once."""
    rows = [tuple(row) for row in core.env_flag_sources()]
    build_rows = python_variable_rows()
    summary = format_report(rows, build_rows)
    if summary:
        log.i(summary)
    legacy = format_legacy_warning(rows, build_rows)
    if legacy:
        warnings.warn(legacy, DeprecationWarning, stacklevel=2)
    ignored = format_ignored_warning(env_config.ignored_bare_names(environ))
    if ignored:
        warnings.warn(ignored, DeprecationWarning, stacklevel=2)
    flag_names = [name for name in dir(core.Flags) if not name.startswith("_")]
    unread = format_unread_warning(
        unread_variables(known_variables(flag_names), environ))
    if unread:
        # UserWarning, not DeprecationWarning: the default filters hide the
        # latter unless it is attributed to __main__, and a setting that
        # silently does nothing is exactly what has to be seen.
        warnings.warn(unread, UserWarning, stacklevel=2)
    return summary


__all__ = ["DIRECT_VARIABLES", "TEST_PREFIX", "canonical_variable",
           "merge_rows", "format_report", "format_legacy_warning",
           "format_ignored_warning", "known_variables", "unread_variables",
           "format_unread_warning", "python_variable_rows", "report"]
