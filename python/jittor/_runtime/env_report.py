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

Three things get reported here, once each, after every static initializer and
every build variable has been read:

* one summary line naming every setting the environment supplied, and which
  variable each came from;
* one ``DeprecationWarning`` listing the unprefixed names that were used, with
  the ``JT_``/``JT_BUILD_`` name to move to;
* one ``DeprecationWarning`` for unprefixed names that were **ignored** because
  they are ordinary words (``debug``), so that a user who did mean the jittor
  switch is told which name to use rather than silently getting the default.

Kept out of ``jittor/__init__.py`` so it can be tested without importing the
core: :func:`format_report` and :func:`format_legacy_warning` are pure
functions of the rows the core hands over.
"""

import warnings

from jittor_utils import env_config

from .flag_policy import STARTUP_FLAGS


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
    merged = {}
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
    return summary


__all__ = ["canonical_variable", "merge_rows", "format_report",
           "format_legacy_warning", "format_ignored_warning",
           "python_variable_rows", "report"]
