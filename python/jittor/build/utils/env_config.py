# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""The one place that reads jittor's configuration out of the environment.

Two namespaces, because there are two lifetimes
-----------------------------------------------

``JT_BUILD_<NAME>``
    Build configuration: it decides *what gets compiled*, is consumed before or
    during the core build, and is part of the cache fingerprint
    (:func:`jittor_utils.get_build_config`). Changing it means a different
    binary.

``JT_<NAME>``
    Runtime policy: a native flag read by the core's static initializers (see
    ``get_from_env`` in ``src/utils/log.h``). Changing it means the same binary
    behaving differently.

The split is the same one ``jittor/_runtime/flag_policy.py`` already draws for
``jt.config`` versus ``jt.runtime``: its ``STARTUP_FLAGS`` are the build
namespace, everything else is runtime. One partition, three consumers.

The same name used to mean both things at once, with two contradictory
semantics. ``cc_flags`` in the environment was *appended* to the compiler's
flags by ``compiler.py`` and *replaced* the native flag wholesale in
``log.h`` -- which was then overwritten again by ``compiler.py``'s own
assignment, so of the two documented behaviours the one that survived depended
on which reader you happened to be reading. There is one reader of
``JT_BUILD_CC_FLAGS`` now.

Old names keep working
----------------------

Every variable also answers to its historical unprefixed lower-case name, so
existing scripts, gates and documentation keep working: ``nvcc_path=""``,
``use_mpi=0``, ``log_v=1``. Using the old name is recorded and reported once at
startup rather than being invisible, which is the whole point -- the previous
report was one ``LOGi`` line per flag, at a level both the default ``log_v=0``
and ``log_silent`` hide.

Except for names that are ordinary words
----------------------------------------

A configuration name with no ``_`` in it is a plain English word, and a plain
English word exported in a shell is far more likely to be somebody else's
variable than a jittor setting. ``export debug=1`` selected a debug build of the
whole framework; ``export SKEY=...`` and ``export name=...`` are the kind of
thing CI systems and shell prompts set for their own reasons.

So those names are read **only** in prefixed form. The unprefixed form is not
deprecated, it is ignored -- and :func:`ignored_bare_names` reports that it was
seen, so a user who really did mean it is told which name to use instead.

(``name`` itself was never a flag: the audit that named it read
``DEFINE_FLAG(type, name, default, doc)`` in ``log.h`` and took the macro's own
parameter for a flag. ``nthread`` is real but lives inside ``#ifdef TEST_LOG``.
``debug`` is the one that was genuinely reachable.)
"""

import os
from typing import Dict, Tuple

#: Build configuration: read before or during the core build.
BUILD_PREFIX = "JT_BUILD_"

#: Runtime policy: read by the compiled core's static initializers.
RUNTIME_PREFIX = "JT_"

#: Build variables that only Python reads -- they are not native flags, so
#: ``flag_policy.py`` does not know about them. Name -> what it decides.
#:
#: ``compile_extern``'s ``use_*`` switches are build configuration rather than
#: runtime policy because they decide whether an extern is compiled and linked
#: at all, not what a compiled core does with it.
PYTHON_BUILD_VARS = {
    "cache_name": "sub-directory of the cache the products go in",
    "conv_opt": "compile the experimental cuDNN convolution path",
    "cuda_debug": "add -G to nvcc, i.e. device-side debug info",
    "cutt_include_path": "where to find cuTT headers instead of downloading",
    "cutt_lib_path": "where to find libcutt instead of downloading",
    "debug": "build the core with -g -O0 and assertions",
    "enable_lto": "link the core with -flto",
    "kernel_flags": "extra flags for JIT-compiled kernels only",
    "mkl_include_path": "where to find oneDNN headers instead of downloading",
    "mkl_lib_path": "where to find libdnnl instead of downloading",
    "nccl_include_path": "where to find NCCL headers instead of downloading",
    "nccl_lib_path": "where to find libnccl instead of downloading",
    "python_config_path": "python3-config to take include paths from",
    "tikcc_path": "Ascend tikcc compiler to build the NPU backend with",
    "use_auto_diff": "record every op for the auto-diff comparison tool",
    "use_cutt": "build and link cuTT",
    "use_mkl": "build and link oneDNN",
    "use_mpi": "build and link MPI",
    "use_nccl": "build and link NCCL",
}

#: Runtime variables only Python reads. Native flags are *not* listed here;
#: they come from ``flag_policy.py`` and the core reads them itself.
PYTHON_RUNTIME_VARS = {
    "mp_log_v": "verbosity of dataset worker processes",
}


def _is_bare_word(name):
    """Whether ``name`` is too generic to be read unprefixed. See the module doc."""
    return "_" not in name


def prefixed_name(name, namespace):
    """The canonical variable a setting is read from."""
    if namespace not in ("build", "runtime"):
        raise ValueError("namespace must be 'build' or 'runtime', not "
                         + repr(namespace))
    prefix = BUILD_PREFIX if namespace == "build" else RUNTIME_PREFIX
    return prefix + name.upper()


#: Every setting the process environment actually supplied, as
#: ``setting -> (variable, value, "prefixed"|"legacy")``. The Python mirror of
#: the core's ``env_flag_sources()``: the bootstrap turns both into one summary
#: line and one DeprecationWarning. See ``jittor/_runtime/env_report.py``.
#:
#: Only lookups against the real process environment are recorded -- a caller
#: that passes its own ``environ`` is asking a question, not being configured.
_resolved: Dict[str, Tuple[str, str, str]] = {}


def resolved():
    """``{setting: (variable, value, kind)}`` for everything the environment set."""
    return dict(_resolved)


def legacy_uses():
    """``{setting: variable}`` for every unprefixed name that was read."""
    return {name: variable for name, (variable, _value, kind)
            in _resolved.items() if kind == "legacy"}


def lookup(name, namespace, default=None, environ=None):
    """The configured value of ``name``, or ``default``.

    Tries the canonical prefixed variable, then the historical unprefixed name
    unless that name is a bare word (see the module docstring). Returns
    ``default`` -- which may be ``None`` to mean "not configured" -- rather than
    raising, because "unset" and "set to the empty string" are different
    configurations for several of these (``nvcc_path=""`` is the documented way
    to force a CPU-only build).
    """
    environ = os.environ if environ is None else environ
    canonical = prefixed_name(name, namespace)
    value = environ.get(canonical)
    if value is not None:
        if environ is os.environ:
            _resolved[name] = (canonical, value, "prefixed")
        return value
    if _is_bare_word(name):
        return default
    value = environ.get(name)
    if value is None:
        return default
    if environ is os.environ:
        _resolved[name] = (name, value, "legacy")
    return value


def build_env(name, default=None, environ=None):
    """The configured value of a build variable. See :func:`lookup`."""
    return lookup(name, "build", default, environ)


def runtime_env(name, default=None, environ=None):
    """The configured value of a runtime variable. See :func:`lookup`."""
    return lookup(name, "runtime", default, environ)


def build_flag(name, default=False, environ=None):
    """A build variable read as an on/off switch.

    ``"0"``, ``""``, ``"false"``, ``"off"`` and ``"no"`` are off; anything else
    that is set is on. The call sites this replaces each spelled their own
    version of this comparison, and they disagreed: ``use_mkl`` and ``use_cutt``
    tested ``== "1"`` (so ``use_mkl=true`` silently meant *off*) while
    ``conv_opt`` went through ``int()`` (so ``conv_opt=true`` raised
    ``ValueError`` out of import).
    """
    value = build_env(name, None, environ)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "off", "no")


def ignored_bare_names(environ=None):
    """Settings whose unprefixed name is set in the environment and ignored.

    A user who exported ``debug=1`` meaning the jittor build switch gets told
    which name to use; a user who exported it for some other tool is told that
    jittor did not act on it. Both need to hear it exactly once, at startup.
    """
    environ = os.environ if environ is None else environ
    found = []
    for name, namespace in (list((n, "build") for n in PYTHON_BUILD_VARS)
                            + list((n, "runtime") for n in PYTHON_RUNTIME_VARS)):
        if _is_bare_word(name) and name in environ:
            found.append((name, prefixed_name(name, namespace)))
    return sorted(found)


def child_env(environ=None, **settings):
    """A copy of ``environ`` with ``settings`` applied under canonical names.

    Callers that hand configuration to a child process used to write the
    unprefixed name, which is the form this module is retiring; a child started
    that way reports a deprecated name that no user chose. Pass ``namespace``
    per setting as ``name=(value, namespace)``, or a plain value for the runtime
    namespace.
    """
    environ = os.environ if environ is None else environ
    result = dict(environ)
    for name, value in settings.items():
        namespace = "runtime"
        if isinstance(value, tuple):
            value, namespace = value
        result[prefixed_name(name, namespace)] = str(value)
    return result
