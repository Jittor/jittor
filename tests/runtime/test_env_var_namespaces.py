# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
"""Configuration comes from a namespace, and says so. [2.22]

Every one of the framework's 80-odd settings used to be readable from an
environment variable of exactly its own name, in lower case, with no namespace:
``get_from_env(#name, default)`` in ``src/utils/log.h`` and
``os.environ.get("debug")`` in ``compiler.py``. So a shell that happened to
export ``debug`` -- a word any CI system, prompt or Makefile might export for
its own reasons -- silently selected a debug build of the whole framework, and
the only trace was one ``LOGi`` line that the default ``log_v=0`` and
``log_silent`` both hide.

Three behaviours are pinned here, and the middle one is the reason the other two
can exist at all:

======================================= ==========================================
``JT_BUILD_DEBUG=1`` / ``JT_LOG_V=1``   the canonical name; takes effect
``debug=1`` (a plain word)              **ignored**, and reported as ignored
``log_v=1``, ``nvcc_path=""`` (old)     still takes effect, warned about once
======================================= ==========================================

Retiring the old names outright is not an option: ``noxfile.py``, the agent
gates and the documentation are full of ``nvcc_path=``, ``use_mpi=`` and
``log_v=``, and breaking all of them at once buys nothing. What was actually
dangerous was the *unnamespaced word*, not the lower case.

**These have to be child processes.** A flag's environment override is read by a
static initializer, i.e. once, before any Python in this process ran; and the
build variables are read while ``jittor``/``jittor_utils`` is being imported.
``os.environ["debug"] = "1"`` inside a test that has already imported jittor
measures nothing at all. Children go through
``tests/_helpers/child_process.py`` so that they import *this* checkout
(``tests/structure/test_child_process_contract.py`` fails the gate on any launch
that names the interpreter directly).

Most of the cases below deliberately import only ``jittor_utils``, not
``jittor``: the cache directory it computes already answers "did this variable
change what gets built" and it answers it without paying for a core compile.

Run::  python -m pytest tests/runtime/test_env_var_namespaces.py
"""

import unittest

from jittor_utils import env_config

from _helpers.child_process import run_child_script


#: ``jittor_utils.cache_path`` is where the products go, so it is the shortest
#: honest answer to "did this setting change the build". A debug build gets its
#: own directory (``jittor_utils.find_cache_path``).
CACHE_PROBE = """
import jittor_utils
print("CACHE", jittor_utils.cache_path)
print("DONE")
"""

FLAG_PROBE = """
import warnings
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    import jittor as jt
print("LOG_V", jt.flags.log_v)
print("IS_DEBUG", jt.compiler.is_debug)
for entry in caught:
    if entry.category is DeprecationWarning:
        print("DEPRECATION", str(entry.message))
print("DONE")
"""


def probe(source, env, name):
    done = run_child_script(source, env=env, text=True, merge_stderr=True,
                            name=name)
    assert done.returncode == 0, done.stdout
    # A child can exit 0 for reasons that have nothing to do with the probe
    # (AGENT-BRIEF 7's SIGCHLD trap), so the marker is what proves it ran.
    assert "DONE" in done.stdout, done.stdout
    return done.stdout


def cache_dir(env, name):
    for line in probe(CACHE_PROBE, env, name).splitlines():
        if line.startswith("CACHE "):
            return line.split(" ", 1)[1]
    raise AssertionError("child printed no cache path")


class TestOrdinaryWordsAreNotSettings(unittest.TestCase):
    """``export debug=1`` and ``export name=x`` do not change what jittor does."""

    def test_bare_debug_does_not_select_a_debug_build(self):
        baseline = cache_dir({}, "debug_baseline")
        bare = cache_dir({"debug": "1"}, "debug_bare")
        self.assertEqual(bare, baseline)

    def test_the_prefixed_name_does_select_a_debug_build(self):
        # The other half of the claim: the setting still exists, it just has a
        # name now. Without this, "ignored" could equally mean "deleted".
        baseline = cache_dir({}, "debug_baseline2")
        prefixed = cache_dir({"JT_BUILD_DEBUG": "1"}, "debug_prefixed")
        self.assertNotEqual(prefixed, baseline)
        self.assertIn("_debug", prefixed)

    def test_bare_debug_is_reported_as_ignored(self):
        out = probe(FLAG_PROBE, {"debug": "1"}, "debug_reported")
        self.assertIn("IS_DEBUG 0", out)
        ignored = [line for line in out.splitlines()
                   if line.startswith("DEPRECATION") and "ignored" in line]
        self.assertEqual(len(ignored), 1, out)
        self.assertIn("JT_BUILD_DEBUG", ignored[0])

    def test_name_was_never_a_flag(self):
        # The audit listed ``name`` among the 76 settable flags. It never was
        # one: ``DEFINE_FLAG(type, name, default, doc)`` in log.h is the macro's
        # own parameter list, and the scan that produced the list read it as a
        # definition. Recorded here so the claim is checkable rather than
        # folklore.
        self.assertNotIn("name", dir(__import__("jittor").flags))
        baseline = cache_dir({}, "name_baseline")
        self.assertEqual(cache_dir({"name": "x"}, "name_bare"), baseline)


class TestDeprecatedNamesKeepWorking(unittest.TestCase):
    """The 70-odd underscored old names still configure the framework."""

    def test_unprefixed_flag_name_still_takes_effect_and_warns_once(self):
        out = probe(FLAG_PROBE, {"log_v": "1"}, "legacy_log_v")
        self.assertIn("LOG_V 1", out)
        warned = [line for line in out.splitlines()
                  if line.startswith("DEPRECATION") and "unprefixed" in line]
        self.assertEqual(len(warned), 1, out)
        self.assertIn("log_v -> JT_LOG_V", warned[0])

    def test_prefixed_flag_name_takes_effect_without_a_warning(self):
        out = probe(FLAG_PROBE, {"JT_LOG_V": "1"}, "prefixed_log_v")
        self.assertIn("LOG_V 1", out)
        # Not "no deprecation at all": the gates themselves run with
        # ``nvcc_path=""``, which the child inherits and which is a deprecated
        # name. The claim is about *this* setting.
        self.assertNotIn("log_v ->", out)
        self.assertIn("log_v='1' from JT_LOG_V", out)

    def test_unprefixed_build_name_still_reaches_the_cache_key(self):
        # ``cache_name`` is the isolation slot every existing script uses.
        baseline = cache_dir({}, "cache_name_baseline")
        legacy = cache_dir({"cache_name": "slot22"}, "cache_name_legacy")
        prefixed = cache_dir({"JT_BUILD_CACHE_NAME": "slot22"},
                             "cache_name_prefixed")
        self.assertNotEqual(legacy, baseline)
        self.assertEqual(legacy, prefixed)


class TestNamespacesDoNotOverlap(unittest.TestCase):
    """Build settings and runtime flags are two namespaces, not one."""

    def test_a_build_flag_is_not_reachable_under_the_runtime_prefix(self):
        self.assertEqual(env_config.prefixed_name("cc_flags", "build"),
                         "JT_BUILD_CC_FLAGS")
        self.assertEqual(env_config.prefixed_name("log_v", "runtime"),
                         "JT_LOG_V")
        baseline = cache_dir({}, "ns_baseline")
        # JT_CC_FLAGS is nobody's variable: naming a build setting under the
        # runtime prefix must not configure it.
        self.assertEqual(cache_dir({"JT_CC_FLAGS": "-O1"}, "ns_wrong_prefix"),
                         baseline)
        self.assertNotEqual(cache_dir({"JT_BUILD_CC_FLAGS": "-O1"},
                                      "ns_right_prefix"), baseline)


class TestResolverRules(unittest.TestCase):
    """The resolution rules, without a subprocess. Same code the children run."""

    def test_prefixed_wins_over_deprecated(self):
        env = {"JT_LOG_V": "2", "log_v": "1"}
        self.assertEqual(env_config.runtime_env("log_v", environ=env), "2")

    def test_empty_is_configured_and_absent_is_not(self):
        # ``nvcc_path=""`` is the documented way to force a CPU-only build, so
        # it must not read as "go and look for nvcc".
        self.assertEqual(
            env_config.build_env("nvcc_path", "fallback", {"nvcc_path": ""}), "")
        self.assertEqual(
            env_config.build_env("nvcc_path", "fallback", {}), "fallback")

    def test_a_bare_word_is_only_read_prefixed(self):
        self.assertIsNone(env_config.build_env("debug", None, {"debug": "1"}))
        self.assertEqual(
            env_config.build_env("debug", None, {"JT_BUILD_DEBUG": "1"}), "1")

    def test_switches_agree_on_what_off_means(self):
        # These comparisons used to be written out per call site and disagreed:
        # ``use_mkl`` tested == "1", so ``use_mkl=true`` meant *off*, while
        # ``conv_opt`` went through int() and raised ValueError on the same word.
        for value in ("0", "", "false", "off", "no", "FALSE"):
            self.assertFalse(env_config.build_flag(
                "use_mkl", True, {"JT_BUILD_USE_MKL": value}), value)
        for value in ("1", "true", "yes", "on", "2"):
            self.assertTrue(env_config.build_flag(
                "use_mkl", False, {"JT_BUILD_USE_MKL": value}), value)
        self.assertTrue(env_config.build_flag("use_mkl", True, {}))


if __name__ == "__main__":
    unittest.main()
