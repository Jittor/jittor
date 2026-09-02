"""Regression tests for Jittor home resolution and its user configuration.

Two properties matter here, and both were broken:

* ``JITTOR_HOME`` in the environment is a *per-process* override. Test runs, CI
  jobs and multi-device jobs use it to keep JIT caches apart. Writing it back to
  the shared ``~/.cache/jittor/config.json`` made one isolated run become the
  default for every later process on the machine.
* Several Jittor processes can start at once. Rewriting that configuration in
  place let a reader observe a truncated file, and the resulting
  ``JSONDecodeError`` propagated out of ``import jittor`` as a startup failure.
"""

from __future__ import print_function

import json
import os
from pathlib import Path
import tempfile
import unittest

import jittor_utils

from _helpers.child_process import run_python_child


_RUNNER = r"""
import json, os, sys
sys.path.insert(0, {repo!r})
os.environ["HOME"] = {home!r}
{env_setup}
import jittor_utils
print("HOME_RESULT", jittor_utils.home())
config = os.path.join({home!r}, ".cache", "jittor", "config.json")
stored = {{}}
if os.path.exists(config):
    with open(config) as f:
        raw = f.read()
    try:
        stored = json.loads(raw)
    except ValueError:
        # Report the damaged file as damaged rather than crashing the probe.
        stored = {{"__unparsable__": raw}}
print("CONFIG_RESULT", json.dumps(stored))
"""


def _run(fake_home, env_setup=""):
    repo_python = str(Path(__file__).resolve().parents[2] / "python")
    script = _RUNNER.format(repo=repo_python, home=fake_home, env_setup=env_setup)
    environment = dict(os.environ)
    environment.pop("JITTOR_HOME", None)
    completed = run_python_child(
        ["-c", script], env=environment, inherit=False, merge_stderr=True)
    assert completed.returncode == 0, completed.stdout
    home = config = None
    for line in completed.stdout.splitlines():
        if line.startswith("HOME_RESULT "):
            home = line[len("HOME_RESULT "):]
        elif line.startswith("CONFIG_RESULT "):
            config = json.loads(line[len("CONFIG_RESULT "):])
    assert home is not None and config is not None, completed.stdout
    return home, config


class TestJittorHome(unittest.TestCase):
    def test_environment_override_is_not_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_home = os.path.join(directory, "home")
            isolated = os.path.join(directory, "isolated-cache")
            os.makedirs(fake_home)
            os.makedirs(isolated)

            home, config = _run(
                fake_home,
                env_setup='os.environ["JITTOR_HOME"] = {!r}'.format(isolated),
            )
            self.assertEqual(home, os.path.abspath(isolated))
            self.assertNotIn(
                "JITTOR_HOME",
                config,
                "a per-process JITTOR_HOME must not become the machine default",
            )

            # A later process without the override must not inherit it.
            home, _config = _run(fake_home)
            self.assertEqual(home, os.path.abspath(fake_home))

    def test_damaged_configuration_does_not_prevent_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_home = os.path.join(directory, "home")
            cache = os.path.join(fake_home, ".cache", "jittor")
            os.makedirs(cache)
            # What a reader sees while another process rewrites the file.
            with open(os.path.join(cache, "config.json"), "w") as handle:
                handle.write('{"JITTOR_HO')

            home, _config = _run(fake_home)
            self.assertEqual(home, os.path.abspath(fake_home))

    def test_set_home_persists_an_explicit_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            fake_home = os.path.join(directory, "home")
            chosen = os.path.join(directory, "chosen")
            os.makedirs(fake_home)

            home, config = _run(
                fake_home,
                env_setup=(
                    "import jittor_utils as _u\n"
                    "_u.set_home({!r})\n"
                    "_u._jittor_home = None".format(chosen)
                ),
            )
            self.assertEqual(home, os.path.abspath(chosen))
            self.assertEqual(config.get("JITTOR_HOME"), os.path.abspath(chosen))

    def test_configuration_helpers_are_public(self):
        self.assertTrue(callable(jittor_utils.home))
        self.assertTrue(callable(jittor_utils.set_home))


if __name__ == "__main__":
    unittest.main()
