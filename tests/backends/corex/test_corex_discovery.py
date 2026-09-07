"""Corex probing must not change anything, and the guard must be able to notice.

The defect this pins is a probe that *ran* the toolchain to decide whether the
toolchain was there. The old assertion was a top-level ``os.listdir`` of one
temporary directory, which is blind to every side effect that matters: spawning
a process, writing into a subdirectory, touching ``os.environ``, or writing
anywhere outside that one directory.

So the guard here records four classes of side effect, and a second test
injects each class into a *copy* of the module and asserts the guard reports
it. Without that second test the first one passes whether or not the guard
still works -- which is the failure mode that hid a broken cuTT build and a
never-called ``setup_cutt()`` behind a green gate.
"""

import builtins
import contextlib
import importlib.util
import os
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

SOURCE = Path(__file__).parents[3] / "backends/corex/__init__.py"


def load_corex_module(source=SOURCE, name="corex_compiler_probe"):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tree(root):
    return {str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in Path(root).rglob("*") if p.is_file()}


@contextlib.contextmanager
def side_effect_recorder(watched):
    """Record process spawns, writes, env mutations and tree changes."""
    found = []
    env_before = dict(os.environ)
    tree_before = _tree(watched)
    cwd_before = os.getcwd()

    real_open = builtins.open
    spawners = {
        "subprocess.run": (subprocess, "run"),
        "subprocess.Popen": (subprocess, "Popen"),
        "subprocess.check_output": (subprocess, "check_output"),
        "os.system": (os, "system"),
        "os.popen": (os, "popen"),
    }
    saved = {label: getattr(mod, attr) for label, (mod, attr) in spawners.items()}

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            found.append("write-open:%s (mode %r)" % (file, mode))
        return real_open(file, mode, *args, **kwargs)

    def make_spawn_guard(label):
        def guard(*args, **kwargs):
            found.append("spawn:%s %r" % (label, args[0] if args else None))
            raise AssertionError("probing must not run anything (%s)" % label)
        return guard

    builtins.open = guarded_open
    for label, (mod, attr) in spawners.items():
        setattr(mod, attr, make_spawn_guard(label))
    try:
        yield found
    finally:
        builtins.open = real_open
        for label, (mod, attr) in spawners.items():
            setattr(mod, attr, saved[label])
        if dict(os.environ) != env_before:
            changed = set(os.environ.items()) ^ set(env_before.items())
            found.append("env:%s" % sorted(k for k, _ in changed))
            os.environ.clear()
            os.environ.update(env_before)
        if _tree(watched) != tree_before:
            found.append("tree:%s" % sorted(
                set(_tree(watched)) ^ set(tree_before)))
        if os.getcwd() != cwd_before:
            found.append("cwd:%s" % os.getcwd())
            os.chdir(cwd_before)


@contextlib.contextmanager
def fake_corex_home():
    with tempfile.TemporaryDirectory() as root:
        home = os.path.join(root, "corex")
        os.makedirs(os.path.join(home, "bin"))
        with open(os.path.join(home, "bin", "clang++"), "w") as stream:
            stream.write("fake clang++\n")
        yield root, home


class TestCorexDiscovery(unittest.TestCase):
    def test_discovery_is_read_only_and_path_configurable(self):
        corex_compiler = load_corex_module()
        with fake_corex_home() as (root, home):
            with side_effect_recorder(root) as found:
                result = corex_compiler.discover(home)
            self.assertEqual(found, [], "probing had side effects")
            self.assertTrue(result.available)
            self.assertEqual(result.home, os.path.abspath(home))
            self.assertEqual(result.compiler_path,
                             os.path.join(home, "bin", "clang++"))

    def test_corex_home_env_is_honoured_without_an_argument(self):
        corex_compiler = load_corex_module()
        with fake_corex_home() as (root, home):
            env = dict(os.environ)
            env["COREX_HOME"] = home
            with unittest.mock.patch.dict(os.environ, env, clear=True):
                with side_effect_recorder(root) as found:
                    result = corex_compiler.discover()
            self.assertEqual(found, [])
            self.assertTrue(result.available)
            self.assertEqual(result.home, os.path.abspath(home))

    def test_missing_compiler_is_reported_without_global_setup(self):
        corex_compiler = load_corex_module()
        with tempfile.TemporaryDirectory() as root:
            with side_effect_recorder(root) as found:
                result = corex_compiler.discover(root)
            self.assertEqual(found, [])
            self.assertFalse(result.available)
            self.assertIn("compiler", result.reason)
            self.assertFalse(hasattr(corex_compiler, "has_corex"))

    def test_configure_refuses_instead_of_probing_a_missing_install(self):
        corex_compiler = load_corex_module()
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(RuntimeError) as caught:
                corex_compiler.configure(object(), root)
            self.assertIn("compiler", str(caught.exception))


class TestTheGuardCanNoticeSideEffects(unittest.TestCase):
    """Inject each class of side effect into a copy; the guard must report it.

    This is what gives the tests above their teeth. A guard that silently
    stopped watching -- a renamed spawner, an ``open`` that no longer goes
    through builtins, a tree snapshot rooted in the wrong place -- would leave
    them green.
    """

    INJECTIONS = {
        "spawn": "subprocess.run(['true'], check=False)",
        "write": ("open(os.path.join(os.path.abspath(os.path.expanduser("
                  "corex_home or '.')), 'probe.stamp'), 'w').close()"),
        "env": "os.environ['COREX_PROBE_RAN'] = '1'",
    }

    def _module_with(self, injection, workdir):
        source = SOURCE.read_text()
        anchor = "    home = corex_home or os.environ.get"
        assert anchor in source, "discover() no longer starts where expected"
        patched = source.replace(
            anchor, "    import subprocess\n    %s\n%s" % (injection, anchor), 1)
        path = Path(workdir) / "corex_compiler_injected.py"
        path.write_text(patched)
        return load_corex_module(path, name="corex_compiler_injected")

    def test_each_injected_side_effect_is_reported(self):
        for label, injection in self.INJECTIONS.items():
            with self.subTest(side_effect=label):
                with fake_corex_home() as (root, home):
                    module = self._module_with(injection, root)
                    try:
                        with side_effect_recorder(root) as found:
                            module.discover(home)
                    except AssertionError as exc:
                        # The spawn guard raises on purpose; that is a report.
                        self.assertIn("must not run anything", str(exc))
                        continue
                    self.assertTrue(
                        found, "%s side effect went unnoticed" % label)


if __name__ == "__main__":
    unittest.main()
