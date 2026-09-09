"""Every recorded public native entry point must still resolve and stay callable.

Today's regressions all had one shape: a cleanup removed something that was only
reachable through indirection, and no gate noticed because nothing referenced
the name. `96800dad9` dropped an ``f`` prefix from a codegen template and the
core stopped building. `6f9d7e7bc` deleted two names that another module
re-exported and Torch mode stopped importing at all. Both were "polish" commits
whose diff looked like dead-code removal.

The OpInfo database exercises 179 operators with real numerics. The public
callable surface is roughly five times that, and the difference is exactly where
a deletion hides: a name nothing imports is a name no gate protects.

This gate resolves names; it does not run kernels. It answers "did the entry
point survive", which is the half no other gate owns -- numerical behaviour
stays the OpInfo suite's job, so this stays cheap enough to run everywhere.

The manifest is generated but **checked in**, on purpose. A gate that rebuilds
its own expectation at run time cannot fail: it would re-record whatever the
tree currently exports, including the deletion it was supposed to catch.
Regenerate deliberately with ``--regenerate`` and read the diff.
"""

import json
import types
import unittest
from pathlib import Path

import jittor as jt

MANIFEST = Path(__file__).resolve().parent / "public_api_manifest.json"


def _holder(dotted):
    """Resolve a recorded module path to the live module object."""
    obj = jt
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def live_surface():
    """The public callable surface as it exists right now."""
    mods = {"jt": jt}
    for name in ("nn", "ops", "linalg", "fft", "init", "distributions", "optim",
                 "sparse", "autograd", "misc", "pool", "contrib", "attention"):
        mod = getattr(jt, name, None)
        if isinstance(mod, types.ModuleType):
            mods["jt." + name] = mod
    surface = {}
    for mod_name, mod in mods.items():
        surface[mod_name] = sorted(
            n for n in dir(mod)
            if not n.startswith("_") and callable(getattr(mod, n, None))
        )
    return surface


class TestPublicApiSurface(unittest.TestCase):
    def setUp(self):
        self.recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_every_recorded_name_still_resolves(self):
        missing = []
        for mod_name, names in sorted(self.recorded.items()):
            if mod_name == "Var":
                continue                      # covered by the Var-method test
            try:
                mod = _holder(mod_name)
            except AttributeError:
                missing.append(mod_name + " (whole module)")
                continue
            for name in names:
                obj = getattr(mod, name, None)
                if obj is None or not callable(obj):
                    missing.append(mod_name + "." + name)
        self.assertEqual(
            missing, [],
            "public entry points disappeared or stopped being callable: %s\n"
            "If the removal is deliberate, regenerate the manifest in the same "
            "commit so the deletion is reviewable." % missing[:20])

    def test_var_methods_still_resolve(self):
        recorded = self.recorded.get("Var", [])
        var = jt.ones((2, 2))
        missing = [n for n in recorded if not callable(getattr(type(var), n, None))]
        self.assertEqual(missing, [], "Var methods disappeared: %s" % missing[:20])

    def test_the_rule_catches_a_removed_name(self):
        """A manifest naming something absent must fail, or this gate is decorative."""
        fabricated = {"jt": ["definitely_not_a_public_jittor_name"]}
        missing = [
            "jt." + n for n in fabricated["jt"]
            if not callable(getattr(jt, n, None))
        ]
        self.assertEqual(len(missing), 1,
                         "the resolution rule no longer detects an absent name")


if __name__ == "__main__":
    import sys
    if "--regenerate" in sys.argv:
        surface = live_surface()
        var = jt.ones((2, 2))
        surface["Var"] = sorted(
            n for n in dir(var)
            if not n.startswith("_") and callable(getattr(type(var), n, None)))
        MANIFEST.write_text(json.dumps(surface, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print("recorded %d names" % sum(len(v) for v in surface.values()))
    else:
        unittest.main()
