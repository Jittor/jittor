"""Every recorded Torch-compatible entry point must still resolve and stay callable.

The native surface has had this gate since ``public_api_manifest.json``: a name
nothing imports is a name no gate protects, and the regressions that motivated
it were all deletions that looked like dead-code removal. ``6f9d7e7bc`` is the
one that matters here -- it dropped two names another module re-exported and
*Torch mode stopped importing at all*, which says plainly that the Torch
namespace is reachable by indirection the native manifest never walks.

That namespace is a different object graph. ``torch`` is a ``TorchNamespace``
view over the compatibility owner, ``torch.Tensor`` is not ``jt.Var``, and the
frontend publishes ``torch.nn``, ``torch.linalg`` and the rest by composition.
So the native manifest cannot cover it and this is a second manifest, not a
second half of the first one.

Scope is the ten namespaces the frontend publishes as first-class API. Public
**callable** names, the same rule the native manifest uses, which is also what
the coverage wrapper in ``tests/_helpers/api_coverage.py`` can measure. Public
non-callable attributes -- ``torch.float32`` and the other dtype objects, the
re-exported submodules -- are deliberately outside both manifests; catching
their removal would need a different rule and is not claimed here.

This gate resolves names; it does not run kernels. Numerical behaviour stays
the OpInfo suite's job, so this stays cheap enough to run everywhere.

The manifest is generated but **checked in**, on purpose. A gate that rebuilds
its own expectation at run time cannot fail: it would re-record whatever the
tree currently exports, including the deletion it was supposed to catch.
Regenerate deliberately with ``--regenerate`` and read the diff.

This module belongs to the Torch process mode: importing ``torch`` installs the
compatibility frontend process-wide, so a native session must not collect it.
``tests/structure`` is already listed in ``process_modes.TORCH_MODE_PATHS``.
"""

import json
import unittest
from pathlib import Path

import torch

MANIFEST = Path(__file__).resolve().parent / "torch_api_manifest.json"

#: The namespaces this manifest claims, in the order a reader wants them.
#:
#: Declared rather than discovered. ``dir(torch)`` also reaches ``onnx``,
#: ``fx``, ``multiprocessing`` and a dozen other shims whose contents are
#: placeholders; recording them would grow the denominator with names no
#: maintained run can exercise, and an inflated denominator is the same lie as
#: a shrunken one.
SURFACE_GROUPS = (
    "torch",
    "torch.Tensor",
    "torch.nn",
    "torch.nn.functional",
    "torch.linalg",
    "torch.fft",
    "torch.cuda",
    "torch.distributions",
    "torch.optim",
    "torch.autograd",
)


def _holder(dotted):
    """Resolve a recorded namespace path to the live object.

    ``torch.Tensor`` is a class rather than a module, and is walked the same
    way: what the manifest records is the owner of the names, not its kind.
    """
    obj = torch
    for part in dotted.split(".")[1:]:
        obj = getattr(obj, part)
    return obj


def live_surface():
    """The public callable Torch surface as it exists right now."""
    surface = {}
    for group in SURFACE_GROUPS:
        holder = _holder(group)
        surface[group] = sorted(
            name for name in dir(holder)
            if not name.startswith("_") and callable(getattr(holder, name, None))
        )
    return surface


class TestTorchApiSurface(unittest.TestCase):
    def setUp(self):
        self.recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_every_recorded_name_still_resolves(self):
        missing = []
        for group, names in sorted(self.recorded.items()):
            try:
                holder = _holder(group)
            except AttributeError:
                missing.append(group + " (whole namespace)")
                continue
            for name in names:
                obj = getattr(holder, name, None)
                if obj is None or not callable(obj):
                    missing.append(group + "." + name)
        self.assertEqual(
            missing, [],
            "Torch entry points disappeared or stopped being callable: %s\n"
            "If the removal is deliberate, regenerate the manifest in the same "
            "commit so the deletion is reviewable." % missing[:20])

    def test_the_manifest_covers_every_declared_namespace(self):
        # A group quietly dropped from the manifest would take its names out of
        # the denominator, and a smaller denominator reads as better coverage.
        self.assertEqual(sorted(self.recorded), sorted(SURFACE_GROUPS))

    def test_the_rule_catches_a_removed_name(self):
        """A manifest naming something absent must fail, or this gate is decorative."""
        fabricated = {"torch": ["definitely_not_a_public_torch_name"]}
        missing = [
            "torch." + name for name in fabricated["torch"]
            if not callable(getattr(torch, name, None))
        ]
        self.assertEqual(len(missing), 1,
                         "the resolution rule no longer detects an absent name")

    def test_the_rule_catches_a_removed_tensor_method(self):
        """The same, through the class the native manifest does not own."""
        self.assertFalse(
            callable(getattr(torch.Tensor, "definitely_not_a_tensor_method", None)),
            "the resolution rule no longer detects an absent Tensor method")


if __name__ == "__main__":
    import sys
    if "--regenerate" in sys.argv:
        surface = live_surface()
        MANIFEST.write_text(json.dumps(surface, indent=1, sort_keys=True) + "\n",
                            encoding="utf-8")
        print("recorded %d names across %d namespaces"
              % (sum(len(v) for v in surface.values()), len(surface)))
    else:
        unittest.main()
