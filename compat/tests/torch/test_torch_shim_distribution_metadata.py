"""The in-process shim answers ``importlib.metadata`` for ``torch``.

Libraries such as Transformers and Diffusers do not import ``torch`` to decide
whether PyTorch is installed; they ask :mod:`importlib.metadata`. The deployed
shim carries a real ``torch-<version>.dist-info``, but the in-process shim
(``JITTOR_TORCH_SHIM=1`` with the packaged ``torch`` entry on ``PYTHONPATH``)
used to publish the module only. Activation now registers a process-local
finder over the packaged ``torch_dist_info`` resource, and these tests pin the
contract from both sides:

* inside a shim process, ``version``/``metadata`` answer the same API version
  the namespace reports, and the answer really comes from the packaged
  resource (not a stray reference PyTorch on ``sys.path``);
* releasing the finder removes that answer, and republishing restores it;
* plain ``import jittor`` publishes nothing.

The child processes drop any reference-PyTorch ``*.dist-info`` directory from
``sys.path`` first. This checkout's test environment keeps a real PyTorch for
oracle runs, and asking "is the distribution absent" while that one is visible
would answer the wrong question.
"""

from __future__ import absolute_import

import importlib.metadata
import os
from pathlib import Path
import unittest

import torch

from _helpers.child_process import TORCH_MODE_VARIABLES, run_python_child


_RESOURCES = Path(__file__).resolve().parents[3] / "compat" / "shim" / "resources"

_DROP_REFERENCE_TORCH = """
import os as _os
import sys as _sys

_kept = []
for _entry in _sys.path:
    try:
        _names = _os.listdir(_entry)
    except OSError:
        _kept.append(_entry)
        continue
    if any(
        _name.startswith("torch") and _name.endswith((".dist-info", ".egg-info"))
        for _name in _names
    ):
        continue
    _kept.append(_entry)
_sys.path[:] = _kept
"""


def _child_env(shim):
    """A complete environment: no inherited Torch mode and no inherited path."""

    env = dict(os.environ)
    for name in TORCH_MODE_VARIABLES:
        env.pop(name, None)
    env.pop("PYTHONPATH", None)
    env["JITTOR_TORCH_SKIP_EXT_BUILD"] = "1"
    env["JITTOR_TORCH_QUIET"] = "1"
    if shim:
        env["JITTOR_TORCH_SHIM"] = "1"
        env["PYTHONPATH"] = os.fspath(_RESOURCES)
    return env


def _run_child(source, shim):
    result = run_python_child(
        ["-c", _DROP_REFERENCE_TORCH + source],
        env=_child_env(shim), inherit=False, merge_stderr=True)
    if result.returncode != 0:
        raise AssertionError(
            "child exited %d:\n%s" % (result.returncode, result.stdout))
    return result.stdout


class TestInProcessShimDistributionMetadata(unittest.TestCase):
    def test_version_and_metadata_answer_the_torch_api_version(self):
        from jittor.compat.shim import torch_metadata

        self.assertTrue(torch_metadata.is_published())
        self.assertEqual(
            importlib.metadata.version("torch"), torch.__torch_version__)
        self.assertEqual(
            importlib.metadata.version("torch"), torch.version.__version__)
        self.assertEqual(
            importlib.metadata.version("torch"), torch_metadata.version())
        self.assertEqual(
            importlib.metadata.metadata("torch")["Name"], "torch")

    def test_the_answer_comes_from_the_packaged_resource(self):
        from jittor.compat.shim import torch_metadata

        distribution = importlib.metadata.distribution("torch")
        self.assertEqual(
            distribution.read_text("METADATA"),
            (torch_metadata.resource_root() / "METADATA").read_text(
                encoding="utf-8"))
        self.assertEqual(distribution.read_text("top_level.txt").strip(), "torch")

    def test_the_reported_version_is_what_torch_version_reports_on_opt_in(self):
        saved = torch.__version__
        try:
            torch.compat_report_torch_api_version(True)
            self.assertEqual(
                importlib.metadata.version("torch"), torch.__version__)
        finally:
            torch.__version__ = saved

    def test_packages_distributions_maps_the_torch_import_to_the_shim(self):
        mapping = importlib.metadata.packages_distributions()
        self.assertIn("torch", mapping.get("torch", ()))

    def test_release_and_publish_round_trip(self):
        from jittor.compat.shim import torch_metadata

        finder = torch_metadata.publish()
        try:
            self.assertTrue(torch_metadata.release(finder))
            self.assertFalse(torch_metadata.is_published())
        finally:
            torch_metadata.publish()
        self.assertTrue(torch_metadata.is_published())


class TestActivationBoundary(unittest.TestCase):
    def test_plain_import_jittor_publishes_no_torch_distribution(self):
        stdout = _run_child(
            """
import importlib.metadata as metadata
import jittor
from jittor.compat.shim import torch_metadata

assert not torch_metadata.is_published(), (
    "plain import jittor published a torch distribution finder")
try:
    metadata.version("torch")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("plain import jittor published a torch distribution")
print("RESULT=boundary")
""",
            shim=False,
        )
        self.assertIn("RESULT=boundary", stdout)

    def test_shim_mode_composition_publishes_metadata(self):
        """``JITTOR_TORCH_SHIM=1`` publishes the namespace on ``import jittor``.

        That composition entry installs a ``torch`` module without the
        deployed runtime site, so metadata has to come from the finder. A
        library imported between ``import jittor`` and an explicit
        ``import torch`` asks here.
        """
        stdout = _run_child(
            """
import importlib.metadata as metadata
import sys
import jittor
from jittor.compat.shim import torch_metadata

assert "torch" in sys.modules
assert torch_metadata.is_published()
assert metadata.version("torch") == sys.modules["torch"].__torch_version__
assert metadata.metadata("torch")["Name"] == "torch"
print("RESULT=composition")
""",
            shim=True,
        )
        self.assertIn("RESULT=composition", stdout)

    def test_release_makes_the_distribution_unavailable(self):
        stdout = _run_child(
            """
import importlib.metadata as metadata
import sys
import jittor
from jittor.compat.shim import torch_metadata

# Shim-mode composition publishes the namespace without deploying the runtime
# site, so the finder is the only ``torch`` distribution in this process.
assert "torch" in sys.modules
assert torch_metadata.is_published()
assert metadata.version("torch") == sys.modules["torch"].__torch_version__
assert "torch" in metadata.packages_distributions().get("torch", ())

assert torch_metadata.release()
assert not torch_metadata.is_published()
try:
    metadata.version("torch")
except metadata.PackageNotFoundError:
    pass
else:
    raise AssertionError("metadata still resolves torch after release")
print("RESULT=released")
""",
            shim=True,
        )
        self.assertIn("RESULT=released", stdout)


if __name__ == "__main__":
    unittest.main()
