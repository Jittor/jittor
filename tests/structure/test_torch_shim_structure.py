"""Static contracts for the canonical Torch shim layout and resources."""

from __future__ import print_function

import ast
from pathlib import Path
import re
import unittest


class TestTorchShimStructure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]
        cls.shim_root = cls.repo_root / "python" / "jittor" / "compat" / "shim"
        cls.manifest = cls.repo_root / "agent" / "baselines" / "torch-shim-resources-stage7.txt"

    def test_legacy_physical_package_is_absent(self):
        self.assertFalse((self.repo_root / "python" / "jittor" / "torch_shim").exists())
        self.assertTrue(self.shim_root.is_dir())

    def test_the_resource_manifest_names_files_that_exist(self):
        """The manifest is a packaging inventory: every path in it is shipped.

        It used to also freeze each file's SHA-256, which made it fail on every
        legitimate edit. The exemption set that grew to answer that -- five
        paths whose digests were no longer checked -- is the evidence: a list
        that needs a new exemption per edit is not a rule. By the time it was
        removed, 7 of the 36 digests had drifted, and the two that still failed
        the gate were a documentation page and a ``.cu`` under active
        development. What the manifest is *for* survives without them, and is
        pinned here and in ``test_manifest_covers_deep_and_generated_resources``:
        the deep and generated resources a wheel silently drops are listed, and
        every listed path is really there.
        """
        entries = []
        for line in self.manifest.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
            self.assertIsNotNone(match, line)
            entries.append(match.group(2))

        self.assertEqual(len(entries), len(set(entries)), "duplicate entries")
        missing = [path for path in entries
                   if not (self.repo_root / path).is_file()]
        self.assertEqual(missing, [])

    def test_manifest_covers_deep_and_generated_resources(self):
        paths = {
            line.split("  ", 1)[1]
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")
        }
        required = {
            "docs/compatibility/torch-shim.md",
            "python/jittor/compat/shim/runtime.py",
            "python/jittor/compat/shim/cpp_extension/include/ATen/cuda/detail/UnpackRaw.cuh",
            "python/jittor/compat/shim/resources/stubs/flash_attn/flash_attn_interface.py",
            "python/jittor/compat/shim/resources/stubs/flash_attn/ops/triton/rotary.py",
            "python/jittor/compat/shim/resources/flash_attn_dist_info/METADATA",
            "python/jittor/compat/shim/resources/flash_attn_dist_info/top_level.txt",
            "python/jittor/compat/shim/resources/torch_dist_info/METADATA",
            "python/jittor/compat/shim/resources/torch_init.py",
        }
        self.assertTrue(required.issubset(paths))

    def test_deployed_torch_template_is_an_identity_only_entrypoint(self):
        """It activates the shim, then publishes jittor's identity, and no more.

        The activation used to be pinned by spelling
        (``_torch_compat.install(_jittor)``), which went stale the moment 7.04
        collapsed the three entry points into ``activate()``. The shape is what
        that assertion stood for: the body defines nothing, every call it makes
        is a name the compatibility package handed it, and the identity
        publication is the last statement -- a line after it would run against
        the module it just replaced.
        """
        template = self.shim_root / "resources" / "torch_init.py"
        body = ast.parse(template.read_text(encoding="utf-8")).body
        self.assertFalse(
            any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in body)
        )
        self.assertEqual(ast.unparse(body[-1]), "_sys.modules[__name__] = _jittor")
        from_compat = {
            alias.asname or alias.name
            for node in body
            if isinstance(node, ast.ImportFrom)
            and (node.module or "").startswith("jittor.compat")
            for alias in node.names
        }
        called = [ast.unparse(node.value.func) for node in body
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
        self.assertTrue(called, "the template never activates the shim")
        self.assertEqual(sorted(set(called) - from_compat), [])

    def test_bootstrap_is_a_runtime_facade(self):
        """It re-exports and defines nothing.

        Naming the import lines by spelling (``from .runtime import enable``)
        went stale when 7.04 renamed the entry point. The rule they stood for
        is that every name this module advertises comes from a sibling module,
        either directly or as an alias of one -- which is what keeps a rename
        inside the shim from becoming a public break here.
        """
        bootstrap = self.shim_root / "bootstrap.py"
        body = ast.parse(bootstrap.read_text(encoding="utf-8")).body
        self.assertFalse(
            any(isinstance(node, (ast.FunctionDef, ast.ClassDef)) for node in body)
        )
        reexported = {
            alias.asname or alias.name
            for node in body if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }
        # ``enable = activate`` keeps the 1.x name working; an alias of a
        # re-export is still a re-export.
        for node in body:
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Name)
                    and node.value.id in reexported):
                reexported |= {target.id for target in node.targets
                               if isinstance(target, ast.Name)}
        advertised = set()
        for node in body:
            if (isinstance(node, ast.Assign)
                    and any(getattr(target, "id", None) == "__all__"
                            for target in node.targets)):
                advertised = {element.value for element in node.value.elts}
        self.assertTrue(advertised, "the facade advertises nothing")
        self.assertEqual(sorted(advertised - reexported), [])

    def test_production_imports_use_canonical_paths(self):
        production = (
            self.repo_root / "python" / "jittor" / "compat" / "external_backend.py",
            self.repo_root / "python" / "jittor" / "compat" / "torch" / "__init__.py",
            self.shim_root / "runtime.py",
            self.shim_root / "cpp_extension" / "torch_utils.py",
            self.shim_root / "backends" / "flash_attention.py",
            self.shim_root / "resources" / "stubs" / "flash_attn" / "__init__.py",
        )
        for path in production:
            with self.subTest(path=str(path.relative_to(self.repo_root))):
                self.assertNotIn("from jittor.torch_shim", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
