"""The compatibility layer's dependencies run one way.

The intended order is ``core -> tensor -> nn/optim -> distributed -> fsdp``.
What was there was a two-way edge: ``installers/tensor.py``, ``optimizers.py``,
``installers/nn.py`` and ``installers/distributed.py`` each reached *up* with
an inline ``from jittor.compat import fsdp2`` in the middle of a hot path,
while ``fsdp2/installer.py`` reached back *down* into ``compat.torch.context``.
Packages that import each other cannot be read, tested or replaced apart.

The inversion is ``jittor/compat/fsdp_hooks.py``: fsdp2 registers itself there
when imported, and the layers below ask for the provider. These are rules over
the tree, not an inventory -- a new upward import fails here.
"""

import ast
import unittest
from pathlib import Path

import jittor


_COMPAT = Path(jittor.__file__).resolve().parent / "compat"

#: Installing ``torch.distributed`` is precisely when the FSDP2 surface has to
#: be hung off it, and the objects that needs exist only there. It is
#: composition, not use: once, at install time, off any hot path.
_COMPOSITION_EDGE = ("torch/installers/distributed.py", "_install_fsdp2_distributed")


def _sources(package):
    root = _COMPAT / package
    return sorted(path for path in root.rglob("*.py")
                  if "__pycache__" not in path.parts)


def _imports(path):
    """Yield (node, dotted module name) for every import in ``path``.

    Relative imports are resolved against the file's own package so that
    ``from ...fsdp2 import x`` is caught as readily as the absolute spelling.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parts = path.relative_to(_COMPAT).parts
    package = ("jittor", "compat") + parts[:-1]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - node.level + 1]
                yield node, ".".join(base + ((node.module,) if node.module else ()))
            else:
                yield node, node.module or ""


def _enclosing_function(tree, lineno):
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno <= lineno <= (node.end_lineno or node.lineno):
                if best is None or node.lineno > best.lineno:
                    best = node
    return best.name if best else None


class TestNothingBelowFsdp2ImportsIt(unittest.TestCase):
    def test_only_the_composition_edge_names_fsdp2(self):
        offenders = []
        for path in _sources("torch"):
            relative = path.relative_to(_COMPAT).as_posix()
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node, module in _imports(path):
                if not module.startswith("jittor.compat.fsdp2"):
                    continue
                where = (relative, _enclosing_function(tree, node.lineno))
                if where == _COMPOSITION_EDGE:
                    continue
                offenders.append("%s:%d imports %s (in %s)"
                                 % (relative, node.lineno, module, where[1]))
        self.assertEqual(
            offenders, [],
            "jittor.compat.torch sits below fsdp2; ask "
            "jittor.compat.fsdp_hooks.provider() instead of importing it. "
            "Only %s::%s may, and only because installing torch.distributed "
            "is where the FSDP2 surface gets hung off it." % _COMPOSITION_EDGE)

    def test_the_composition_edge_still_exists(self):
        # A rule whose only allowance has silently disappeared is a rule that
        # stopped being tested. If the edge moves, this test should be updated
        # deliberately rather than pass by vacuum.
        path = _COMPAT / _COMPOSITION_EDGE[0]
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found = [module for node, module in _imports(path)
                 if module.startswith("jittor.compat.fsdp2")
                 and _enclosing_function(tree, node.lineno) == _COMPOSITION_EDGE[1]]
        self.assertEqual(len(found), 1, found)


class TestTheSeamIsUsable(unittest.TestCase):
    def test_fsdp_hooks_is_a_leaf(self):
        # It is the one module both sides depend on, so it may not depend on
        # either -- nor on jittor itself, so that importing it can never be the
        # thing that pulls the core in.
        tree = ast.parse((_COMPAT / "fsdp_hooks.py").read_text(encoding="utf-8"))
        imported = [module for _node, module in _imports(_COMPAT / "fsdp_hooks.py")]
        self.assertEqual(
            [m for m in imported if m != "__future__"], [],
            "fsdp_hooks is the seam between two packages; it must import nothing")
        self.assertTrue(any(isinstance(n, ast.FunctionDef) and n.name == "register"
                            for n in tree.body))

    def test_importing_fsdp2_registers_a_provider_that_satisfies_the_contract(self):
        from jittor.compat import fsdp_hooks
        import jittor.compat.fsdp2 as fsdp2

        provider = fsdp_hooks.provider()
        self.assertIsNotNone(
            provider, "importing jittor.compat.fsdp2 must register it")
        self.assertIs(provider, fsdp2)
        for name in fsdp_hooks.REQUIRED:
            with self.subTest(name=name):
                self.assertTrue(callable(getattr(provider, name, None)))

    def test_register_refuses_an_implementation_missing_a_name(self):
        # The point of validating: an fsdp2 that stops exporting one of these
        # must fail at import, not as an AttributeError inside a training step.
        from jittor.compat import fsdp_hooks

        previous = fsdp_hooks.provider()
        try:
            with self.assertRaises(TypeError) as caught:
                fsdp_hooks.register(object())
            self.assertIn("_execute_with_true_fsdp", str(caught.exception))
        finally:
            fsdp_hooks.register(previous)
        self.assertIs(fsdp_hooks.provider(), previous)

    def test_the_callers_only_use_names_the_contract_promises(self):
        # REQUIRED is the contract. If a caller below fsdp2 starts using a
        # method that is not on it, the contract has quietly stopped
        # describing the coupling it exists to describe.
        from jittor.compat import fsdp_hooks

        used = set()
        for path in _sources("torch"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id in ("_fsdp", "_fsdp2_zero",
                                              "_fsdp2_step", "_fsdp2_backward")):
                    used.add(node.attr)
        self.assertTrue(used, "found no provider call sites to check")
        self.assertEqual(
            sorted(used - set(fsdp_hooks.REQUIRED)), [],
            "add it to fsdp_hooks.REQUIRED so registration validates it")


class TestCollectivesMovedBelowBothSides(unittest.TestCase):
    def test_the_distributed_installer_no_longer_borrows_from_fsdp2_common(self):
        source = (_COMPAT / "torch" / "installers" / "distributed.py").read_text(
            encoding="utf-8")
        self.assertNotIn("fsdp2 import common", source)
        self.assertIn("_collectives._all_gather_shards", source)

    def test_fsdp2_common_still_re_exports_them(self):
        # The move must be invisible to the ~10 `common._all_gather_shards(...)`
        # call sites inside fsdp2.
        from jittor.compat import collectives
        from jittor.compat.fsdp2 import common

        for name in ("_all_gather_shards", "_reduce_scatter_padded", "_rank",
                     "_world_size", "_nccl_ops", "_slice_flat",
                     "_in_true_distributed"):
            with self.subTest(name=name):
                self.assertIs(getattr(common, name), getattr(collectives, name))

    def test_collectives_does_not_depend_on_fsdp2(self):
        offenders = [module for _node, module in _imports(_COMPAT / "collectives.py")
                     if "fsdp2" in module]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
