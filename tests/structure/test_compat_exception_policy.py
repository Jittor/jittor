"""No failure inside the compatibility layer may disappear without a trace.

``except Exception: pass`` was the layer's single most common statement: 144
handlers with a bare ``pass`` body and 276 catching ``Exception`` at large. Each
one threw away the only evidence that would later explain the symptom -- a
propagation marker that never got attached, a dtype that was never restored, an
optimizer grad map that was never rebuilt. The program continued, the numbers
changed, and nothing said why. ``Exception`` also absorbed MemoryError and
SystemExit, so a full disk read as "the optional feature is unavailable", and
absorbed NameError and AssertionError, which are this layer's own bugs.

These are rules, not an inventory: they say what a handler in
``python/jittor/compat`` must look like, and they fail on any new one that does
not. See ``jittor/compat/diagnostics.py`` for the policy tuple and the recorder.
"""

import ast
import unittest
from pathlib import Path

import jittor


_COMPAT = Path(jittor.__file__).resolve().parent / "compat"
_POLICY = "EXPECTED"


#: ``shim/resources`` is not this layer's code. Those files are *deployed*
#: into the shim's site directory and imported there as standalone third-party
#: packages (``flash_attn``, ``torchvision``), so a relative import back into
#: ``jittor.compat`` -- which is what the policy's recorder needs -- cannot
#: resolve once they have been copied. They keep plain handlers.
_NOT_OURS = ("resources",)


def _sources():
    return sorted(path for path in _COMPAT.rglob("*.py")
                  if "__pycache__" not in path.parts
                  and not set(_NOT_OURS) & set(path.relative_to(_COMPAT).parts))


def _handlers():
    for path in _sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                yield path, node


def _is_broad(handler):
    node = handler.type
    if node is None:
        return True
    return isinstance(node, ast.Name) and node.id in ("Exception", "BaseException")


def _only_reraises(handler):
    """A handler that translates an error into a clearer one is not swallowing."""
    return any(isinstance(stmt, ast.Raise) for stmt in handler.body)


def _records(handler):
    return any(isinstance(node, ast.Call)
               and isinstance(node.func, ast.Name)
               and node.func.id == "swallowed"
               for stmt in handler.body for node in ast.walk(stmt))


def _where(path, node):
    return "%s:%d" % (path.relative_to(_COMPAT.parent), node.lineno)


class TestCompatExceptionPolicy(unittest.TestCase):
    def test_no_handler_body_is_only_pass(self):
        offenders = [_where(path, handler) for path, handler in _handlers()
                     if len(handler.body) == 1
                     and isinstance(handler.body[0], ast.Pass)]
        self.assertEqual(
            offenders, [],
            "a handler whose whole body is `pass` leaves nothing behind; "
            "record it with jittor.compat.diagnostics.swallowed(what, exc)")

    def test_no_handler_catches_exception_at_large(self):
        offenders = [_where(path, handler) for path, handler in _handlers()
                     if _is_broad(handler) and not _only_reraises(handler)]
        self.assertEqual(
            offenders, [],
            "name the exceptions the block can raise, or use the declared "
            "policy tuple `diagnostics.EXPECTED`; `except Exception` also "
            "absorbs MemoryError, SystemExit and this layer's own NameErrors")

    def test_every_policy_handler_records_what_it_swallowed(self):
        # EXPECTED is the "this block is heterogeneous enough that a tighter
        # tuple would be a guess" case. Exactly there, the record is the only
        # thing left to debug with.
        offenders = [_where(path, handler) for path, handler in _handlers()
                     if isinstance(handler.type, ast.Name)
                     and handler.type.id == _POLICY
                     and not _records(handler)
                     and not _only_reraises(handler)]
        self.assertEqual(
            offenders, [],
            "a handler catching diagnostics.EXPECTED must call swallowed()")

    def test_the_deployed_stubs_are_excluded_for_a_stated_reason(self):
        # Not an oversight: a bulk rewrite that reached these files wrote
        # `from .....diagnostics import ...` into packages that get copied out
        # of the tree, which cannot resolve after deployment. Pin the exclusion
        # so the next sweep does not rediscover it the same way.
        stubs = _COMPAT / "shim" / "resources" / "stubs"
        self.assertTrue(stubs.is_dir())
        self.assertTrue(any(stubs.rglob("*.py")))
        self.assertNotIn(stubs / "flash_attn" / "__init__.py", _sources())
        for path in stubs.rglob("*.py"):
            with self.subTest(path=path.name):
                self.assertNotIn("diagnostics import",
                                 path.read_text(encoding="utf-8"))

    def test_modules_exec_d_standalone_import_diagnostics_absolutely(self):
        # The stubs above are excluded because they are *copied* out of the
        # tree. These two are not copied, but they are `exec`-ed outside their
        # package, which breaks a relative import just the same:
        #
        #   shim/backends/flash_attention.py -- tests/compat/torch/
        #     test_torch_compat_attention.py reloads it by path, to check its
        #     environment-epoch hook survives a re-exec;
        #   shim/deploy.py -- tests/structure/test_torch_shim_deploy.py loads
        #     it through spec_from_file_location.
        #
        # Both already import the rest of jittor.compat absolutely for exactly
        # this reason, and deploy.py wraps its one relative import in a
        # try/except fallback. The bulk rewrite that introduced `swallowed()`
        # wrote `from ...diagnostics import` into flash_attention.py anyway,
        # and the failure was an ImportError before its first statement ran.
        for relative in ("shim/backends/flash_attention.py", "shim/deploy.py"):
            path = _COMPAT / relative
            with self.subTest(module=relative):
                self.assertTrue(path.is_file())
                tree = ast.parse(path.read_text(encoding="utf-8"))
                offenders = [
                    "%s:%d imports .%s relatively" % (relative, node.lineno,
                                                      node.module or "")
                    for node in tree.body                     # module level only
                    if isinstance(node, ast.ImportFrom) and node.level
                ]
                self.assertEqual(
                    offenders, [],
                    "this module is exec'd with no parent package; a top-level "
                    "relative import raises before the file runs -- spell it "
                    "`from jittor.compat... import`, or guard it with a "
                    "try/except like deploy.py does")

    def test_the_rules_are_actually_looking_at_something(self):
        # A rule over an empty set passes for the wrong reason.
        handlers = list(_handlers())
        self.assertGreater(len(handlers), 200)
        self.assertGreater(
            sum(1 for _, handler in handlers
                if isinstance(handler.type, ast.Name) and handler.type.id == _POLICY),
            100)


class TestExpectedTupleLetsRealFailuresThrough(unittest.TestCase):
    """The value of a policy tuple is what it refuses to catch."""

    def test_it_covers_what_the_probes_provoke(self):
        from jittor.compat.diagnostics import EXPECTED

        for kind in (AttributeError, TypeError, ValueError, KeyError,
                     IndexError, RuntimeError, OSError, ImportError,
                     NotImplementedError):
            with self.subTest(exception=kind.__name__):
                self.assertTrue(issubclass(kind, EXPECTED))

    def test_it_refuses_what_must_never_be_silenced(self):
        from jittor.compat.diagnostics import EXPECTED

        for kind in (KeyboardInterrupt, SystemExit, MemoryError,
                     NameError, AssertionError, StopIteration, GeneratorExit):
            with self.subTest(exception=kind.__name__):
                self.assertFalse(
                    issubclass(kind, EXPECTED),
                    "%s must reach the caller, not be absorbed as 'unsupported'"
                    % kind.__name__)

    def test_the_one_leak_is_stated_rather_than_discovered(self):
        from jittor.compat.diagnostics import EXPECTED

        # RuntimeError has to be on the list: it is what Jittor's C++ core
        # raises. RecursionError is a subclass of it, so a stack overflow inside
        # a probe is still absorbed -- but now with a record attached, which is
        # what separates it from the old behaviour. Pinned here so a future
        # reader finds a decision, not a surprise.
        self.assertTrue(issubclass(RecursionError, RuntimeError))
        self.assertTrue(issubclass(RecursionError, EXPECTED))


if __name__ == "__main__":
    unittest.main()
