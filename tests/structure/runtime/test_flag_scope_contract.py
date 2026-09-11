"""A test that changes a jittor flag has to put it back.

``jt.flags`` is process-global. A test that assigns one and never restores it
does not fail; the tests that run after it do, somewhere else, for reasons that
have nothing to do with them.

The expensive instance is 6.P23. ``tests/ops/test_linalg.py::TestBUG4_2Op`` set
``use_cuda=1`` and never restored it, so every later case in that file ran on
CUDA -- where ``eigh``'s gradient is wrong. The file reported success, because
the cases that would have caught the wrong gradient were exactly the ones being
silently redirected onto the broken path.

``jt.flag_scope`` exists for this and unwinds on exceptions too. A ``setUp`` that
saves the previous value and a ``tearDown`` that restores it is equally fine, and
so is ``try/finally`` -- the rule below accepts all three and rejects a bare
assignment with nothing to undo it.

Restore what was there, not a constant: Jittor turns CUDA on by default when a
GPU is present, so a ``tearDown`` that hard-codes ``use_cuda = 0`` switches the
accelerator off for the rest of the session rather than putting it back.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_ROOT = REPO_ROOT / "tests"

_TEARDOWN = {"tearDown", "tearDownClass", "tearDownModule"}

#: Files allowed to assign a flag without restoring it, and why.
_ALLOWED = {
    "tests/runtime/test_flags.py":
        "the flag mechanism itself is what is under test: it asserts that an "
        "unknown name raises and that a known one round-trips",
    "compat/tests/torch/_ecosystem_runner.py":
        "not a test module -- a child-process entry point whose whole job is "
        "to select the device the run executes on",
}

#: Directories outside the rule: probes a person runs by hand, never collected.
_ALLOWED_PREFIXES = ("tests/backends/acl/manual/",)


def _candidate_files():
    from _helpers.paths import iter_test_files
    for path in iter_test_files("*.py"):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if relative in _ALLOWED:
            continue
        if relative.startswith(_ALLOWED_PREFIXES):
            continue
        if path.name.startswith("test_") or path.parent.name == "_helpers":
            yield path, relative


#: Trailing attribute names of ``with`` context managers that assert their body
#: raises. An assignment that *is* the whole body of one of these never
#: happened: the exception it provokes is the thing under test, so there is no
#: leftover value for a ``tearDown`` to put back, and demanding one would ask a
#: test to restore a flag it was asserting it could not change.
#:
#: Narrow in two directions, both load-bearing. ``pytest.warns`` is absent
#: because an assignment under it *succeeds* -- warning and all -- and still
#: has to be restored. And only a single-statement body qualifies: in a longer
#: block the exception may come from any line, leaving every assignment ahead
#: of it in force.
_ASSERTS_ITS_BODY_RAISES = ("raises", "assertRaises", "assertRaisesRegex")


def _asserts_its_body_raises(node):
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        function = call.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(
            function, "id", "")
        if name in _ASSERTS_ITS_BODY_RAISES:
            return True
    return False


def _assignment_is_the_asserted_failure(node):
    """``with pytest.raises(...): jt.flags.X = ...`` -- the assignment fails."""
    return (isinstance(node, (ast.With, ast.AsyncWith))
            and len(node.body) == 1
            and isinstance(node.body[0], (ast.Assign, ast.AugAssign))
            and _asserts_its_body_raises(node))


#: Context managers that snapshot the flags they are handed and put them back
#: on the way out, exceptions included -- the very mechanism this rule tells
#: people to use. An assignment inside one, to a flag that scope was given, is
#: restored by it.
_RESTORING_SCOPES = ("flag_scope", "scope")

#: One native flag published under five names. A scope handed any of them
#: restores all of them, which is what
#: ``tests/runtime/test_native_backend_registry.py`` asserts directly: assign
#: ``use_device`` and every name in this family reads back as 1.
_ACCELERATOR_MODE_ALIASES = frozenset(
    ("use_cuda", "use_device", "use_acl", "use_rocm", "use_corex"))


def _flags_a_scope_restores(node):
    """Flags a ``with jt.flag_scope(...)`` puts back when the block ends."""
    names = set()
    for item in node.items:
        call = item.context_expr
        if not isinstance(call, ast.Call):
            continue
        function = call.func
        name = function.attr if isinstance(function, ast.Attribute) else getattr(
            function, "id", "")
        if name not in _RESTORING_SCOPES:
            continue
        for keyword in call.keywords:
            if keyword.arg is None:
                continue
            names.add(keyword.arg)
            if keyword.arg in _ACCELERATOR_MODE_ALIASES:
                names |= _ACCELERATOR_MODE_ALIASES
    return names


def _assigned_flag_names(node):
    """``(name, line)`` for every ``something.flags.NAME`` this node assigns."""
    found = []
    targets = (node.targets if isinstance(node, ast.Assign) else [node.target])
    pending = list(targets)
    while pending:
        target = pending.pop()
        if isinstance(target, (ast.Tuple, ast.List)):
            pending.extend(target.elts)
        elif (isinstance(target, ast.Attribute)
              and isinstance(target.value, ast.Attribute)
              and target.value.attr == "flags"):
            found.append((target.attr, node.lineno))
    return found


def _flag_assignments(node, skip_nested_classes=True):
    """``jt.flags.X = ...`` in ``node`` with nothing on this node undoing it.

    Two of the three accepted ways to undo one are visible right here and are
    applied as the walk descends: an assignment the test asserts will raise
    (it never took effect) and an assignment inside a scope that restores that
    flag. The third -- a ``tearDown`` or a ``finally`` elsewhere in the class --
    stays with the caller, which is the only place that can see it.
    """
    found = []
    _walk_flag_assignments(node, skip_nested_classes, frozenset(), found)
    return found


def _walk_flag_assignments(node, skip_nested_classes, restored, found):
    for current in ast.iter_child_nodes(node):
        if skip_nested_classes and isinstance(current, ast.ClassDef):
            continue
        if _assignment_is_the_asserted_failure(current):
            continue
        inner = restored
        if isinstance(current, (ast.With, ast.AsyncWith)):
            inner = restored | _flags_a_scope_restores(current)
        if isinstance(current, (ast.Assign, ast.AugAssign)):
            for name, line in _assigned_flag_names(current):
                if name not in restored:
                    found.append((name, line))
        _walk_flag_assignments(current, skip_nested_classes, inner, found)


def _restored_in_finally(function):
    """Flags this function puts back in a ``finally:`` block."""
    names = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Try):
            continue
        for statement in node.finalbody:
            body = ast.Module(body=[statement], type_ignores=[])
            names.update(name for name, _line in _flag_assignments(body))
    return names


def _cleanup_method_names(class_node):
    """Methods registered with ``self.addCleanup`` count as teardown."""
    names = set()
    for node in ast.walk(class_node):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (isinstance(function, ast.Attribute) and function.attr == "addCleanup"):
            continue
        for argument in node.args:
            if isinstance(argument, ast.Attribute):
                names.add(argument.attr)
    return names


def _class_offenders(class_node, relative):
    cleanup_methods = _cleanup_method_names(class_node) | _TEARDOWN
    restored = set()
    methods = [node for node in class_node.body
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for method in methods:
        if method.name in cleanup_methods:
            restored.update(name for name, _line in _flag_assignments(method))
    offenders = []
    for method in methods:
        if method.name in cleanup_methods:
            continue
        allowed = restored | _restored_in_finally(method)
        for name, line in _flag_assignments(method):
            if name not in allowed:
                offenders.append(
                    "%s:%d %s.%s assigns jt.flags.%s and nothing restores it; "
                    "use jt.flag_scope or a tearDown that puts the previous "
                    "value back"
                    % (relative, line, class_node.name, method.name, name))
    return offenders


def _module_offenders(tree, relative):
    """Every unrestored flag assignment in one parsed module."""
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            offenders.extend(_class_offenders(node, relative))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            allowed = _restored_in_finally(node)
            for name, line in _flag_assignments(node):
                if name not in allowed:
                    offenders.append(
                        "%s:%d %s() assigns jt.flags.%s and nothing restores "
                        "it; use jt.flag_scope"
                        % (relative, line, node.name, name))
        elif not isinstance(node, ast.ClassDef):
            for name, line in _flag_assignments(node, skip_nested_classes=False):
                offenders.append(
                    "%s:%d assigns jt.flags.%s at module scope; every file "
                    "imported afterwards inherits it" % (relative, line, name))
    return offenders


def test_no_test_leaves_a_jittor_flag_changed():
    offenders = []
    for path, relative in _candidate_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        offenders.extend(_module_offenders(tree, relative))
    assert offenders == [], "\n".join(sorted(offenders))


#: The five shapes the ``pytest.raises`` exemption has to tell apart. The first
#: two are what the rule used to get wrong: the assignment *is* the failure
#: being asserted, so the flag still holds the value it had, and demanding a
#: ``tearDown`` asked the test to put back something it never changed. Three
#: real cases sat behind that report -- an invalid CUDA device index, a
#: deprecated accelerator alias under ``simplefilter("error")``, and the
#: startup-only ``disable_lock`` -- and none of them could be fixed, only
#: exempted, which is how a rule stops being read.
#:
#: The last three are why the exemption is not simply "anything under a
#: ``with``": a ``warns`` block does not stop the assignment, a longer
#: ``raises`` block may raise after it, and a bare assignment is the original
#: defect. Line numbers are asserted below; keep them in step when editing.
_RAISES_AND_LEAK_SHAPES = """\
import pytest


def test_the_setter_refuses_and_the_flag_keeps_its_value():
    with pytest.raises(RuntimeError, match="Invalid CUDA device index"):
        jt.flags.device_id = 99


class TestUnittestForm:

    def test_the_setter_refuses(self):
        with self.assertRaises(ValueError):
            jt.flags.log_vprefix = "not a prefix spec"


def test_a_warning_does_not_stop_the_assignment():
    with pytest.warns(DeprecationWarning):
        jt.flags.use_device = True


def test_a_longer_raises_block_may_raise_after_the_assignment():
    with pytest.raises(RuntimeError):
        jt.flags.lazy_execution = 0
        _something_else()


def test_a_bare_assignment_is_still_an_offender():
    jt.flags.no_grad = 1
"""


def test_an_assignment_asserted_to_raise_is_not_an_unrestored_flag():
    offenders = _module_offenders(ast.parse(_RAISES_AND_LEAK_SHAPES), "shapes.py")
    flagged = {line.split("jt.flags.", 1)[1].split(" ", 1)[0] for line in offenders}

    # The assignment never took effect, so there is nothing to restore.
    assert "device_id" not in flagged, offenders
    assert "log_vprefix" not in flagged, offenders

    # ...and the rule still has teeth in the three shapes that do leak.
    assert "use_device" in flagged, offenders
    assert "lazy_execution" in flagged, offenders
    assert "no_grad" in flagged, offenders
    assert len(offenders) == 3, offenders


#: What ``jt.flag_scope`` covers and where its cover stops. The rule's own
#: docstring names it as one of the three ways to put a flag back, but the
#: implementation only ever recognised it by accident: a test inside a scope
#: normally passes flags as *arguments* and assigns nothing, so an assignment
#: that the scope does restore was reported like one nobody restores.
#: ``tests/backends/cuda/test_cuda_runtime_device_state.py`` is that case --
#: two assignments inside ``with jt.flag_scope(use_cuda=0)``, one of them
#: through the deprecated ``use_device`` name of the same native flag.
#: Line numbers are asserted below; keep them in step.
_SCOPE_SHAPES = """\
import jittor as jt


def test_the_scope_puts_back_what_the_body_assigns():
    with jt.flag_scope(use_cuda=0):
        jt.flags.use_cuda = 1


def test_the_scope_restores_every_published_name_of_that_flag():
    with jt.runtime.scope(use_cuda=0):
        jt.flags.use_device = True


def test_a_scope_not_given_this_flag_restores_nothing():
    with jt.flag_scope(use_cuda=0):
        jt.flags.lazy_execution = 0


def test_the_cover_ends_with_the_block():
    with jt.flag_scope(use_cuda=0):
        pass
    jt.flags.use_cuda = 1
"""


def test_a_scope_restores_the_flags_it_was_handed_and_no_others():
    offenders = _module_offenders(ast.parse(_SCOPE_SHAPES), "scopes.py")

    assert [entry.split(" ", 1)[0] for entry in offenders] == [
        "scopes.py:16",  # a flag the scope was never given
        "scopes.py:22",  # assigned after the block ended
    ], offenders
    assert "lazy_execution" in offenders[0]
    assert "use_cuda" in offenders[1]


def test_every_exemption_states_a_reason_and_still_exists():
    problems = []
    for relative, reason in _ALLOWED.items():
        if not reason.strip():
            problems.append("%s is exempt with no reason" % relative)
        if not (REPO_ROOT / relative).exists():
            problems.append("%s is exempt but no longer exists" % relative)
    assert problems == [], "\n".join(problems)
