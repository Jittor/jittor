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


def _flag_assignments(node, skip_nested_classes=True):
    """``jt.flags.X = ...`` written directly in ``node``, not in a nested class."""
    found = []

    def dotted(value):
        if isinstance(value, ast.Name):
            return value.id
        if isinstance(value, ast.Attribute):
            prefix = dotted(value.value)
            return (prefix + "." if prefix else "") + value.attr
        return ""

    def visit(current, scoped=False):
        if skip_nested_classes and isinstance(current, ast.ClassDef):
            return
        if isinstance(current, ast.With):
            scoped = scoped or any(
                (
                    dotted(item.context_expr.func)
                    if isinstance(item.context_expr, ast.Call)
                    else dotted(item.context_expr)
                ) in ("jt.flag_scope", "jittor.flag_scope",
                      "jt.runtime.scope", "jittor.runtime.scope")
                for item in current.items
            )
        if not scoped and isinstance(current, (ast.Assign, ast.AugAssign)):
            targets = (current.targets if isinstance(current, ast.Assign)
                       else [current.target])
            pending = list(targets)
            while pending:
                target = pending.pop()
                if isinstance(target, (ast.Tuple, ast.List)):
                    pending.extend(target.elts)
                elif (isinstance(target, ast.Attribute)
                      and isinstance(target.value, ast.Attribute)
                      and target.value.attr == "flags"):
                    found.append((target.attr, current.lineno))
        for child in ast.iter_child_nodes(current):
            visit(child, scoped)

    for child in ast.iter_child_nodes(node):
        visit(child)
    return found


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


def test_no_test_leaves_a_jittor_flag_changed():
    offenders = []
    for path, relative in _candidate_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
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
    assert offenders == [], "\n".join(sorted(offenders))


def test_every_exemption_states_a_reason_and_still_exists():
    problems = []
    for relative, reason in _ALLOWED.items():
        if not reason.strip():
            problems.append("%s is exempt with no reason" % relative)
        if not (REPO_ROOT / relative).exists():
            problems.append("%s is exempt but no longer exists" % relative)
    assert problems == [], "\n".join(problems)
