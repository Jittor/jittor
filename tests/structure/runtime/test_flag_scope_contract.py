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
    stack = list(ast.iter_child_nodes(node))
    while stack:
        current = stack.pop()
        if skip_nested_classes and isinstance(current, ast.ClassDef):
            continue
        if isinstance(current, (ast.Assign, ast.AugAssign)):
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
        stack.extend(ast.iter_child_nodes(current))
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


def _scope_with_aliased_flags(sync_in_setter=False):
    """Execute production scopes with a host-only flag double, without JIT."""
    import functools
    import sys
    from types import SimpleNamespace
    class Flags:
        def __init__(self):
            self.device = 0
            self.no_grad = 0
            self._controlled = 0
        @property
        def use_cuda(self):
            return self.device
        @use_cuda.setter
        def use_cuda(self, value):
            previous = self.device
            self.device = value
            if sync_in_setter and previous != value:
                # Match the native setter: submit on the old device and
                # transactionally undo the write if submission fails.
                self.device = previous
                namespace['sync_all']()
                self.device = value
        use_acl = use_cuda
        @property
        def controlled(self):
            return self._controlled
        @controlled.setter
        def controlled(self, value):
            self._controlled = value
            if value == 99:
                raise ValueError('rejected setting')
    state = Flags()
    synced = []
    namespace = {'flags': state, 'sync_all': lambda: synced.append(state.device),
                 '_functools': functools, '_sys': sys}
    snapshots = []
    def push_device_mode():
        token = object()
        snapshots.append((token, state.device))
        return token
    def pop_device_mode(token, restore):
        saved_token, previous = snapshots.pop()
        assert token is saved_token
        if restore:
            state.device = previous
    namespace['core'] = SimpleNamespace(
        _push_device_mode_scope=push_device_mode,
        _pop_device_mode_scope=pop_device_mode,
    )
    path = REPO_ROOT / 'python/jittor/_core/flags.py'
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace['flag_scope'], state, synced


def test_alias_scope_restores_device_with_both_keyword_orders():
    for values in ({'use_acl': 1, 'use_cuda': 1}, {'use_cuda': 1, 'use_acl': 1}):
        scope, flags, synced = _scope_with_aliased_flags()
        with scope(**values):
            assert flags.device == 1
        assert flags.device == 0
        assert synced == [0, 1]


def test_alias_scope_nesting_and_decorated_calls_restore_outer_state():
    scope, flags, _ = _scope_with_aliased_flags()
    outer = scope(use_acl=1, use_cuda=1)
    @scope(use_acl=0, use_cuda=0)
    def inner(depth):
        assert flags.device == 0
        if depth:
            inner(depth - 1)
        assert flags.device == 0
    with outer:
        inner(2)
        assert flags.device == 1
        with outer:
            assert flags.device == 1
        assert flags.device == 1
    assert flags.device == 0


def test_alias_scope_body_exception_restores_without_another_flush():
    import pytest
    scope, flags, synced = _scope_with_aliased_flags()
    with pytest.raises(ValueError, match='body failed'):
        with scope(use_acl=1, use_cuda=1):
            raise ValueError('body failed')
    assert flags.device == 0
    assert synced == [0]


def test_alias_scope_setting_failure_rolls_back_and_preserves_outer_entry():
    import pytest
    scope, flags, synced = _scope_with_aliased_flags()
    with scope(use_acl=1, use_cuda=1):
        failed = scope(use_acl=0, use_cuda=0, controlled=99)
        with pytest.raises(ValueError, match='rejected setting'):
            with failed:
                raise AssertionError('failed setting must not enter body')
        assert flags.device == 1 and flags.controlled == 0
        assert failed._flags_bk_stack == []
        assert synced == [0, 1]
    assert flags.device == 0


def test_alias_only_scope_flushes_both_device_boundaries():
    scope, flags, synced = _scope_with_aliased_flags()
    with scope(use_acl=1):
        assert flags.device == 1
    assert flags.device == 0 and synced == [0, 1]


def test_alias_scope_snapshot_failure_does_not_mutate_or_flush():
    import pytest
    scope, flags, synced = _scope_with_aliased_flags()
    failed = scope(use_acl=1, use_cuda=1, nonexistent=1)
    with pytest.raises(AttributeError):
        failed.__enter__()
    assert flags.device == 0 and synced == []
    assert failed._flags_bk_stack == []


def test_alias_scope_exit_flush_failure_still_restores_all_originals():
    import pytest
    scope, flags, _ = _scope_with_aliased_flags()
    def fail_sync():
        raise RuntimeError('synchronization failed')
    with pytest.raises(RuntimeError, match='synchronization failed'):
        with scope(use_acl=1, use_cuda=1):
            scope.__enter__.__globals__['sync_all'] = fail_sync
    assert flags.device == 0


def test_scope_body_error_does_not_resubmit_through_native_style_setter():
    import pytest
    scope, flags, _ = _scope_with_aliased_flags(sync_in_setter=True)
    original = ValueError('original body error')
    attempts = []
    def failed_pending_graph():
        attempts.append(flags.device)
        raise RuntimeError('pending graph must not be resubmitted')
    with pytest.raises(ValueError) as caught:
        with scope(use_acl=1, use_cuda=1, no_grad=1):
            scope.__enter__.__globals__['sync_all'] = failed_pending_graph
            raise original
    assert caught.value is original
    assert attempts == []
    assert flags.device == 0 and flags.no_grad == 0


def test_scope_flush_failure_restores_without_retrying_native_style_setter():
    import pytest
    scope, flags, _ = _scope_with_aliased_flags(sync_in_setter=True)
    original = RuntimeError('first graph submission failed')
    attempts = []
    def failed_pending_graph():
        attempts.append(flags.device)
        raise original
    with pytest.raises(RuntimeError) as caught:
        with scope(use_cuda=1, use_acl=1, no_grad=1):
            scope.__enter__.__globals__['sync_all'] = failed_pending_graph
    assert caught.value is original
    assert attempts == [1]
    assert flags.device == 0 and flags.no_grad == 0


def test_direct_device_setter_still_rejects_switch_on_submission_failure():
    import pytest
    scope, flags, _ = _scope_with_aliased_flags(sync_in_setter=True)
    flags.use_cuda = 1
    def failed_pending_graph():
        raise RuntimeError('cannot switch while graph submission fails')
    scope.__enter__.__globals__['sync_all'] = failed_pending_graph
    with pytest.raises(RuntimeError, match='cannot switch'):
        flags.use_cuda = 0
    assert flags.device == 1
