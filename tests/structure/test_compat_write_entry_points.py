"""Every process-global write in ``jittor/compat`` is classified, or the run fails.

7.05 asks for an install that is either fully reversible or an explicit hard
failure. The obstacle was never the ledger -- it was not knowing what still had
to go into it. Eleven waves of the board recorded "some other installer's write
entry points remain", each time from a fresh grep, each time with a different
answer, because a grep says what matched and not what is left.

So the inventory lives here as a closed set. The scanner finds every statement in
``compat`` that writes ``os.environ``, ``sys.modules``,
``sys.meta_path``, ``sys.path``, ``builtins.__import__``, or a Jittor flag, and
every one of them has to appear in ``CLASSIFIED`` with a category. A new write
entry point fails this test until somebody writes down which of the five it is,
and finishing a ``PENDING`` item forces the list to shrink in the same diff.

The categories are deliberately few, and four of them are reasons *not* to be in
the install ledger:

``ledger``
    Recorded through an ``InstallTransaction``/``ActivationTransaction``, so a
    failed install reverts it.
``runtime``
    A request the caller made after the install finished -- ``Module.to()``,
    ``allow_tf32``, a ``no_grad`` scope, an optimizer's ``node_order``. Rolling
    these back with an install would undo something the user asked for. Each one
    restores itself where it has a scope at all.
``pre-ledger``
    Runs before any transaction exists: the shim preflight prepares the
    environment *before* the core is imported (``setup_nccl`` reads ``use_nccl``
    once during that import), and plain composition publishes aliases before Torch
    mode is even chosen.
``deployed-payload``
    Inside the deployed shim payload, which executes in a separate process from
    the install that produced it. There is nothing in-process to roll back.
``pending``
    Genuinely still owed to 7.05, with the specific obstacle recorded.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPAT = ROOT / "compat"

_ENVIRON_OWNERS = ("os.environ", "environ")
_MODULE_OWNERS = ("sys.modules", "_sys.modules", "modules")
_META_PATH_OWNERS = ("sys.meta_path", "_sys.meta_path", "meta_path")
_PATH_OWNERS = ("sys.path", "_sys.path")
_BUILTINS_OWNERS = ("builtins", "_builtins", "builtins_module")
# Per-owner verbs, because the bare local names matter here and their shapes
# differ. ``modules.insert(0, self)`` in installers/nn.py is a plain list of
# submodules, not the module table -- ``sys.modules`` is a dict and has no
# ``insert`` -- so matching every mutating verb on every owner reported it as a
# process-global write.
_DICT_VERBS = ("setdefault", "update", "pop")
_LIST_VERBS = ("insert", "append", "remove", "pop")


def _dotted(node):
    """``a.b.c`` for an attribute or name chain, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def _kind_of_assignment(target):
    if isinstance(target, ast.Subscript):
        base = _dotted(target.value)
        if base in _ENVIRON_OWNERS:
            return "env"
        if base in _MODULE_OWNERS:
            return "sys.modules"
        if base in _META_PATH_OWNERS:
            return "sys.meta_path"
        if base in _PATH_OWNERS:
            return "sys.path"
        return None
    if isinstance(target, ast.Attribute):
        name = _dotted(target)
        if name is None:
            return None
        if name.endswith(".__import__"):
            return "builtins.__import__"
        if name.startswith("flags.") or ".flags." in name:
            return "flags"
    return None


def _kind_of_call(node):
    name = _dotted(node.func)
    if name is None:
        return None
    for owners, verbs, kind in (
        (_ENVIRON_OWNERS, _DICT_VERBS, "env"),
        (_MODULE_OWNERS, _DICT_VERBS, "sys.modules"),
        (_META_PATH_OWNERS, _LIST_VERBS, "sys.meta_path"),
        (_PATH_OWNERS, _LIST_VERBS, "sys.path"),
    ):
        for base in owners:
            if name in tuple(base + "." + verb for verb in verbs):
                return kind
    if name == "setattr" and node.args:
        first = _dotted(node.args[0])
        if first is None:
            return None
        if first == "flags" or first.endswith(".flags"):
            return "flags"
        if first in _BUILTINS_OWNERS:
            return "builtins.__import__"
    return None


def _enclosing_function_names(tree):
    """{id(node): name of the innermost enclosing def, or "<module>"}."""
    parents = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def owner_of(node):
        current = node
        while id(current) in parents:
            current = parents[id(current)]
            if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
                return current.name
        return "<module>"

    owner_of.parents = parents
    return owner_of


def _is_local_module_memo(node, parents):
    """A closure over a fresh dict is not the process module table."""
    scope = parents.get(id(node))
    while scope is not None:
        if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module)):
            if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = getattr(scope.args, "posonlyargs", []) + scope.args.args + scope.args.kwonlyargs
                if any(argument.arg == "modules" for argument in arguments):
                    return False
            bindings = []
            pending = list(scope.body)
            while pending:
                current = pending.pop()
                if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
                    continue
                if isinstance(current, (ast.Global, ast.Nonlocal)) and "modules" in current.names:
                    return False
                if isinstance(current, (ast.Assign, ast.AnnAssign)):
                    targets = current.targets if isinstance(current, ast.Assign) else [current.target]
                    if any(isinstance(target, ast.Name) and target.id == "modules" for target in targets):
                        bindings.append(current.value)
                pending.extend(ast.iter_child_nodes(current))
            if bindings:
                return all(isinstance(value, ast.Dict) and not value.keys for value in bindings)
        scope = parents.get(id(scope))
    return False


def discover_write_entry_points():
    """{(relative path, enclosing def, kind): sorted line numbers}.

    Keyed on the enclosing definition rather than the line, so moving code inside
    a function does not churn the table while moving a write into a *different*
    function does -- which is the case worth re-reviewing.
    """
    found = {}
    for path in sorted(COMPAT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        owner_of = _enclosing_function_names(tree)
        relative = path.relative_to(ROOT).as_posix()
        for node in ast.walk(tree):
            kind = None
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    kind = kind or _kind_of_assignment(target)
            elif isinstance(node, ast.AugAssign):
                kind = _kind_of_assignment(node.target)
            elif isinstance(node, ast.Call):
                kind = _kind_of_call(node)
            if kind is None:
                continue
            if kind == "sys.modules":
                targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
                bases = [_dotted(target.value) for target in targets
                         if isinstance(target, ast.Subscript) and _kind_of_assignment(target) == kind]
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    bases = [_dotted(node.func.value)]
                if bases and all(base == "modules" for base in bases) \
                        and _is_local_module_memo(node, owner_of.parents):
                    continue
            key = (relative, owner_of(node), kind)
            found.setdefault(key, []).append(node.lineno)
    return {key: sorted(lines) for key, lines in found.items()}


def test_local_memo_detection_does_not_exempt_module_table_aliases():
    cases = (
        ("def build():\n modules = {}\n def copy():\n  modules[id(source)] = result\n", True),
        ("def build():\n modules = sys.modules\n def copy():\n  modules[name] = result\n", False),
        ("def publish(modules):\n modules[name] = result\n", False),
        ("def publish():\n modules = {}\n modules = sys.modules\n modules[name] = result\n", False),
    )
    for source, expected in cases:
        tree = ast.parse(source)
        owners = _enclosing_function_names(tree)
        write = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Subscript))
        assert _kind_of_assignment(write.targets[0]) == "sys.modules"
        assert _is_local_module_memo(write, owners.parents) is expected


C = "compat/"

CLASSIFIED = {
    # ---- the ledger itself -------------------------------------------------
    (C + "transaction.py", "mutate_flag", "flags"): "ledger",
    (C + "transaction.py", "set_flag", "flags"): "ledger",
    (C + "transaction.py", "publish_module", "sys.modules"): "ledger",
    # `_restore_namespace` is registered as the transaction's whole-namespace
    # undo (`record_undo`) before any required step runs.
    (C + "torch/__init__.py", "_restore_namespace", "sys.modules"): "ledger",
    # Module publication goes through ModuleRegistry, which the namespace undo
    # above covers as one snapshot.
    (C + "torch/library.py", "install_torch_library", "sys.modules"): "ledger",
    (C + "torch/nn_modules.py", "install_module_namespace", "sys.modules"): "ledger",
    (C + "torch/publication.py", "bind_published_namespace", "sys.modules"): "ledger",
    # Owner-aware finder/registry undos.
    (C + "module_patcher.py", "install_module_patches", "sys.meta_path"): "ledger",
    (C + "module_patcher.py", "restore_finder", "sys.meta_path"): "ledger",
    (C + "module_patcher.py", "uninstall_module_patches", "sys.meta_path"): "ledger",
    (C + "permissive.py", "install_permissive_package", "sys.meta_path"): "ledger",
    (C + "permissive.py", "restore_finder", "sys.meta_path"): "ledger",
    (C + "vllm/__init__.py", "register", "sys.meta_path"): "ledger",
    (C + "vllm/__init__.py", "restore_finder", "sys.meta_path"): "ledger",
    # Activation transaction: mutate_path / publish_module / mutate_flag.
    (C + "shim/runtime.py", "_activate_once", "flags"): "ledger",
    (C + "shim/runtime.py", "_activate_once", "sys.modules"): "ledger",
    (C + "shim/runtime.py", "_activate_once", "env"): "ledger",
    (C + "torch/installers/core.py", "install_misc", "sys.modules"): "ledger",
    (C + "shim/runtime.py", "_publish_torch_module", "sys.modules"): "ledger",

    # ---- runtime requests, not installation steps --------------------------
    # torch.backends.cuda.matmul.allow_tf32 = True and friends.
    (C + "torch/installers/cuda/api.py", "_tf32_set", "flags"): "runtime",
    # Module.to(device="cuda") turns CUDA on because the caller asked, after the
    # install has finished.
    (C + "torch/installers/nn/module_methods.py", "_module_to", "flags"): "runtime",
    # Scoped: __enter__/__exit__ restore the entry value themselves.
    (C + "torch/grad.py", "__enter__", "flags"): "runtime",
    (C + "torch/grad.py", "__exit__", "flags"): "runtime",
    # node_order is set and restored inside one optimizer step.
    (C + "torch/optimizer_api.py", "_adam_step", "flags"): "runtime",
    (C + "torch/optimizer_api.py", "_step_with_closure", "flags"): "runtime",
    (C + "torch/optimizer_api.py", "_torch_post_step", "flags"): "runtime",
    (C + "torch/serialization/portable.py", "_apply_map_location", "flags"): "runtime",
    (C + "fsdp2/optimizer.py", "optimizer_step", "flags"): "runtime",
    # Scoped environment overrides around one borrow/copy region.
    (C + "shim/extensions/readonly.py", "_borrow_scope", "env"): "runtime",
    (C + "shim/extensions/readonly.py", "_copy_scope", "env"): "runtime",
    (C + "shim/backends/flash_attention/__init__.py",
     "_merge_capability_env_list", "env"): "runtime",
    # A collective asking for NCCL on a job the preflight did not see; setdefault,
    # so it never overwrites a decision someone else made.
    (C + "collectives.py", "_nccl_ops", "env"): "runtime",

    # ---- before any transaction exists -------------------------------------
    # The preflight runs before `import jittor` loads the core, which is the only
    # safe place to decide use_nccl: setup_nccl() reads it once during that
    # import. There is no ledger yet, and creating one here would have to outlive
    # the core import.
    (C + "shim/preflight.py", "_prepend_env_path", "env"): "pre-ledger",
    (C + "shim/preflight.py", "_set_env_dir", "env"): "pre-ledger",
    (C + "shim/preflight.py", "_prepare_kernel_math", "env"): "pre-ledger",
    (C + "shim/preflight.py", "_configure_cuda", "env"): "pre-ledger",
    (C + "shim/preflight.py", "configure_torch_math_flags", "env"): "pre-ledger",
    (C + "shim/preflight.py", "configure_torch_math_flags", "flags"): "pre-ledger",
    (C + "shim/preflight.py", "prepend_sys_path", "sys.path"): "pre-ledger",
    (C + "shim/preflight.py", "append_sys_path", "sys.path"): "pre-ledger",
    # compose() publishes aliases before Torch mode is chosen, so this runs on
    # plain `import jittor` too and is not part of any Torch install.
    # The canonical Triton domain is part of plain Jittor startup, same reason.
    (C + "triton/__init__.py", "install", "sys.modules"): "pre-ledger",
    (C + "triton/__init__.py", "_ensure_libcuda_linkable", "env"): "pre-ledger",

    # ---- deployed payload, a different process -----------------------------
    (C + "shim/resources/torch/__init__.py", "<module>", "sys.modules"): "deployed-payload",
    (C + "shim/resources/stubs/torchvision/__init__.py",
     "<module>", "sys.meta_path"): "deployed-payload",
    (C + "shim/resources/stubs/torchvision/__init__.py",
     "<module>", "sys.modules"): "deployed-payload",
    (C + "shim/resources/stubs/torchaudio/__init__.py",
     "__getattr__", "sys.modules"): "deployed-payload",
    (C + "shim/resources/stubs/torchdata/__init__.py",
     "__getattr__", "sys.modules"): "deployed-payload",

    # Resolver-owned path tokens and current-thread loader publications only.
    # Untracked publications remain intact and cause a hard conflict.
    (C + "external_backend.py", "_add_source_to_sys_path", "sys.path"): "ledger",
    (C + "external_backend.py", "_restore_source_import_state", "sys.path"): "ledger",
    (C + "external_backend.py",
     "_restore_source_import_state", "sys.modules"): "ledger",
    (C + "external_backend.py", "import_local", "sys.modules"): "ledger",
    (C + "external_backend.py", "publish_source_module", "sys.modules"): "ledger",
    (C + "external_backend.py", "_load_candidate", "sys.meta_path"): "ledger",
    (C + "transaction.py", "replace_module", "sys.modules"): "ledger",
    (C + "transaction.py", "undo", "sys.modules"): "ledger",
}

#: What 7.05 still owes, as the reason it is not done rather than a bare list.
PENDING = {}

CATEGORIES = ("ledger", "runtime", "pre-ledger", "deployed-payload", "pending")


def test_every_process_global_write_in_compat_is_classified():
    discovered = discover_write_entry_points()
    unclassified = sorted(
        "%s::%s writes %s (lines %s)"
        % (path, owner, kind, ", ".join(str(line) for line in lines))
        for (path, owner, kind), lines in discovered.items()
        if (path, owner, kind) not in CLASSIFIED
    )
    assert unclassified == [], (
        "new process-global write entry points; classify each one in "
        "CLASSIFIED (see this module's docstring for the five categories):\n"
        + "\n".join(unclassified)
    )


def test_the_classification_table_has_no_entries_that_no_longer_exist():
    """A stale exemption is how a closed set stops being closed."""
    discovered = discover_write_entry_points()
    stale = sorted(
        "%s::%s writes %s" % key
        for key in CLASSIFIED
        if key not in discovered
    )
    assert stale == [], (
        "these classified write entry points are gone; drop them from "
        "CLASSIFIED so the table keeps describing the tree:\n" + "\n".join(stale)
    )


def test_every_category_is_one_of_the_five_and_pending_is_explained():
    unknown = sorted(
        "%s::%s writes %s -> %r" % (key + (category,))
        for key, category in CLASSIFIED.items()
        if category not in CATEGORIES
    )
    assert unknown == []

    pending_files = {path for (path, _owner, _kind), category
                     in CLASSIFIED.items() if category == "pending"}
    assert pending_files == set(PENDING), (
        "PENDING must name exactly the files with unfinished write entry "
        "points, so finishing one shrinks the recorded list in the same diff"
    )
    for path, reason in PENDING.items():
        assert len(reason) > 40, path


def test_installers_do_not_rederive_the_active_transaction():
    """One owner for the lookup, or the copies drift.

    Six installers used to inline ``context.state.get("_install_transaction")``
    and five of them forgot to check whether the transaction was still open, so a
    ledger left behind by a failed install turned the next write into a
    RuntimeError. ``jittor/compat/transaction.py`` owns the lookup now.
    """
    offenders = []
    for path in sorted(COMPAT.rglob("*.py")):
        if path.name == "transaction.py":
            continue
        text = path.read_text(encoding="utf-8")
        if '"_install_transaction"' not in text:
            continue
        for number, line in enumerate(text.splitlines(), start=1):
            if '.get("_install_transaction")' in line:
                offenders.append(
                    "%s:%d" % (path.relative_to(ROOT).as_posix(), number)
                )
    assert offenders == [], (
        "call jittor.compat.transaction.active_transaction() instead of "
        "re-deriving the active transaction:\n" + "\n".join(offenders)
    )
