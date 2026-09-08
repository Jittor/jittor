"""Python dispatch uses native placement and the table used by public ops."""

import ast
from pathlib import Path


RUNTIME = Path(__file__).resolve().parents[3] / "python/jittor/_runtime"


def _backend_reads(tree):
    names = {"use_cuda", "use_acl", "use_rocm", "use_corex", "use_device",
             "is_cuda", "has_acl", "has_rocm"}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load)
                and node.attr in names):
            yield node
        elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
              and node.func.id == "getattr" and len(node.args) >= 2
              and isinstance(node.args[1], ast.Str) and node.args[1].s in names):
            yield node


def test_operator_domains_use_registered_device_selection():
    source = RUNTIME.parent
    offenders = []
    for directory in ("nn", "ops", "contrib/math_util", "fft", "optim"):
        for path in (source / directory).rglob("*.py"):
            tree = ast.parse(path.read_text())
            # These are explicit user device-transfer commands, not kernel
            # selectors. Their setter validation belongs to the runtime.
            if path == source / "ops/tensor_protocol.py":
                tree.body = [node for node in tree.body
                             if not isinstance(node, ast.FunctionDef)
                             or node.name not in {"cuda", "npu"}]
            for node in _backend_reads(tree):
                offenders.append("%s:%s" % (path.relative_to(source), node.lineno))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and node.attr == "compile_extern"
                        and isinstance(node.value, ast.Name) and node.value.id == "jt"):
                    offenders.append("%s:%s direct compiler library access" %
                                     (path.relative_to(source), node.lineno))
    assert not offenders, "\n".join(offenders)


def test_device_selection_rule_detects_guards_but_not_flag_writes():
    guards = ast.parse("""if jt.introspection.policy.runtime.use_cuda or getattr(jt.compiler, "has_acl", False):
    run()""")
    assert len(list(_backend_reads(guards))) == 2
    assert not list(_backend_reads(ast.parse("jt.flags.use_cuda = 1")))


def _parse(name):
    return ast.parse((RUNTIME / name).read_text())


def test_python_backend_prototype_is_retired_from_runtime_exports():
    assert not (RUNTIME / "registry.py").exists()
    tree = _parse("__init__.py")
    imports = {node.module: {alias.name for alias in node.names}
               for node in tree.body if isinstance(node, ast.ImportFrom)}
    assert "registry" not in imports
    assert imports["dispatch"] == {
        "DispatchContext", "KernelRegistration", "dispatch_context", "optional_kernel",
        "override_kernel", "register_kernel", "registered_kernel", "select_kernel",
        "try_dispatch", "unregister_kernel",
    }
    assert imports["fallback"] == {"forbid_backend_fallbacks"}


def test_python_dispatch_queries_native_placement_without_fake_backend_capabilities():
    tree = _parse("dispatch.py")
    context = next(node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "dispatch_context")
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "dispatch_context"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "core"
        for node in ast.walk(context)
    )
    # Backend selection belongs to the native query, including no-input calls.
    assert not any(isinstance(node, ast.Str) and node.s in {"cpu", "cuda", "acl"}
                   for node in ast.walk(context))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == "bytearray" for node in ast.walk(tree))
    assert not any(isinstance(node, (ast.ClassDef, ast.FunctionDef))
                   and node.name in {"BackendRegistry", "BackendSpec", "_cpu_allocator"}
                   for node in ast.walk(tree))


def test_core_api_registers_and_calls_the_same_python_dispatch_table():
    tree = ast.parse((RUNTIME.parent / "_core/var.py").read_text())
    imports = {alias.name: alias.asname or alias.name
               for node in tree.body if isinstance(node, ast.ImportFrom)
               and node.module == "_runtime.dispatch" for alias in node.names}
    register = imports["register_kernel"]
    dispatch = imports["try_dispatch"]
    registrations = {
        node.value.args[0].s: node.value.args[1].s
        for node in tree.body
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id == register
        and len(node.value.args) >= 2
        and all(isinstance(arg, ast.Str) for arg in node.value.args[:2])
    }
    for name in ("outer", "clamp", "flatten"):
        assert registrations[name] == "*"
        function = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == name)
        assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                   and node.func.id == dispatch and node.args
                   and isinstance(node.args[0], ast.Str) and node.args[0].s == name
                   for node in ast.walk(function))


def test_native_cpu_registry_dispatch_matches_outer_and_clamp_values():
    import numpy as np
    import jittor as jt
    from jittor._runtime.dispatch import registered_kernel, select_kernel

    with jt.flag_scope(use_cuda=0):
        x = jt.array([1, 2, 3])
        y = jt.array([4, 5])
        selected = select_kernel("outer", x, y)
        assert selected is not None
        assert selected is registered_kernel("outer", "*")
        expected = np.outer([1, 2, 3], [4, 5])
        np.testing.assert_array_equal(jt.outer(x, y).numpy(), expected)
        np.testing.assert_array_equal(selected(x, y).numpy(), expected)

        value = jt.array([-2.0, 0.5, 3.0])
        selected = select_kernel("clamp", value, 0.0, 1.0)
        assert selected is not None
        assert selected is registered_kernel("clamp", "*")
        np.testing.assert_allclose(jt.clamp(value, 0.0, 1.0).numpy(), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(selected(value, 0.0, 1.0).numpy(), [0.0, 0.5, 1.0])


def test_native_cpu_registry_dispatches_flatten_and_reports_it():
    import numpy as np
    import jittor as jt
    from jittor._runtime.dispatch import registered_kernel, select_kernel

    with jt.flag_scope(use_cuda=0):
        value = jt.array([[1, 2], [3, 4]])
        selected = select_kernel("flatten", value, 0, -1)
        assert selected is not None
        assert selected is registered_kernel("flatten", "*")
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(jt.flatten(value).numpy(), expected)
        np.testing.assert_array_equal(jt.flatten(value, 0, 1).numpy(), expected)
        np.testing.assert_array_equal(selected(value, 0, -1).numpy(), expected)
