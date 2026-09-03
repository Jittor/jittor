from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _function_body(source, signature):
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:index]
    raise AssertionError("unterminated function: " + signature)


def test_var_holder_registration_does_not_execute_the_graph():
    source = (ROOT / "python/jittor/src/var_holder.cc").read_text()
    bodies = [_function_body(source, "void add_hold_vars(VarHolder* self)")]
    bodies.extend(_function_body(source, signature) for signature in (
        "VarHolder::VarHolder(Var* v)",
        "VarHolder::VarHolder(VarPtr&& v)",
        "VarHolder::VarHolder(PyObject* obj, NanoString dtype)",
    ))
    for forbidden in ("sync(", "run_sync(", "submit_pending(", "auto_flush("):
        assert all(forbidden not in body for body in bodies)


def test_python_var_conversion_is_the_submission_boundary():
    converter = (ROOT / "python/jittor/src/pyjt/py_converter.h").read_text()
    executor = (ROOT / "python/jittor/src/executor.h").read_text()
    assert "schedule_pending_from_python(" in converter
    assert "void submit_pending(Var* target, bool force=false);" in executor
    assert "flush_suspended" not in executor
