import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


class _CommandFailed(Exception):
    def __init__(self, reason, *, return_code=None):
        super().__init__(reason)
        self.return_code = return_code


def _load_restart_helper():
    tree = ast.parse((REPO_ROOT / "noxfile.py").read_text(encoding="utf-8"))
    names = {
        "_JIT_UTILS_UPDATED_EXIT_CODE",
        "_JIT_UTILS_RESTART_ATTEMPTS",
        "_run_with_jit_utils_restart",
    }
    selected = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    namespace = {"nox": SimpleNamespace(command=SimpleNamespace(CommandFailed=_CommandFailed))}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "noxfile.py", "exec"), namespace)
    return namespace["_run_with_jit_utils_restart"]


class _Session:
    def __init__(self, codes):
        self.codes = codes
        self.calls = []
        self.logs = []
        self.errors = []

    def run(self, *args, **kwargs):
        code = self.codes[len(self.calls)]
        self.calls.append((args, kwargs))
        if code == 0:
            return "actual command completed"
        error = _CommandFailed("Returned code 3: jit_utils was rebuilt", return_code=code)
        self.errors.append(error)
        raise error

    def log(self, message):
        self.logs.append(message)


@pytest.mark.parametrize(
    "codes, expected_calls, expected_code",
    [
        ([3, 0], 2, 0),
        ([3, 3, 3, 0], 3, 3),
        ([1, 0], 1, 1),
        ([None, 0], 1, None),
        ([-9, 0], 1, -9),
    ],
    ids=[
        "restart-then-success",
        "bounded-restart-failure",
        "ordinary-failure",
        "missing-code",
        "signal",
    ],
)
def test_restart_uses_only_exact_code_and_preserves_command(codes, expected_calls, expected_code):
    run = _load_restart_helper()
    session = _Session(codes)
    args = ("python", "-m", "jittor.selftest")
    env = {"JITTOR_HOME": "/isolated/cache", "PYTHONPATH": "/installed/wheel", "use_cuda": "0"}
    if expected_code == 0:
        assert run(session, *args, env=env) == "actual command completed"
    else:
        with pytest.raises(_CommandFailed) as caught:
            run(session, *args, env=env)
        assert caught.value is session.errors[-1]
        assert caught.value.return_code == expected_code
    assert len(session.calls) == expected_calls
    assert len(session.logs) == expected_calls - 1
    assert all(call_args == args and kwargs["env"] is env for call_args, kwargs in session.calls)
    assert all("success_codes" not in kwargs for _args, kwargs in session.calls)


def test_restart_code_matches_compiler_without_importing_jittor():
    compiler = REPO_ROOT / "python" / "jittor" / "build" / "compiler.py"
    values = {}
    for node in ast.parse(compiler.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "JIT_UTILS_UPDATED_EXIT_CODE":
                    values[target.id] = ast.literal_eval(node.value)
    assert values == {"JIT_UTILS_UPDATED_EXIT_CODE": 3}


@pytest.mark.parametrize(
    "name", ["_install_docs_wheel", "packaging", "_upper_python_compatibility", "smoke", "cpu"]
)
def test_jit_runtime_callers_use_the_restart_boundary(name):
    tree = ast.parse((REPO_ROOT / "noxfile.py").read_text(encoding="utf-8"))
    caller = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    calls = [
        node
        for node in ast.walk(caller)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_with_jit_utils_restart"
    ]
    assert calls
    if name in {"smoke", "cpu"}:
        assert any(
            any(isinstance(arg, ast.Name) and arg.id == "_CPU_PROBE" for arg in call.args)
            for call in calls
        )


def test_structure_preflights_native_jit_before_pytest_only_for_default_gate():
    tree = ast.parse((REPO_ROOT / "noxfile.py").read_text(encoding="utf-8"))
    structure = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "structure"
    )
    structure.decorator_list = []
    events = []
    dependencies = {
        name: name
        for name in (
            "PYTEST",
            "PYTEST_TIMEOUT",
            "SCIPY",
            "PYTEST_XDIST",
            "SETUPTOOLS",
            "JUPYTEXT",
            "NBFORMAT",
        )
    }
    dependencies.update(
        {
            "REPO_ROOT": REPO_ROOT,
            "STRUCTURE_TESTS": ("tests/structure",),
            "NATIVE_MODE_PATHS": ("tests/structure/backends/acl/test_acl_dtype_preservation.py",),
            "_session_env": lambda _session, _name: (Path("/isolated"), {}),
            "_install_compat_source": lambda _session, _env: events.append("install-source"),
            "_mode_env": lambda env, _paths: env,
            "_adapter_source_test_env": lambda env, _paths: env,
            "_run_with_jit_utils_restart": lambda _session, *args, **kwargs: events.append(
                ("preflight", args, kwargs)
            ),
        }
    )
    exec(compile(ast.Module(body=[structure], type_ignores=[]), "noxfile.py", "exec"), dependencies)

    class Session:
        def __init__(self, posargs):
            self.posargs = posargs

        def install(self, *_args):
            pass

        def run(self, *args, **kwargs):
            events.append(("run", args, kwargs))

    dependencies["structure"](Session(()))
    preflight = next(
        i for i, item in enumerate(events) if isinstance(item, tuple) and item[0] == "preflight"
    )
    native = next(
        i
        for i, item in enumerate(events)
        if isinstance(item, tuple) and item[0] == "run" and "pytest" in item[1]
    )
    assert preflight < native
    assert events[preflight][2]["env"]["JITTOR_TORCH_SHIM"] == "0"
    assert events[native][2]["env"]["JITTOR_TORCH_SHIM"] == "0"
    assert events[preflight][1][:2] == ("python", "-c")
    assert str(REPO_ROOT / "python" / "jittor") in events[preflight][1][2]
    compile(events[preflight][1][2], "<structure-preflight>", "exec")

    events.clear()
    dependencies["structure"](Session(("tests/structure/test_nox_jit_restart.py",)))
    assert not any(isinstance(item, tuple) and item[0] == "preflight" for item in events)
