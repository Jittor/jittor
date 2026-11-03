"""The real policy must audit a distributed session from controller metadata."""

import json
import os
from pathlib import Path
import subprocess

import pytest

from _helpers.child_process import PYTHON, child_env


_SOURCE_TESTS = Path(__file__).resolve().parents[1]


def _run_project(tmp_path, sources, extra=(), extra_env=None, native_paths=(), worker_mode=None):
    root = tmp_path / "project"
    tests = root / "tests"
    tests.mkdir(parents=True)
    (root / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (root / "conftest.py").write_text(
        "import json\nimport os\nfrom pathlib import Path\nimport pytest\n"
        "from _helpers import pytest_policy as policy\n"
        "ROOT = Path(__file__).resolve().parent\n"
        "policy.TEST_ROOT = ROOT / 'tests'\n"
        "policy.REPO_ROOT = ROOT\n"
        "policy.TEST_ROOTS = (ROOT / 'tests',)\n"
        "policy.NATIVE_MODE_PATHS = " + repr(tuple(native_paths)) + "\n"
        "pytest_plugins = ['_helpers.pytest_policy']\n"
        # Set only the collection-mode flag after policy bootstrap. These
        # synthetic tests contain no tensors and must not start a JIT build.
        "WORKER_MODE = " + repr(worker_mode) + "\n"
        "@pytest.hookimpl(tryfirst=True)\n"
        "def pytest_sessionstart(session):\n"
        " if hasattr(session.config, 'workerinput') and WORKER_MODE is not None:\n"
        "  os.environ['JITTOR_TORCH_SHIM'] = WORKER_MODE\n"
        "@pytest.hookimpl(trylast=True)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        " worker = getattr(session.config, 'workerinput', {}).get('workerid')\n"
        " if worker is not None:\n"
        "  data = {'exitstatus': int(session.exitstatus),"
        " 'seen_files': sorted(policy._FILE_OUTCOMES)}\n"
        "  (ROOT / (worker + '.json')).write_text(json.dumps(data))\n",
        encoding="utf-8",
    )
    for name, source in sources.items():
        (tests / name).write_text(source, encoding="utf-8")
    environment = {
        name: value for name, value in os.environ.items()
        if not name.startswith(("JT_", "JITTOR_", "PYTEST_"))
        and name not in ("PYTHONPATH", "PYTHONHOME", "REAL_TORCH_SITE")
    }
    environment.update(
        PYTHONPATH=str(_SOURCE_TESTS), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTEST_PLUGINS="xdist.plugin", JITTOR_TORCH_SHIM="0",
        JITTOR_TEST_REQUIRE_EXECUTION="1",
    )
    environment.update(extra_env or {})
    result = subprocess.run(
        [PYTHON, "-m", "pytest", "tests", "-q", "-ra", "-n", "2",
         "--dist=loadfile", "-p", "no:cacheprovider", *extra],
        cwd=root, env=child_env(environment, inherit=False), capture_output=True, text=True, timeout=60,
    )
    workers = [json.loads(path.read_text()) for path in sorted(root.glob("gw*.json"))]
    return result, workers


def test_distributed_passes_do_not_report_unassigned_files_as_empty(tmp_path):
    sources = {"test_" + name + ".py": "def test_first(): pass\ndef test_second(): pass\n"
               for name in ("a", "b")}
    result, workers = _run_project(tmp_path, sources)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "collected 0 tests" not in result.stdout
    assert "JITTOR-SESSION-COMPLETE collected=4 executed=4" in result.stdout
    assert len(workers) == 2
    assert all(worker["exitstatus"] == 0 for worker in workers), workers
    assert {path for worker in workers for path in worker["seen_files"]} == {
        "tests/test_a.py", "tests/test_b.py"}


def test_distributed_true_empty_file_fails_closed(tmp_path):
    result, _ = _run_project(tmp_path, {
        "test_pass.py": "def test_value(): pass\n",
        "test_empty.py": "class TestEmpty:\n    pass\n",
    })
    assert result.returncode == 1, result.stdout + result.stderr
    assert "tests/test_empty.py  collected 0 tests" in result.stdout
    assert "tests/test_pass.py  collected 0 tests" not in result.stdout


def test_distributed_unexplained_all_skip_fails_closed(tmp_path):
    result, _ = _run_project(tmp_path, {
        "test_skip.py": "import pytest\n@pytest.mark.skip(reason='unimplemented sentinel')\n"
                        "def test_value(): pass\n",
    })
    assert result.returncode == 1, result.stdout + result.stderr
    assert "unimplemented sentinel" in result.stdout


def test_distributed_deselection_and_hardware_skip_are_explained(tmp_path):
    result, workers = _run_project(tmp_path, {
        "test_pass.py": "def test_value(): pass\n",
        "test_slow.py": "import pytest\npytestmark=pytest.mark.slow\ndef test_value(): pass\n",
        "test_skip.py": "import pytest\n@pytest.mark.skip(reason='requires CUDA device')\n"
                        "def test_value(): pass\n",
    }, extra=("-m", "not slow"))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "collected 0 tests" not in result.stdout
    assert all(worker["exitstatus"] == 0 for worker in workers), workers


@pytest.mark.parametrize("reason,expected", [("requires CUDA device", 0), ("unimplemented sentinel", 1)])
def test_distributed_collection_skip_keeps_its_reason(tmp_path, reason, expected):
    result, _ = _run_project(tmp_path, {
        "test_pass.py": "def test_value(): pass\n",
        "test_skip.py": "import pytest\npytest.skip(" + repr(reason) + ", allow_module_level=True)\n"
                        "def test_value(): pass\n",
    })
    assert result.returncode == expected, result.stdout + result.stderr
    assert "collected 0 tests" not in result.stdout
    assert reason.lower() in result.stdout.lower()


_PROFILE_SKIPS = (
    ("JITTOR_ECOSYSTEM_LONGFORMER_32K", "opt-in Longformer 32768-token gate"),
    ("JITTOR_ECOSYSTEM_OPENMMLAB", "opt-in real OpenMMLab training gate"),
    ("JITTOR_ECOSYSTEM_OPENMMLAB_INSTALLED", "opt-in source-bound OpenMMLab training gate"),
    ("JITTOR_ECOSYSTEM_LARGE", "set JITTOR_ECOSYSTEM_LARGE=1 to run the realistic-size measurement"),
    ("JITTOR_ECOSYSTEM_SWIFT", "opt-in real Swift Trainer gate"),
    ("JITTOR_REQUIRE_VERL", "opt-in VERL NumPy2/Transformers5 component profile is not selected"),
    ("JITTOR_ECOSYSTEM_VERL_TRAINING", "opt-in source-bound VERL training gate"),
    ("JITTOR_ECOSYSTEM_VLLM", "opt-in vLLM component gate"),
)


def test_distributed_unselected_real_profiles_are_explained(tmp_path):
    sources = {"test_pass.py": "def test_value(): pass\n"}
    sources.update({"test_profile_" + str(index) + ".py":
                    "import pytest\ndef test_value(): pytest.skip(" + repr(reason) + ")\n"
                    for index, (_, reason) in enumerate(_PROFILE_SKIPS)})
    result, _ = _run_project(tmp_path, sources)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "other skipped: 0" in result.stdout


@pytest.mark.parametrize("switch,reason", _PROFILE_SKIPS)
def test_distributed_selected_profile_skip_fails_closed(tmp_path, switch, reason):
    result, _ = _run_project(tmp_path, {"test_selected.py":
        "import pytest\ndef test_value(): pytest.skip(" + repr(reason) + ")\n"},
        extra_env={switch: "1"})
    assert result.returncode == 1, result.stdout + result.stderr


@pytest.mark.parametrize("reason,environment,expected", [
    ("opt-in invented profile gate", {}, 1),
    ("missing diffusers", {}, 0),
    ("missing swift", {}, 0),
    ("missing diffusers", {"JITTOR_REQUIRE_DIFFUSERS": "1"}, 1),
    ("missing swift", {"JITTOR_REQUIRE_SWIFT": "1"}, 1),
    ("missing diffusers", {"JITTOR_REQUIRE_OPTIONAL_DEPS": "1"}, 1),
])
def test_distributed_dependency_and_unknown_profile_boundaries(tmp_path, reason, environment, expected):
    result, _ = _run_project(tmp_path, {"test_optional.py":
        "import pytest\ndef test_value(): pytest.skip(" + repr(reason) + ")\n"},
        extra_env=environment)
    assert result.returncode == expected, result.stdout + result.stderr


@pytest.mark.parametrize("torch_mode,expected_passed", [("1", 1), ("0", 2)],
                         ids=("torch-ignores-native", "native-executes-native"))
def test_distributed_other_mode_file_matches_collection(tmp_path, torch_mode, expected_passed):
    result, workers = _run_project(tmp_path, {
        "test_pass.py": "def test_value(): pass\n",
        "test_native.py": "def test_native(): pass\n",
    }, native_paths=("tests/test_native.py",), worker_mode=torch_mode)
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(expected_passed) + " passed" in result.stdout
    assert "collected 0 tests" not in result.stdout
    seen = {path for worker in workers for path in worker["seen_files"]}
    assert ("tests/test_native.py" in seen) == (torch_mode == "0")


@pytest.mark.parametrize("source,diagnostic", [
    ("class TestEmpty:\n    pass\n", "collected 0 tests"),
    ("import pytest\ndef test_value(): pytest.skip('unimplemented sentinel')\n", "unimplemented sentinel"),
], ids=("selected-empty", "unexplained-skip"))
def test_distributed_mode_filter_keeps_real_empty_and_skip_failures(tmp_path, source, diagnostic):
    result, _ = _run_project(tmp_path, {
        "test_pass.py": "def test_value(): pass\n",
        "test_native.py": "def test_native(): pass\n",
        "test_broken.py": source,
    }, native_paths=("tests/test_native.py",), worker_mode="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert diagnostic in result.stdout
    assert "tests/test_native.py  collected 0 tests" not in result.stdout


def test_distributed_explicit_other_mode_empty_file_still_fails_closed(tmp_path):
    result, _ = _run_project(tmp_path, {
        "test_pass.py": "def test_value(): pass\n",
        "test_native.py": "class TestEmpty:\n    pass\n",
    }, extra=("tests/test_native.py",),
        native_paths=("tests/test_native.py",), worker_mode="1")
    assert result.returncode == 1, result.stdout + result.stderr
    assert "tests/test_native.py  collected 0 tests" in result.stdout
