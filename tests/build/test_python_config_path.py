"""Python build helpers must follow the interpreter used by the environment."""

import sysconfig

import jittor_utils


def test_uv_base_interpreter_config_helper_is_discovered(monkeypatch):
    expected = "/opt/uv/python/cpython-3.11/bin/python3.11-config"

    monkeypatch.delenv("JT_BUILD_PYTHON_CONFIG_PATH", raising=False)
    monkeypatch.delenv("python_config_path", raising=False)
    monkeypatch.setattr(
        sysconfig,
        "get_config_var",
        lambda name: expected.rsplit("/", 1)[0] if name == "BINDIR" else None,
    )
    monkeypatch.setattr(
        jittor_utils.os.path,
        "isfile",
        lambda path: path == expected,
    )
    monkeypatch.setattr(jittor_utils, "_py3_config_path", None)

    assert jittor_utils.get_py3_config_path() == expected
