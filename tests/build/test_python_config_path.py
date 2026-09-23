"""Python build helpers must follow the interpreter used by the environment."""

import sys
import sysconfig

import jittor_utils


def test_uv_base_interpreter_config_helper_is_discovered(monkeypatch):
    # The version has to come from the running interpreter, not a literal:
    # `get_py3_config_path` builds the name as `python3.{minor}-config`, so a
    # hardcoded "3.11" makes the patched `isfile` answer for a file the code
    # never asks about. It looked for `cpython-3.11/bin/python3.14-config`
    # here and the whole test only passed on Python 3.11.
    tag = "3.%d" % sys.version_info.minor
    expected = "/opt/uv/python/cpython-%s/bin/python%s-config" % (tag, tag)

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
