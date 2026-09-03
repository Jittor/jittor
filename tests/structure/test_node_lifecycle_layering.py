from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
NODE_HEADER = REPO_ROOT / "python" / "jittor" / "src" / "node.h"


def test_node_header_does_not_depend_on_python_tracing():
    source = NODE_HEADER.read_text(encoding="utf-8")

    assert 'pybind/py_var_tracer.h' not in source
    assert "NodeLifecycleObserver" in source
    assert "notify_node_created" in source
    assert "notify_node_destroyed" in source
