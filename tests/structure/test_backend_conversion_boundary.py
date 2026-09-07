"""4.12 contract: backend providers must not rewrite shared sources.

The selected providers and their native resources must share one owner.
"""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FORBIDDEN_CALLS = {
    "transform_sources",
    "process_jittor_source",
    "process_acl",
    "process_rocm",
}
NATIVE_PROVIDERS = {
    "acl": ROOT / "backends/acl/__init__.py",
    "rocm": ROOT / "backends/rocm/__init__.py",
    "corex": ROOT / "backends/corex/__init__.py",
}


def _calls(path):
    tree = ast.parse(path.read_text(encoding="utf8"), filename=str(path))
    return {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in FORBIDDEN_CALLS
    }


def _entry_points():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf8")
    section = text.split('[project.entry-points."jittor.backends"]', 1)[1]
    section = section.split("[", 1)[0]
    values = {}
    for line in section.splitlines():
        if "=" not in line:
            continue
        name, value = (part.strip().strip('"') for part in line.split("=", 1))
        values[name] = value
    return values


def test_native_provider_sources_are_complete_and_do_not_rewrite_sources():
    for name, path in NATIVE_PROVIDERS.items():
        assert path.is_file(), f"missing native {name} provider: {path}"
    assert (ROOT / "backends/acl/src/backend.cc").is_file()
    assert (ROOT / "backends/rocm/runtime/driver.cc").is_file()
    assert (ROOT / "backends/corex/runtime/corex_backend.cc").is_file()


def test_backend_entry_points_are_native_and_conversion_free():
    expected = {
        "acl": "jittor.backends.acl",
        "rocm": "jittor.backends.rocm",
        "corex": "jittor.backends.corex",
    }
    entries = _entry_points()
    assert {name: entries.get(name) for name in expected} == expected
    discovery = (ROOT / "python/jittor_utils/backend_discovery.py").read_text(
        encoding="utf8"
    )
    assert '"acl": "jittor.backends.acl"' in discovery
    assert '"corex": "jittor.backends.corex"' in discovery


def test_native_provider_configure_is_conversion_free():
    for name, path in NATIVE_PROVIDERS.items():
        functions = {
            node.name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf8")))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert "configure" in functions, f"missing native {name} configure()"
        assert not _calls(path), f"{name} provider still rewrites shared sources"


def test_legacy_converter_definitions_are_unreferenced():
    refs = []
    for path in (ROOT / "python/jittor").rglob("*.py"):
        if "/tests/" in str(path):
            continue
        text = path.read_text(encoding="utf8")
        if "transform_sources(" in text or "process_acl" in text or "process_rocm" in text:
            refs.append(str(path.relative_to(ROOT)))
    assert refs == [], "legacy converter references remain: " + ", ".join(refs)
