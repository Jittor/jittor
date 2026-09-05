"""Backend source mirrors must not compile obsolete copies after a move."""

import ast
import os
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]


def test_transformed_source_cache_archives_obsolete_native_paths(tmp_path, monkeypatch):
    path = ROOT / "python/jittor_utils/__init__.py"
    tree = ast.parse(path.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "process_jittor_source"]
    namespace = {"os": os, "LOG": SimpleNamespace(i=lambda *args: None)}
    exec(compile(tree, str(path), "exec"), namespace)
    source, cache = tmp_path / "source", tmp_path / "cache"
    original = source / "src/misc/helper.cc"
    original.parent.mkdir(parents=True)
    original.write_text("int helper() { return 1; }\n")
    cache.mkdir()
    class Config(SimpleNamespace):
        def evolve(self, **changes):
            return Config(**dict(vars(self), **changes))
    config = Config(jittor_path=str(source), cache_path=str(cache), cc_flags="")
    transform = namespace["process_jittor_source"]
    transformed = transform(config, "probe", lambda text, name, kwargs: text)
    assert config.jittor_path == str(source)
    cached = Path(transformed.jittor_path)
    assert (cached / "src/misc/helper.cc").is_file()

    moved = source / "src/runtime/helper.cc"
    moved.parent.mkdir()
    original.rename(moved)
    original.parent.rmdir()
    transform(config, "probe", lambda text, name, kwargs: text)
    assert (cached / "src/runtime/helper.cc").is_file()
    assert not (cached / "src/misc").exists()
    assert list(cached.glob("src/**/*.cc")) == [cached / "src/runtime/helper.cc"]
    archives = list(cache.glob("probe_source_stale_*/src/misc/helper.cc"))
    assert len(archives) == 1
    assert archives[0].read_text() == moved.read_text()

    transform(config, "probe", lambda text, name, kwargs: text)
    assert len(list(cache.glob("probe_source_stale_*"))) == 1
