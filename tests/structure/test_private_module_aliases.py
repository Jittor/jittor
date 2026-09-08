"""Legacy private imports stay available without expanding the NN facade."""

import importlib.util
from pathlib import Path
import pickle
import sys
from types import ModuleType


def test_private_nn_alias_keeps_import_and_pickle_identity_without_facade_export(monkeypatch):
    source = Path(__file__).resolve().parents[2] / "python/jittor/_runtime/import_aliases.py"
    spec = importlib.util.spec_from_file_location("offline_backend_aliases", source)
    aliases = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aliases)
    root = ModuleType("jittor")
    root.__path__ = []
    facade = ModuleType("jittor.nn")
    facade.__path__ = []
    root.nn = facade
    legacy = "jittor.nn._cuda_inference"
    canonical_name = aliases.ALIASES[legacy]
    canonical = ModuleType(canonical_name)
    public_legacy = "jittor.nn.rms_norm_cuda"
    public = ModuleType(aliases.ALIASES[public_legacy])

    def cached_source(template, params):
        return template % params

    canonical.cached_source = cached_source
    for name, module in (("jittor", root), ("jittor.nn", facade),
                         (canonical_name, canonical), (public.__name__, public)):
        monkeypatch.setitem(sys.modules, name, module)
    # Track restoration even when publication creates or replaces an alias.
    monkeypatch.setitem(sys.modules, legacy, canonical)
    monkeypatch.setitem(sys.modules, public_legacy, public)
    aliases.publish_loaded_aliases(root)
    assert sys.modules[legacy] is canonical
    assert legacy.rsplit(".", 1)[1] not in vars(facade)
    assert facade.rms_norm_cuda is public
    assert aliases.import_alias(legacy) is canonical
    assert importlib.import_module(legacy) is canonical
    assert pickle.loads(b"cjittor.nn._cuda_inference\ncached_source\n.") is cached_source
    assert "_cuda_inference" not in vars(facade)
