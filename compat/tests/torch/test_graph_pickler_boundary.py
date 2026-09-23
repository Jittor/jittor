"""Eager consumers may import FX pickling APIs but cannot serialize FX graphs."""

import importlib
import io

import pytest


def _api():
    return importlib.import_module("torch.fx._graph_pickler")


def test_graph_pickler_import_has_stable_identity_and_explicit_fidelity():
    module = _api()
    compiler = importlib.import_module("jittor.compat.torch.installers.compiler")
    fidelity = importlib.import_module("jittor.compat.torch.fidelity")
    assert importlib.import_module("torch.fx._graph_pickler") is module
    assert module.GraphPickler is compiler.GraphPickler
    assert module.Options is compiler.Options
    record = fidelity.fidelity_of("torch.fx._graph_pickler.GraphPickler")
    assert record.implementation is module.GraphPickler
    assert record.level is fidelity.Fidelity.UNIMPLEMENTED


def test_graph_pickler_options_keep_native_filter_fields():
    options = _api().Options()
    assert options.ops_filter("torch.ops.aten.add")
    assert not options.ops_filter("custom.operation")
    assert not options.node_metadata_key_filter("nn_module_stack")
    assert options.node_metadata_key_filter("tensor_meta")
    unfiltered = _api().Options(ops_filter=None, node_metadata_key_filter=None)
    assert unfiltered.ops_filter is None
    assert unfiltered.node_metadata_key_filter is None


@pytest.mark.parametrize("allow_stubs", ["0", "1"])
@pytest.mark.parametrize("operation", ["construct", "dumps", "loads", "reducer_override"])
def test_graph_pickling_never_silently_fabricates_artifacts(monkeypatch, allow_stubs, operation):
    monkeypatch.setenv("JITTOR_TORCH_ALLOW_STUB", allow_stubs)
    pickler = _api().GraphPickler
    with pytest.raises(NotImplementedError, match="FX graph serialization.*eager"):
        if operation == "construct":
            pickler(io.BytesIO())
        elif operation == "dumps":
            pickler.dumps({"graph": object()})
        elif operation == "loads":
            pickler.loads(b"not an FX artifact", fake_mode=None)
        else:
            pickler.reducer_override(object(), object())
