import types

from jittor_adapters.vllm import model_loader


def test_eager_model_loader_does_not_request_meta_storage():
    calls = []
    module = types.ModuleType("vllm.model_executor.model_loader.utils")
    module.record_metadata_for_reloading = lambda model: calls.append(model)

    assert model_loader.skip_layerwise_reload_metadata(module)
    marker = object()
    assert module.record_metadata_for_reloading(marker) is None
    assert calls == []
    assert not model_loader.skip_layerwise_reload_metadata(module)
