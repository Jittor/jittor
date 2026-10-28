"""Model-loader boundaries unavailable in eager Jittor execution."""

from jittor.compat.transaction import set_attr


def skip_layerwise_reload_metadata(module):
    """Skip meta-device reload bookkeeping; eager loading owns real tensors.

    vLLM's layerwise reloader captures every parameter through ``.to('meta')``.
    Jittor has no meta storage, and eager model loading never invokes the
    deferred reloader. Leaving the call as an explicit no-op avoids pretending
    that a meta tensor or zero-copy restoration exists.
    """

    if getattr(module, "_jittor_eager_reload_metadata", False):
        return False

    def record_metadata_for_reloading(model):
        del model
        return None

    set_attr(module, "record_metadata_for_reloading", record_metadata_for_reloading)
    set_attr(module, "_jittor_eager_reload_metadata", True)
    return True


PATCHES = {
    "vllm.model_executor.model_loader.utils": skip_layerwise_reload_metadata,
}
