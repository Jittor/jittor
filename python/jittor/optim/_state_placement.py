"""Per-parameter placement for internal optimizer state."""
from contextlib import contextmanager



@contextmanager
def optimizer_state_scope(param):
    """Shadow ambient placement and restore it after creating state."""
    import jittor as jt
    from jittor._core.var import device_scope_like

    backend = int(param.placement_backend)
    if backend < 0:
        # Preserve the pre-existing FollowRuntime behavior; no Python-only
        # operation can safely shadow both the context and native TLS here.
        yield
        return
    index = max(int(param.device_id), 0)
    placement_token = jt.core._set_tensor_placement(backend, index)
    try:
        native_token = jt.core._push_native_tensor_placement(backend, index)
        try:
            result_type = getattr(type(param), "_frontend_result_type", type(param))
            frontend_token = jt.core._set_tensor_frontend_type(result_type)
            try:
                with device_scope_like(param):
                    yield
            finally:
                jt.core._reset_tensor_frontend_type(frontend_token)
        finally:
            jt.core._pop_native_tensor_placement(native_token)
    finally:
        jt.core._reset_tensor_placement(placement_token)
