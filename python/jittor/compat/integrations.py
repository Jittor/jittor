"""Independent optional runtime integrations and their structured report."""

from __future__ import absolute_import
from .diagnostics import EXPECTED, swallowed


def _warning(logger, message):
    if logger is not None:
        logger.w(message)


def apply_external_runtime_patches(logger=None):
    report = {}
    import jittor as jt

    try:
        from jittor.compat import triton as _triton_compat  # noqa: F401

        report["triton_shim"] = {"ok": True}
    except EXPECTED as error:
        swallowed("integrations.py apply_external_runtime_patches: from jittor.compat import triton as _triton_compat # no...", error)
        report["triton_shim"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external runtime patch triton skipped: %s" % error)

    try:
        from jittor.compat import vllm as _vllm_compat

        report["vllm_shim"] = {"ok": True, "armed": bool(_vllm_compat.register())}
    except EXPECTED as error:
        swallowed("integrations.py apply_external_runtime_patches: from jittor.compat import vllm as _vllm_compat", error)
        report["vllm_shim"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external runtime patch vllm skipped: %s" % error)

    try:
        from jittor.compat.module_patcher import install_module_patches

        transaction = getattr(
            getattr(jt, "_torch_compat_install_context", None),
            "state", {}).get("_install_transaction")
        patch_report = install_module_patches(transaction=transaction)
        results = [
            {
                "kind": item.kind,
                "name": item.name,
                "callback": item.callback,
                "status": item.status,
                "detail": item.detail,
            }
            for item in patch_report.results
        ]
        report["module_patches"] = {"ok": patch_report.ok, "results": results}
        for item in patch_report.failures:
            _warning(
                logger,
                "external module patch %s (%s) failed: %s"
                % (item.name, item.callback, item.detail or "unknown error"),
            )
    except EXPECTED as error:
        swallowed("integrations.py apply_external_runtime_patches: from jittor.compat.module_patcher import install_module...", error)
        report["module_patches"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external module patch registry failed: %s" % error)

    try:
        from jittor.compat.external_backend import load_external_backend_entry_points

        backend_results = load_external_backend_entry_points()
        results = [
            {
                "name": item.name,
                "value": item.value,
                "status": item.status,
                "detail": item.detail,
            }
            for item in backend_results
        ]
        failures = [item for item in backend_results if item.status == "failed"]
        report["external_backends"] = {"ok": not failures, "results": results}
        for item in failures:
            _warning(
                logger,
                "external backend %s (%s) failed: %s"
                % (item.name, item.value, item.detail or "unknown error"),
            )
    except EXPECTED as error:
        swallowed("integrations.py apply_external_runtime_patches: from jittor.compat.external_backend import load_externa...", error)
        report["external_backends"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external backend registry failed: %s" % error)

    apply_external_runtime_patches.last_report = report
    return report


apply_external_runtime_patches.last_report = {}


# ---------------------------------------------------------------------------
# Custom-operator replacements for specific downstream libraries.
#
# torch.library.custom_op is a generic registration API and must not know any
# model's operator names; it used to carry a hard-coded branch for
# "transformers::grouped_mm_fallback" that discarded the caller's own
# implementation.  The knowledge lives here instead, where library-specific
# adaptation belongs, and torch.library looks it up by name.
# ---------------------------------------------------------------------------

def _transformers_grouped_mm_fallback(input, weight, offsets, *args, **kwargs):
    """Grouped matmul over row ranges delimited by `offsets`.

    transformers registers this op as a fallback for a fused CUDA kernel that
    Jittor does not provide; its own body is written against torch primitives
    that do not survive the shim, so the adaptation is done here.
    """
    import jittor as jt

    output = jt.zeros((input.shape[0], weight.shape[2]), dtype=input.dtype)
    values = offsets.numpy().tolist() if hasattr(offsets, "numpy") else list(offsets)
    start = 0
    for index, end in enumerate(values):
        end = int(end)
        if end > start:
            output[start:end] = jt.matmul(input[start:end], weight[index])
        start = end
    return output


_CUSTOM_OP_OVERRIDES = {
    "transformers::grouped_mm_fallback": _transformers_grouped_mm_fallback,
}


def custom_op_overrides():
    """{"namespace::op": implementation} that replace a library's own version."""
    return dict(_CUSTOM_OP_OVERRIDES)
