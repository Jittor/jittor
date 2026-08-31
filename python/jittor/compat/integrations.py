"""Independent optional runtime integrations and their structured report."""

from __future__ import absolute_import


def _warning(logger, message):
    if logger is not None:
        logger.w(message)


def apply_external_runtime_patches(logger=None):
    report = {}

    try:
        from jittor.compat import triton as _triton_compat  # noqa: F401

        report["triton_shim"] = {"ok": True}
    except Exception as error:
        report["triton_shim"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external runtime patch triton skipped: %s" % error)

    try:
        from jittor.compat import vllm as _vllm_compat

        report["vllm_shim"] = {"ok": True, "armed": bool(_vllm_compat.register())}
    except Exception as error:
        report["vllm_shim"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external runtime patch vllm skipped: %s" % error)

    try:
        from jittor.compat.module_patcher import install_module_patches

        patch_report = install_module_patches()
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
    except Exception as error:
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
    except Exception as error:
        report["external_backends"] = {
            "ok": False,
            "error": "%s: %s" % (type(error).__name__, error),
        }
        _warning(logger, "external backend registry failed: %s" % error)

    apply_external_runtime_patches.last_report = report
    return report


apply_external_runtime_patches.last_report = {}
