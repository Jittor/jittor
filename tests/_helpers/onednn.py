"""Explicit oneDNN initialization with fail-closed capability handling."""
from jittor._runtime.backend_libraries import get_library_ops
from _helpers.capability import check_library, require_library


def onednn_ops(required=False):
    capability = require_library("mkl") if required else check_library("mkl", load=True)
    if capability.unprobed:
        raise AssertionError("oneDNN remained unprobed after explicit initialization")
    if not capability.enabled:
        return None
    ops = get_library_ops("mkl")
    if ops is None:
        raise AssertionError("enabled oneDNN library has no operator module")
    return ops


def requires_onednn():
    return onednn_ops(required=True)
