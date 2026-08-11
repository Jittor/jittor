"""Gaussian Splatting integration registration for Jittor."""

from __future__ import annotations

_READONLY_FUNCTIONS = {
    "diff_gaussian_rasterization._C": (
        "rasterize_gaussians",
        "rasterize_gaussians_backward",
        "mark_visible",
        "fusedssim",
        "fusedssim_backward",
    ),
    "fused_ssim_cuda": ("fusedssim", "fusedssim_backward"),
    "simple_knn._C": ("distCUDA2",),
}


def register_patches(register) -> None:
    from jittor.torch_shim.readonly_extensions import (
        register_readonly_extension_borrow,
    )

    from . import runtime

    runtime.register_patches(register)
    register_readonly_extension_borrow(
        registry=_READONLY_FUNCTIONS,
        register_patch=register,
    )


def install():
    from jittor.compat.module_patcher import (
        install_module_patches,
        register_module_patch,
    )

    register_patches(register_module_patch)
    return install_module_patches()


__all__ = ["install", "register_patches"]
