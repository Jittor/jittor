"""TRELLIS.2 integration registration for Jittor."""

from __future__ import annotations

import os
from pathlib import Path


_READONLY_FUNCTIONS = {
    "flex_gemm.kernels.cuda": (
        "hashmap_build_sparse_conv_out_coords",
        "expand_unique_build_sparse_conv_out_coords",
        "hashmap_build_sparse_conv_neighbour_map",
        "hashmap_lookup",
        "hashmap_lookup_3d",
        "z_order_decode",
        "hilbert_decode",
        "neighbor_map_post_process_for_masked_implicit_gemm_1_no_bwd",
        "neighbor_map_post_process_for_masked_implicit_gemm_1",
        "neighbor_map_post_process_for_masked_implicit_gemm_2",
    ),
    "o_voxel._C": (
        "hashmap_lookup_cuda",
        "hashmap_lookup_3d_cuda",
        "z_order_decode_cuda",
        "hilbert_decode_cuda",
        "rasterize_voxels_cuda",
    ),
    "cumesh._C": (
        "hashmap_lookup_cuda",
        "hashmap_lookup_3d_cuda",
        "get_sparse_voxel_grid_active_vertices",
        "simple_dual_contour",
    ),
}

_READONLY_ARGUMENTS = {
    "flex_gemm.kernels.cuda": {
        "hashmap_insert": (2, 3),
        "hashmap_insert_3d": (2, 3),
        "hashmap_insert_3d_idx_as_val": (2,),
        "z_order_encode": (0,),
        "hilbert_encode": (0,),
    },
    "o_voxel._C": {
        "hashmap_insert_cuda": (2, 3),
        "hashmap_insert_3d_cuda": (2, 3),
        "hashmap_insert_3d_idx_as_val_cuda": (2,),
        "z_order_encode_cuda": (0, 1, 2),
        "hilbert_encode_cuda": (0, 1, 2),
    },
    "cumesh._C": {
        "hashmap_insert_cuda": (2, 3),
        "hashmap_insert_3d_cuda": (2, 3),
        "hashmap_insert_3d_idx_as_val_cuda": (2,),
    },
}

_SCRATCH_BORROW_FUNCTIONS = {
    "flex_gemm.kernels.cuda": (
        "hashmap_build_submanifold_conv_neighbour_map",
    )
}

_COPY_SCOPE_FUNCTIONS = {
    "o_voxel.postprocess": ("to_glb",),
    "nvdiffrast.torch.ops": (
        "rasterize",
        "interpolate",
        "texture",
        "texture_construct_mip",
        "antialias",
        "antialias_construct_topology_hash",
    ),
}

_FLASH_ATTN_RELATIVE_DIRS = (
    "flashattn_jittor",
    "flash_attn_jittor",
    "flash-attention-jittor",
    "flash-attention",
    "third_party/flashattn_jittor",
    "third_party/flash_attn_jittor",
    "third_party/flash-attention-jittor",
    "third_party/flash-attention",
    "extensions/flashattn_jittor",
    "extensions/flash_attn_jittor",
    "extensions/flash-attention",
)


def configure_runtime_environment() -> str:
    """Configure a project-scoped FlexGEMM autotune cache if unset."""
    configured = os.environ.get("FLEX_GEMM_AUTOTUNE_CACHE_PATH")
    if configured:
        path = Path(configured).expanduser()
    else:
        runtime_root = os.environ.get("JITTOR_TORCH_RUNTIME_ROOT")
        project_root = os.environ.get("TRELLIS2_ROOT") or os.environ.get(
            "TRELLIS_ROOT"
        )
        if project_root:
            path = (
                Path(project_root).expanduser()
                / ".cache"
                / "jittor_trellis"
                / "flex_gemm"
                / "autotune_cache.json"
            )
        elif runtime_root:
            path = Path(runtime_root).expanduser() / "flex_gemm" / "autotune_cache.json"
        else:
            cache_root = Path(
                os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
            ).expanduser()
            path = cache_root / "jittor-trellis" / "flex_gemm" / "autotune_cache.json"
        os.environ["FLEX_GEMM_AUTOTUNE_CACHE_PATH"] = os.fspath(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return os.fspath(path)


def register_patches(register) -> None:
    """Register TRELLIS runtime, dependency, and extension-boundary patches."""
    configure_runtime_environment()
    from . import patches, runtime

    runtime.register_patches(register)
    patches.register_patches(register)

    from jittor.compat.shim.extensions.readonly import (
        register_readonly_extension_borrow,
    )

    register_readonly_extension_borrow(
        registry=_READONLY_FUNCTIONS,
        copy_scope_registry=_COPY_SCOPE_FUNCTIONS,
        scratch_borrow_registry=_SCRATCH_BORROW_FUNCTIONS,
        readonly_arg_registry=_READONLY_ARGUMENTS,
        register_patch=register,
    )


def register_backends() -> None:
    """Extend Jittor's generic flash-attn discovery with TRELLIS roots."""
    configure_runtime_environment()
    from jittor.compat.external_backend import register_external_backend_hint

    register_external_backend_hint(
        "flash-attn",
        project_root_envs=("TRELLIS2_ROOT", "TRELLIS_ROOT"),
        relative_source_dirs=_FLASH_ATTN_RELATIVE_DIRS,
        environment_names=("TRELLIS2_ROOT", "TRELLIS_ROOT"),
    )


def install():
    """Activate this adapter explicitly instead of through entry points."""
    from jittor.compat.module_patcher import (
        install_module_patches,
        register_module_patch,
    )

    register_patches(register_module_patch)
    register_backends()
    return install_module_patches()


__all__ = [
    "configure_runtime_environment",
    "install",
    "register_backends",
    "register_patches",
]
