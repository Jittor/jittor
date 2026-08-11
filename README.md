# jittor-gs

Optional graphdeco Gaussian Splatting integration for Jittor. Importing the
package is side-effect free. Jittor activates it through the
`jittor.module_patches` entry-point group, or an application can call
`jittor_gs.install()` explicitly.

The adapter owns two project-level runtime patches:

- `scene.gaussian_model.GaussianModel.save_ply` batches device synchronization
  before writing PLY data. It falls back to the project's original method if the
  optimized path is unavailable.
- `lpipsPyTorch.lpips` reuses one LPIPS criterion per network, version, and
  device instead of constructing it for every call.

It also marks known read-only calls in `diff_gaussian_rasterization._C`,
`fused_ssim_cuda`, and `simple_knn._C`. The policy is deliberately limited to
`rasterize_gaussians`, its backward operation, visibility checks, fused SSIM and
its backward operation, and `distCUDA2`; no blanket extension borrowing is
enabled.

Set `JITTOR_GS_RUNTIME_PATCHES=0` to disable both runtime patches, or
`JITTOR_GS_BATCH_SAVE_PLY=0` to retain only the original `save_ply` behavior.
The `ply` extra installs `plyfile` for the batched writer:

```sh
python -m pip install 'jittor-gs[ply]'
```

`scripts/run_gaussian_splatting.sh` and `scripts/check_gaussian_splatting.py`
are launch and validation helpers for a graphdeco checkout. This package does
not provide TRELLIS, Transformers, FSDP, or general torch compatibility.
