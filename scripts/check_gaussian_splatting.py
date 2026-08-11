import os
import sys

import torch


def main():
    import jittor as jt
    from simple_knn import _C as simple_knn_c
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
    from fused_ssim import fused_ssim

    print("python", sys.executable)
    print("torch", getattr(torch, "__version__", None), getattr(torch, "__file__", None))
    print("jittor", jt.__version__, jt.__file__)
    print("cache", jt.flags.cache_path)
    print("use_cuda", jt.flags.use_cuda)
    print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES"))

    pts = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float32,
        device="cuda",
    )
    dist = simple_knn_c.distCUDA2(pts)
    print("distCUDA2", dist.shape, dist.numpy().tolist())

    means3D = torch.zeros((1, 3), dtype=torch.float32, device="cuda").requires_grad_(True)
    means2D = torch.zeros((1, 3), dtype=torch.float32, device="cuda", requires_grad=True)
    shs = torch.zeros((1, 16, 3), dtype=torch.float32, device="cuda")
    opacities = torch.ones((1, 1), dtype=torch.float32, device="cuda") * 0.5
    scales = torch.ones((1, 3), dtype=torch.float32, device="cuda") * 0.01
    rotations = torch.zeros((1, 4), dtype=torch.float32, device="cuda")
    rotations[:, 0] = 1.0
    bg = torch.zeros((3,), dtype=torch.float32, device="cuda")
    view = torch.eye(4, dtype=torch.float32, device="cuda")
    proj = torch.eye(4, dtype=torch.float32, device="cuda")
    campos = torch.zeros((3,), dtype=torch.float32, device="cuda")
    settings = GaussianRasterizationSettings(
        8, 8, 1.0, 1.0, bg, 1.0, view, proj, 3, campos, False, False, False
    )
    rasterizer = GaussianRasterizer(settings)
    image, radii, depth = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacities,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    loss = image.sum() + depth.sum()
    loss.backward()
    print("raster", image.shape, radii.shape, depth.shape, float(loss.item()))
    print("raster_grad", means3D.grad.shape if means3D.grad is not None else None)

    a = torch.rand((1, 3, 16, 16), dtype=torch.float32, device="cuda")
    b = a.clone()
    ssim = fused_ssim(a, b)
    print("fused_ssim", ssim.shape, float(ssim.item()))


if __name__ == "__main__":
    main()
