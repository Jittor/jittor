"""Public InstanceNorm module paths preserve class identity."""

import torch
import torch.nn.modules.instancenorm as instancenorm


def test_instancenorm_submodule_reexports_public_classes():
    for name in ("InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d"):
        assert getattr(instancenorm, name) is getattr(torch.nn, name)
