"""Real MMEngine training owners and MMCV registry-built convolution blocks.

These tiny user models expose BaseModel's unmodified ``train_step``. Compiled
MMCV operators are outside these cases; no extension module is substituted.
"""

PARAMETERS_BY_CASE = {
    "mmengine_train_step": (
        "fc1.bias",
        "fc1.weight",
        "fc2.bias",
        "fc2.weight",
        "norm.bias",
        "norm.weight",
    ),
    "mmcv_convmodule_train_step": (
        "block1.bn.bias",
        "block1.bn.weight",
        "block1.conv.weight",
        "block2.conv.weight",
        "block2.gn.bias",
        "block2.gn.weight",
        "head.bias",
        "head.weight",
    ),
}
BUFFER_NAMES_BY_CASE = {
    "mmengine_train_step": (),
    "mmcv_convmodule_train_step": (
        "block1.bn.num_batches_tracked",
        "block1.bn.running_mean",
        "block1.bn.running_var",
    ),
}
REQUIRED_GRADIENTS = {
    case: tuple("grad::" + name for name in names) + ("ingrad::sample",)
    for case, names in PARAMETERS_BY_CASE.items()
}


def _model(torch, use_mmcv):
    from mmengine.model import BaseModel
    from torch import nn

    class Regressor(BaseModel):
        def __init__(self):
            super().__init__()
            self.loss_forward_calls = 0
            if use_mmcv:
                import mmcv.cnn  # noqa: F401 -- registers the real ConvModule
                from mmengine.registry import MODELS

                self.block1 = MODELS.build(
                    dict(
                        type="ConvModule",
                        in_channels=3,
                        out_channels=4,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type="BN"),
                        act_cfg=dict(type="ReLU"),
                    )
                )
                self.block2 = MODELS.build(
                    dict(
                        type="ConvModule",
                        in_channels=4,
                        out_channels=4,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type="GN", num_groups=2),
                        act_cfg=dict(type="LeakyReLU"),
                    )
                )
                self.head = nn.Linear(4, 2)
            else:
                self.norm = nn.LayerNorm(6)
                self.fc1 = nn.Linear(6, 8)
                self.fc2 = nn.Linear(8, 2)

        def forward(self, sample, targets=None, mode="tensor"):
            if use_mmcv:
                prediction = self.head(self.block2(self.block1(sample)).mean(dim=(2, 3)))
            else:
                prediction = self.fc2(nn.functional.silu(self.fc1(self.norm(sample))))
            if mode == "loss":
                self.loss_forward_calls += 1
                self.last_prediction = prediction
                return dict(
                    loss_regression=(prediction - targets).square().mean(),
                    loss_regularization=[prediction.square().mean() * 0.01],
                )
            return prediction

    shape = (2, 3, 8, 8) if use_mmcv else (4, 6)
    return Regressor(), {"sample": ("float32", shape, None)}


def mmengine_train_step(torch):
    return _model(torch, False)


def mmcv_convmodule_train_step(torch):
    return _model(torch, True)
