"""Stateful neural-network modules with explicit public exports."""

# ruff: noqa: F401

from jittor import Module
from jittor.misc import CTCLoss
from jittor.pool import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool3d,
    AdaptiveMaxPool2d,
    AdaptiveMaxPool3d,
    AvgPool1d,
    MaxPool1d,
    MaxPool2d,
    MaxPool3d,
    MaxUnpool2d,
    MaxUnpool3d,
    Pool,
    Pool3d,
)

from ..legacy_complex import ComplexNumber
from .attention import MultiheadAttention
from .activation import (
    ELU,
    GELU,
    GLU,
    Hardsigmoid,
    Hardswish,
    LeakyReLU,
    Leaky_relu,
    Mish,
    PReLU,
    RReLU,
    ReLU,
    ReLU6,
    Relu,
    SiLU,
    Sigmoid,
    Softmax,
    Softplus,
    Softsign,
    Tanh,
)
from .bilinear import Bilinear
from .container import Sequential
from .convolution import Conv, Conv1d
from .convolution3d import Conv3d
from .convolution_transpose import ConvTranspose, ConvTranspose3d
from .depthwise import DepthwiseConv
from .dropout import Dropout, Dropout2d, DropPath
from .embedding import Embedding, EmbeddingBag
from .fold import Fold, Unfold
from .linear import Conv1d_sp, Linear
from .loss import BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, KLDivLoss, L1Loss, MSELoss
from .normalization import (
    BatchNorm,
    BatchNorm1d,
    BatchNorm2d,
    BatchNorm3d,
    GroupNorm,
    InstanceNorm,
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LayerNorm,
    LayerNorm1d,
    LayerNorm2d,
    LayerNorm3d,
)
from .padding import (
    ConstantPad1d,
    ConstantPad2d,
    ConstantPad3d,
    ReflectionPad2d,
    ReplicationPad2d,
    ZeroPad2d,
)
from .parameter import Parameter, ParameterList
from .pooling import AdaptiveAvgPool2d, AvgPool2d, AvgPool3d
from .recurrent import GRU, LSTM, RNN
from .recurrent_base import RNNBase
from .recurrent_cells import GRUCell, LSTMCell, RNNCell
from .shape import Flatten, Identity, PixelShuffle
from .upsampling import Resize, Upsample, UpsamplingBilinear2d, UpsamplingNearest2d


Conv2d = Conv
ConvTranspose2d = ConvTranspose
ModuleList = Sequential
ParameterDict = ParameterList


__all__ = sorted(
    (
        "AdaptiveAvgPool1d",
        "AdaptiveAvgPool2d",
        "AdaptiveAvgPool3d",
        "AdaptiveMaxPool2d",
        "AdaptiveMaxPool3d",
        "AvgPool1d",
        "AvgPool2d",
        "AvgPool3d",
        "BCELoss",
        "BCEWithLogitsLoss",
        "BatchNorm",
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "Bilinear",
        "CTCLoss",
        "ComplexNumber",
        "ConstantPad1d",
        "ConstantPad2d",
        "ConstantPad3d",
        "Conv",
        "Conv1d",
        "Conv1d_sp",
        "Conv2d",
        "Conv3d",
        "ConvTranspose",
        "ConvTranspose2d",
        "ConvTranspose3d",
        "CrossEntropyLoss",
        "DepthwiseConv",
        "DropPath",
        "Dropout",
        "Dropout2d",
        "ELU",
        "Embedding",
        "EmbeddingBag",
        "Flatten",
        "Fold",
        "GELU",
        "GLU",
        "GRU",
        "GRUCell",
        "GroupNorm",
        "Hardsigmoid",
        "Hardswish",
        "Identity",
        "InstanceNorm",
        "InstanceNorm1d",
        "InstanceNorm2d",
        "InstanceNorm3d",
        "KLDivLoss",
        "L1Loss",
        "LSTM",
        "LSTMCell",
        "LayerNorm",
        "LayerNorm1d",
        "LayerNorm2d",
        "LayerNorm3d",
        "LeakyReLU",
        "Leaky_relu",
        "Linear",
        "MSELoss",
        "MaxPool1d",
        "MaxPool2d",
        "MaxPool3d",
        "MaxUnpool2d",
        "MaxUnpool3d",
        "Mish",
        "Module",
        "ModuleList",
        "MultiheadAttention",
        "PReLU",
        "Parameter",
        "ParameterDict",
        "ParameterList",
        "PixelShuffle",
        "Pool",
        "Pool3d",
        "RNN",
        "RNNBase",
        "RNNCell",
        "RReLU",
        "ReLU",
        "ReLU6",
        "ReflectionPad2d",
        "Relu",
        "ReplicationPad2d",
        "Resize",
        "Sequential",
        "SiLU",
        "Sigmoid",
        "Softmax",
        "Softplus",
        "Softsign",
        "Tanh",
        "Unfold",
        "Upsample",
        "UpsamplingBilinear2d",
        "UpsamplingNearest2d",
        "ZeroPad2d",
    )
)
