"""Declared Torch paths whose final binding receives fidelity metadata.

Family owners keep detailed records. This manifest covers standard aliases and
native-backed NN classes which are published by composition rather than a new
mathematical implementation. Native-only helpers copied into compatibility
namespaces are not silently counted as PyTorch APIs.
"""
from .fidelity import Fidelity, register_api_bindings


API_PATHS = {
    "torch": (
        "Function", "GradScaler", "Module", "Var", "Tag", "device", "dtype",
        "add", "argwhere", "clamp", "clamp_max", "clamp_min", "clip", "clone",
        "conv1d", "conv2d", "conv3d", "conv_transpose1d", "conv_transpose2d",
        "conv_transpose3d", "ne", "nonzero", "not_equal", "pixel_shuffle",
        "pixel_unshuffle", "scaled_dot_product_attention", "sum",
        "compat_approximate_apis", "compat_debug", "compat_swallowed",
        "compat_swallowed_counts", "compat_unimplemented_apis",
    ),
    "torch.Tensor": (
        "aminmax", "argsort", "broadcast_to", "clone", "copysign", "count_nonzero",
        "diag_embed", "diagonal", "exp2", "float_power", "heaviside", "index_copy_",
        "index_put_", "kron", "log10", "log_softmax", "logaddexp", "logcumsumexp",
        "logsumexp", "masked_select", "median", "nan_to_num", "nanmean", "nansum",
        "pdist", "ravel", "sign", "signbit", "softmax", "sort", "swapaxes",
        "swapdims", "topk", "trace", "trunc", "unflatten", "xlogy",
    ),
    "torch.nn": (
        "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
        "AdaptiveMaxPool2d", "AdaptiveMaxPool3d", "AvgPool1d", "AvgPool2d", "AvgPool3d",
        "BCELoss", "BCEWithLogitsLoss", "BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
        "Bilinear", "CTCLoss", "ConstantPad1d", "ConstantPad2d", "ConstantPad3d",
        "Conv1d", "Conv2d", "Conv3d", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d",
        "CrossEntropyLoss", "Dropout", "Dropout2d", "Dropout3d", "ELU", "Embedding",
        "EmbeddingBag", "Flatten", "Fold", "GELU", "GLU", "GRU", "GRUCell", "GroupNorm",
        "InstanceNorm1d", "InstanceNorm2d", "InstanceNorm3d", "KLDivLoss", "L1Loss",
        "LSTM", "LSTMCell", "LayerNorm", "LazyBatchNorm1d", "LazyBatchNorm2d",
        "LazyBatchNorm3d", "LeakyReLU", "Linear", "MSELoss", "MaxPool1d", "MaxPool2d",
        "MaxPool3d", "MaxUnpool2d", "MaxUnpool3d", "Mish", "Module", "ModuleList",
        "MultiheadAttention", "PReLU", "ParameterDict", "RNN", "RNNBase", "RNNCell",
        "RReLU", "ReLU", "ReLU6", "ReflectionPad2d", "ReplicationPad2d", "Sequential",
        "SiLU", "Sigmoid", "Softmax", "Tanh", "Unfold", "Upsample",
        "UpsamplingBilinear2d", "UpsamplingNearest2d", "ZeroPad2d",
    ),
    "torch.nn.functional": (
        "adaptive_avg_pool2d", "adaptive_max_pool2d", "affine_grid", "avg_pool2d",
        "avg_pool3d", "batch_norm", "bilinear", "binary_cross_entropy",
        "binary_cross_entropy_with_logits", "conv1d", "conv2d", "conv3d",
        "conv_transpose1d", "conv_transpose2d", "conv_transpose3d", "cosine_similarity",
        "cross_entropy", "dropout", "dropout2d", "elu", "embedding", "embedding_bag",
        "fold", "gelu", "glu", "grid_sample", "group_norm", "hardsigmoid", "hardswish",
        "hardtanh", "instance_norm", "interpolate", "kl_div", "l1_loss", "layer_norm",
        "leaky_relu", "linear", "logsigmoid", "log_softmax", "max_pool2d", "max_pool3d",
        "mish", "mse_loss", "multi_head_attention_forward", "nll_loss", "normalize",
        "one_hot", "pad", "pairwise_distance", "prelu", "relu", "relu6", "rrelu",
        "sigmoid", "silu", "smooth_l1_loss", "softmax", "softplus", "softsign",
        "unfold", "upsample", "upsample_bilinear",
    ),
    "torch.nn.Module": ("__call__",),
    "torch.nn.utils": ("skip_init",),
    "torch.nn.utils.rnn": ("PackedSequence", "pack_padded_sequence", "pad_packed_sequence", "pad_sequence"),
    "torch.nn.utils.parametrizations": ("weight_norm", "spectral_norm"),
    "torch.nn.init": ("calculate_gain", "eye_", "constant", "kaiming_normal", "kaiming_uniform",
                      "normal", "uniform", "xavier_normal", "xavier_uniform"),
    "torch.utils.data._utils.collate": ("default_collate",),
    "torch.optim": ("LBFGS",),
    "torch.autograd": ("Variable", "enable_grad", "no_grad", "once_differentiable"),
    "torch.autograd.Function": ("__call__", "save_for_backward", "saved_tensors", "set_materialize_grads"),
    "torch.autograd.functional": ("jvp", "vjp"),
    "torch.distributed": ("GroupMember", "group", "is_torchelastic_launched"),
}

UNIMPLEMENTED_PATHS = {
    "torch": ("ScriptModule", "SymBool", "SymFloat", "SymInt"),
    "torch.jit": ("ScriptModule",),
    "torch.nn.utils.parametrizations": ("orthogonal",),
}


def resolve(root, namespace):
    value = root
    for name in namespace.split(".")[1:]:
        value = getattr(value, name, None)
        if value is None:
            return None
    return value


def register_public_apis(context):
    for namespace, names in API_PATHS.items():
        owner = resolve(context.target_namespace, namespace)
        if owner is not None:
            register_api_bindings(owner, namespace, names, Fidelity.APPROXIMATE,
                "Uses the published frontend/native implementation; full Torch argument, dtype, backend and advanced-mode equivalence is not claimed")
    for namespace, names in UNIMPLEMENTED_PATHS.items():
        owner = resolve(context.target_namespace, namespace)
        if owner is not None:
            register_api_bindings(owner, namespace, names, Fidelity.UNIMPLEMENTED,
                "Concrete compatibility placeholder; symbolic values, TorchScript or the requested parametrization are not implemented")
