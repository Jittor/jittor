"""TRELLIS.2-specific dependency and sparse-backend patches."""

from __future__ import annotations

import inspect
import math
import os
import sys
import types


_TRUTHY = {"1", "true", "yes", "on", "jittor"}
FLEXGEMM_BRIDGE_CONV_CALLS = 0


def _jittor_sparse_backend_enabled() -> bool:
    value = os.environ.get("JITTOR_TRELLIS_SPARSE_BACKEND", "")
    return value.strip().lower() in _TRUTHY


def _patch_flexgemm_triton_autotuner(module) -> bool:
    """Tolerate FlexGEMM's newer Autotuner arguments on Triton 3.1."""
    autotuner = getattr(module, "Autotuner", None)
    original = getattr(autotuner, "__init__", None)
    if original is None or getattr(original, "_jittor_trellis_tolerant", False):
        return False

    signature = inspect.signature(original)
    parameters = list(signature.parameters.values())[1:]
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepted = {
        parameter.name
        for parameter in parameters
        if parameter.kind != parameter.VAR_POSITIONAL
    }
    names = [parameter.name for parameter in positional]
    key_position = names.index("key") if "key" in names else None
    bench_position = names.index("do_bench") if "do_bench" in names else None

    def bridge_benchmark():
        try:
            from jittor.triton_shim import backend

            return backend.make_do_bench()
        except Exception:
            return None

    def tolerant_init(self, *args, **kwargs):
        key_value = kwargs.get("key")
        if key_value is None and key_position is not None and key_position < len(args):
            key_value = args[key_position]
        args = list(args[: len(positional)])
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
        benchmark = bridge_benchmark()
        if benchmark is not None and bench_position is not None:
            if bench_position < len(args):
                if args[bench_position] is None:
                    args[bench_position] = benchmark
            elif kwargs.get("do_bench") is None:
                kwargs["do_bench"] = benchmark
        original(self, *args, **kwargs)
        if not hasattr(self, "keys"):
            self.keys = list(key_value) if key_value is not None else []

    tolerant_init._jittor_trellis_tolerant = True
    autotuner.__init__ = tolerant_init
    return True


def _patch_dinov3_encoder_layer_layout(module) -> bool:
    """Adapt TRELLIS' DINOv3 extractor across Transformers layouts."""
    extractor = getattr(module, "DinoV3FeatureExtractor", None)
    if extractor is None:
        return False
    changed = False
    local_path = os.environ.get("TRELLIS_DINOV3_PATH")
    if (
        local_path
        and os.path.isdir(local_path)
        and not getattr(extractor.__init__, "_jittor_trellis_local_redirect", False)
    ):
        from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTModel
        from torchvision import transforms

        def local_init(self, model_name, image_size=512, _path=local_path):
            del model_name
            self.model_name = _path
            self.model = DINOv3ViTModel.from_pretrained(_path)
            self.model.eval()
            self.image_size = image_size
            self.transform = transforms.Compose(
                [
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                ]
            )

        local_init._jittor_trellis_local_redirect = True
        extractor.__init__ = local_init
        changed = True

    if getattr(
        getattr(extractor, "extract_features", None),
        "_jittor_trellis_layer_compat",
        False,
    ):
        return changed
    import torch.nn.functional as functional

    def extract_features(self, image):
        model = self.model
        image = image.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden_states = model.embeddings(image, bool_masked_pos=None)
        position_embeddings = model.rope_embeddings(image)
        if hasattr(model, "layer"):
            layers = model.layer
        elif hasattr(model, "model") and hasattr(model.model, "layer"):
            layers = model.model.layer
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
            layers = model.encoder.layer
        else:
            raise AttributeError(
                "DINOv3ViTModel transformer layers were not found under "
                ".layer, .model.layer, or .encoder.layer"
            )
        for layer in layers:
            hidden_states = layer(
                hidden_states, position_embeddings=position_embeddings
            )
        return functional.layer_norm(hidden_states, hidden_states.shape[-1:])

    extract_features._jittor_trellis_layer_compat = True
    extractor.extract_features = extract_features
    return True


def _patch_trellis2_rembg_lazy(module) -> bool:
    """Load the gated background-removal model only when it is used."""
    birefnet = getattr(module, "BiRefNet", None)
    if birefnet is None or getattr(
        getattr(birefnet, "__init__", None), "_jittor_trellis_lazy", False
    ):
        return False
    from torchvision import transforms

    original_call = birefnet.__call__

    def lazy_init(self, model_name="ZhengPeng7/BiRefNet"):
        self.model_name = model_name
        self.model = None
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                ),
            ]
        )

    lazy_init._jittor_trellis_lazy = True

    def ensure_model(self):
        if self.model is None:
            from transformers import AutoModelForImageSegmentation

            self.model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model.eval()

    def guarded_call(self, image):
        ensure_model(self)
        return original_call(self, image)

    def to(self, device):
        if self.model is not None:
            self.model.to(device)
        return self

    def cuda(self):
        return to(self, "cuda")

    def cpu(self):
        return to(self, "cpu")

    birefnet.__init__ = lazy_init
    birefnet.__call__ = guarded_call
    birefnet.to = to
    birefnet.cuda = cuda
    birefnet.cpu = cpu
    return True


def _build_trellis2_jittor_conv_module(module_name):
    """Build the module API expected by TRELLIS' sparse-conv dispatcher."""
    import torch
    import torch.nn as nn

    module = types.ModuleType(module_name)
    module.__package__ = "trellis2.modules.sparse.conv"

    def sparse_conv3d_init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=None,
        bias=True,
        indice_key=None,
    ):
        del indice_key
        unit_stride = (
            tuple(stride) == (1, 1, 1)
            if isinstance(stride, (list, tuple))
            else stride == 1
        )
        if not unit_stride or padding is not None:
            raise ValueError(
                "the Jittor sparse backend only supports submanifold "
                "convolution with stride=1 and padding=None"
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            tuple(kernel_size)
            if isinstance(kernel_size, (list, tuple))
            else (kernel_size,) * 3
        )
        self.stride = (
            tuple(stride) if isinstance(stride, (list, tuple)) else (stride,) * 3
        )
        self.dilation = (
            tuple(dilation)
            if isinstance(dilation, (list, tuple))
            else (dilation,) * 3
        )
        shape = (out_channels, in_channels) + self.kernel_size
        self.weight = nn.Parameter(torch.empty(shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)
        self.weight = nn.Parameter(
            self.weight.permute(0, 2, 3, 4, 1).contiguous()
        )

    def sparse_conv3d_forward(self, sparse_tensor):
        from jittor.nn.sparse import (
            build_submanifold_conv3d_neighbors,
            submanifold_conv3d,
        )

        _, kernel_depth, kernel_height, kernel_width, _ = self.weight.shape
        kernel = (kernel_depth, kernel_height, kernel_width)
        key = "SubMConv3d_jittor_neighbors_%sx%sx%s_dilation%s" % (
            kernel_width,
            kernel_height,
            kernel_depth,
            self.dilation,
        )
        neighbors = sparse_tensor.get_spatial_cache(key)
        if neighbors is None:
            neighbors = build_submanifold_conv3d_neighbors(
                sparse_tensor.coords, kernel, dilation=self.dilation
            )
            sparse_tensor.register_spatial_cache(key, neighbors)
        output = submanifold_conv3d(
            sparse_tensor.feats,
            sparse_tensor.coords,
            self.weight,
            self.bias,
            dilation=self.dilation,
            neighbors=neighbors,
        )
        return sparse_tensor.replace(output)

    def sparse_inverse_conv3d_init(self, *args, **kwargs):
        del self, args, kwargs
        raise NotImplementedError("SparseInverseConv3d is not implemented")

    def sparse_inverse_conv3d_forward(self, sparse_tensor):
        del self, sparse_tensor
        raise NotImplementedError("SparseInverseConv3d is not implemented")

    module.sparse_conv3d_init = sparse_conv3d_init
    module.sparse_conv3d_forward = sparse_conv3d_forward
    module.sparse_inverse_conv3d_init = sparse_inverse_conv3d_init
    module.sparse_inverse_conv3d_forward = sparse_inverse_conv3d_forward
    return module


def install_trellis2_sparse_conv_jittor(config_module=None) -> bool:
    """Register and select the production Jittor submanifold backend."""
    if config_module is None:
        try:
            import trellis2.modules.sparse.config as config_module
        except Exception:
            return False
    module_name = "trellis2.modules.sparse.conv.conv_jittor"
    if module_name not in sys.modules:
        sys.modules[module_name] = _build_trellis2_jittor_conv_module(module_name)
    setter = getattr(config_module, "set_conv_backend", None)
    if not callable(setter):
        return False
    setter("jittor")
    dispatcher = sys.modules.get("trellis2.modules.sparse.conv.conv")
    backends = getattr(dispatcher, "_backends", None)
    if isinstance(backends, dict):
        backends.pop("jittor", None)
    return True


def _patch_sparse_config_module(module) -> bool:
    if not _jittor_sparse_backend_enabled():
        return False
    return install_trellis2_sparse_conv_jittor(module)


def force_flexgemm_bridge_algorithm(
    algorithm="IMPLICIT_GEMM", spconv_module=None
):
    """Select the FlexGEMM algorithm validated through Jittor's bridge."""
    if _jittor_sparse_backend_enabled():
        return None
    if spconv_module is None:
        try:
            import flex_gemm.ops.spconv as spconv_module
        except Exception:
            return None
    algorithm_type = getattr(spconv_module, "Algorithm", None)
    setter = getattr(spconv_module, "set_algorithm", None)
    if algorithm_type is None or not callable(setter):
        return None
    selected = getattr(algorithm_type, algorithm, algorithm)
    setter(selected)
    config_value = (
        selected
        if isinstance(selected, str)
        else getattr(selected, "value", str(selected))
    )
    config = sys.modules.get("trellis2.modules.sparse.conv.config")
    if config is not None:
        config.FLEX_GEMM_ALGO = config_value

    try:
        submanifold = sys.modules.get("flex_gemm.ops.spconv.submanifold_conv3d")
        if submanifold is None:
            from flex_gemm.ops.spconv import submanifold_conv3d as submanifold
        function = submanifold.SubMConv3dFunction
        if not getattr(function, "_jittor_trellis_bridge_counted", False):
            current = function._sparse_submanifold_conv_forward
            original = current.__func__ if hasattr(current, "__func__") else current

            def counted_forward(feats, neighbor_cache, weight, bias=None):
                global FLEXGEMM_BRIDGE_CONV_CALLS
                FLEXGEMM_BRIDGE_CONV_CALLS += 1
                return original(feats, neighbor_cache, weight, bias)

            function._sparse_submanifold_conv_forward = staticmethod(counted_forward)
            function._jittor_trellis_bridge_counted = True
    except Exception:
        pass
    return algorithm


def _patch_flexgemm_spconv_module(module) -> bool:
    algorithm = os.environ.get(
        "JITTOR_TRELLIS_FLEXGEMM_ALGORITHM", "IMPLICIT_GEMM"
    )
    return force_flexgemm_bridge_algorithm(algorithm, module) is not None


def _patch_trellis_conv_config(module) -> bool:
    if _jittor_sparse_backend_enabled():
        return False
    algorithm = os.environ.get(
        "JITTOR_TRELLIS_FLEXGEMM_ALGORITHM", "IMPLICIT_GEMM"
    )
    value = algorithm.lower()
    if getattr(module, "FLEX_GEMM_ALGO", None) == value:
        return False
    module.FLEX_GEMM_ALGO = value
    return True


_MODULE_PATCHES = (
    ("triton.runtime.autotuner", _patch_flexgemm_triton_autotuner),
    (
        "trellis2.modules.image_feature_extractor",
        _patch_dinov3_encoder_layer_layout,
    ),
    ("trellis2.pipelines.rembg.BiRefNet", _patch_trellis2_rembg_lazy),
    ("trellis2.modules.sparse.config", _patch_sparse_config_module),
    ("flex_gemm.ops.spconv", _patch_flexgemm_spconv_module),
    ("trellis2.modules.sparse.conv.config", _patch_trellis_conv_config),
)


def register_patches(register) -> None:
    for path, callback in _MODULE_PATCHES:
        register(path, callback)


__all__ = [
    "FLEXGEMM_BRIDGE_CONV_CALLS",
    "force_flexgemm_bridge_algorithm",
    "install_trellis2_sparse_conv_jittor",
    "register_patches",
]
