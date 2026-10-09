"""Fail-closed source identity and library-private import adaptation."""

import ast
import hashlib
from pathlib import Path

from .._common import UnsupportedAdapterVersion

# Critical upstream sources are identical for both explicitly pinned builds.
# The two generated manifests come from the same 0.17.6 sdist SHA256:
# b3318064ee5798e8a27d201ea8b888f0439973c4eac9af9ab381dd1862ebdf45.
# Source identity admission is not runtime/device acceptance evidence.
SOURCE_SHA256 = {
    "__init__.py": ("55ec79604e5d62ff53ca2e5622395d04a36476be201300f1fb938d07c9ee5e3b",),
    "accelerator/real_accelerator.py": (
        "a3067861c647fe388f221fbc95840d88112ff0c8dd0444268586a344304b7e5b",
    ),
    "runtime/engine.py": ("c1c402959e019dd6ef77353a37a6dbc36d9115a0b02124877c9514f3be4e2a45",),
    "comm/comm.py": ("272733a53182d7282ae6528281fa117282ff953c82e7a6757479e9d0497a08e7",),
    "elasticity/__init__.py": ("0a68b7804378f1561d3d2838ccb389d4a18a3f61b75deaeecfd4f29200b5de6e",),
    "runtime/zero/utils.py": ("adff6097123056836e4814bcdaf85c005db9879cee7872c7005cdb9d2ba71aa2",),
    "runtime/zero/stage_1_and_2.py": (
        "b446f62e84e1cc6344434134cb2dd874efcbe477811ddece2357be732ec67e9a",
    ),
    "runtime/zero/stage3.py": ("38674f8ba31b42b2083d49e48272ad1929f013254833e1c99c2695f31ba2a0b2",),
    "utils/torch.py": ("c7156c2878cef07a9e0c979f7d279dd5388fdfa20094c9fc96c6a42ee8fe306a",),
    "git_version_info_installed.py": (
        "ad44588cefdf97440706c2c62229619ca0f3801501d9ddc71e43abaa3b49adff",  # NPU build
        "ab1206bd635f789c618086e23bfffd24849012a564a1b7852dec0e537ba60f58",  # CPU build
    ),
}


def verified_source(package_root, relative):
    path = Path(package_root) / relative
    try:
        data = path.read_bytes()
    except OSError as error:
        raise UnsupportedAdapterVersion("DeepSpeed source is unavailable: " + str(path)) from error
    digest = hashlib.sha256(data).hexdigest()
    if digest not in SOURCE_SHA256[relative]:
        raise UnsupportedAdapterVersion(
            "Unvalidated DeepSpeed source %s: SHA256 %s" % (relative, digest)
        )
    return data.decode("utf-8")


def inspect_package(package_root):
    sources = {name: verified_source(package_root, name) for name in SOURCE_SHA256}
    tree = ast.parse(sources["git_version_info_installed.py"])
    versions = [
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "version" for target in node.targets)
    ]
    if len(versions) != 1 or not isinstance(versions[0], str):
        raise UnsupportedAdapterVersion("DeepSpeed source must define one literal version")
    return versions[0]


def _once(source, before, after):
    if source.count(before) != 1:
        raise UnsupportedAdapterVersion(
            "DeepSpeed source adaptation anchor changed: " + before[:90]
        )
    return source.replace(before, after, 1)


def transform_elasticity(source):
    """Skip the unused elastic launch agent on fixed single-node ranks."""
    return _once(
        source,
        "if is_torch_elastic_compatible():\n    from .elastic_agent import DSElasticAgent\n",
        "",
    )


def transform_comm(source):
    """Remove DeepSpeed 0.17.6 unused GradBucket import; no fake DDP bucket is exposed."""
    return _once(source, "from torch.distributed import GradBucket  # noqa: F401\n", "")


def transform_engine(source):
    """Defer unused compile imports and register children via public Module API."""
    for statement in (
        "from deepspeed.compile.backend import register_compile_pass, opt_passes\n",
        "from deepspeed.compile.passes import zero3_compile, prefetch, selective_gather, offload_adam_states\n",
        "from deepspeed.compile.init_z1 import init_z1\n",
        "from deepspeed.compile.init_z3 import init_z3\n",
    ):
        source = _once(source, statement, "")
    source = _once(
        source,
        "        if is_deepcompile_supported():\n",
        "        if is_deepcompile_supported():\n"
        "            from deepspeed.compile.passes import zero3_compile, prefetch, selective_gather, offload_adam_states\n",
    )
    source = _once(
        source,
        "        if enable_deepcompile:\n",
        "        if enable_deepcompile:\n"
        "            from deepspeed.compile.backend import opt_passes\n"
        "            from deepspeed.compile.init_z1 import init_z1\n"
        "            from deepspeed.compile.init_z3 import init_z3\n",
    )
    source = _once(
        source,
        "        register_compile_pass(pass_name, pass_fn)\n",
        "        from deepspeed.compile.backend import register_compile_pass\n"
        "        register_compile_pass(pass_name, pass_fn)\n",
    )
    return _once(
        source,
        "        modules = self.__dict__.get('_modules')\n        modules['module'] = model\n",
        "        self.add_module('module', model)\n",
    )


def transform_zero_utils(source):
    """Keep the pinned ZeRO optimizer whitelist within the explicit AdamW scope."""
    return _once(
        source,
        "torch.optim.Adam, torch.optim.AdamW, FusedAdam, DeepSpeedCPUAdam, torch.optim.Adagrad, DeepSpeedCPUAdagrad,",
        "torch.optim.Adam, torch.optim.AdamW, FusedAdam, DeepSpeedCPUAdam, DeepSpeedCPUAdagrad,",
    )


def transform_zero_adagrad_guard(source, stage):
    """Do not evaluate missing Adagrad on the explicitly AdamW-only engine path."""
    if stage == "stage_1_and_2":
        return _once(
            source,
            "if isinstance(self.optimizer, torch.optim.Adagrad):",
            "if False:  # Adagrad is outside this adapter scope",
        )
    if stage == "stage3":
        return _once(
            source,
            "is_adagrad = isinstance(self.optimizer, torch.optim.Adagrad)",
            "is_adagrad = False  # Adagrad is outside this adapter scope",
        )
    raise ValueError("Unknown DeepSpeed ZeRO stage: " + stage)


def transform_zero_stage_1_and_2(source):
    """Replace one shape-only meta allocation in the pinned ZeRO source."""
    source = _once(
        source,
        "from deepspeed.utils import groups\n# Toggle this to true to enable correctness test\n",
        "from deepspeed.utils import groups\n\n"
        "class _JittorShapeOnly:\n"
        "    __slots__ = ('shape',)\n"
        "    def __init__(self, shape):\n"
        "        self.shape = tuple(shape)\n\n"
        "# Toggle this to true to enable correctness test\n",
    )
    return _once(
        source,
        'torch.zeros_like(param.cpu_data, device="meta")',
        "_JittorShapeOnly(param.cpu_data.shape)",
    )


def transform_utils_torch(source):
    """Use the public post-accumulate hook supported by torch compat."""
    before = (
        "def register_grad_hook(param, hook):\n"
        "    if required_torch_version(min_version=2.1):\n"
        "        return param.register_post_accumulate_grad_hook(hook)\n"
        "    else:\n"
        "        param_tmp = param.expand_as(param)\n"
        "        grad_acc = param_tmp.grad_fn.next_functions[0][0]\n"
        "        return grad_acc.register_hook(hook)\n"
    )
    after = (
        "def register_grad_hook(param, hook):\n"
        "    return param.register_post_accumulate_grad_hook(hook)\n"
    )
    return _once(source, before, after)


def transform_zero_npu_norm(source):
    """Keep ZeRO gradient-norm reduction on ACL-supported FP32."""
    before = ".data.double()"
    if source.count(before) != 3:
        raise UnsupportedAdapterVersion("DeepSpeed NPU norm adaptation anchors changed: " + before)
    return source.replace(before, ".data.float()")


def transform_zero_npu_stage3_norm(source):
    """Match torch_npu Stage 3 norm accumulation on hardware without FP64."""
    before = ".double()"
    if source.count(before) != 5:
        raise UnsupportedAdapterVersion(
            "DeepSpeed Stage 3 NPU norm adaptation anchors changed: " + before
        )
    return source.replace(before, ".float()")


def transform_zero_npu_stage1(source):
    """Preserve ZeRO Stage 1 gradient reduction on Jittor NPU tensors.

    DeepSpeed's non-contiguous fallback buckets tensors by device-specific
    ``Tensor.type()`` strings. Torch compat exposes the public dtype contract
    without manufacturing NPU tensor subclasses, so bucket by dtype instead.
    In the default contiguous path DeepSpeed aliases every parameter gradient
    onto a communication-buffer slice through ``grad.data = slice.data``.
    Jittor data assignment copies values but does not create PyTorch storage
    aliases; copy reduced slices back before building the optimizer partition.
    """
    source = _once(
        source,
        "def split_half_float_double(tensors):\n"
        "    device_type = get_accelerator().device_name()\n"
        "    dtypes = [\n"
        '        "torch.{}.HalfTensor".format(device_type), "torch.{}.FloatTensor".format(device_type),\n'
        '        "torch.{}.DoubleTensor".format(device_type), "torch.{}.BFloat16Tensor".format(device_type)\n'
        "    ]\n"
        "    buckets = []\n"
        "    for i, dtype in enumerate(dtypes):\n"
        "        bucket = [t for t in tensors if t.type() == dtype]\n"
        "        if bucket:\n"
        "            buckets.append(bucket)\n"
        "    return buckets\n",
        "def split_half_float_double(tensors):\n"
        "    dtypes = (torch.float16, torch.float32, torch.float64, torch.bfloat16)\n"
        "    buckets = []\n"
        "    for dtype in dtypes:\n"
        "        bucket = [tensor for tensor in tensors if tensor.dtype == dtype]\n"
        "        if bucket:\n"
        "            buckets.append(bucket)\n"
        "    return buckets\n",
    )
    return _once(
        source,
        "                else:\n"
        "                    self.average_tensor(bucket.buffer[bucket.index].narrow(0, 0, bucket.elements), comm_dtype)\n"
        "            else:\n",
        "                else:\n"
        "                    reduced = bucket.buffer[bucket.index].narrow(0, 0, bucket.elements)\n"
        "                    self.average_tensor(reduced, comm_dtype)\n"
        "                    offset = 0\n"
        "                    for group_idx, param_idx_in_group, _ in bucket.params:\n"
        "                        param = self.bit16_groups[group_idx][param_idx_in_group]\n"
        "                        grad_reduc = self.get_gradient_for_reduction(param)\n"
        "                        count = param.numel()\n"
        "                        grad_reduc.copy_(reduced.narrow(0, offset, count).view_as(grad_reduc))\n"
        "                        offset += count\n"
        "            else:\n",
    )
