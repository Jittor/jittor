"""Native module aliases and the shared, lazy alias import mechanism."""

from __future__ import absolute_import

import importlib
import importlib.abc
import importlib.util
import sys

ALIASES = {
    "jittor._arg_policy": "jittor._core.arg_policy",
    "jittor._composition": "jittor._runtime.composition",
    "jittor._install_order": "jittor._runtime.install_order",
    "jittor.benchmarking": "jittor.tools.benchmarking",
    "jittor.misc.concatenation": "jittor.ops.concatenation",
    "jittor.misc.indexing": "jittor.ops.indexing",
    "jittor.misc.reductions": "jittor.ops.reductions",
    "jittor.misc.shape_composition": "jittor.ops.shape_composition",
    "jittor.misc.shape_transforms": "jittor.ops.shape_transforms",
    "jittor.misc.tensor_ops": "jittor.ops.tensor_ops",
    "jittor.compiler": "jittor.build.compiler",
    "jittor.compile_extern": "jittor.build.compile_extern",
    "jittor.pyjt_compiler": "jittor.build.pyjt_compiler",
    "jittor.init_cupy": "jittor.build.init_cupy",
    "jittor.ccl": "jittor.contrib.ccl",
    "jittor.ccl.ccl_2d": "jittor.contrib.ccl.ccl_2d",
    "jittor.ccl.ccl_3d": "jittor.contrib.ccl.ccl_3d",
    "jittor.ccl.ccl_link": "jittor.contrib.ccl.ccl_link",
    "jittor.loss3d": "jittor.contrib.loss3d",
    "jittor.loss3d.chamfer": "jittor.contrib.loss3d.chamfer",
    "jittor.loss3d.emd": "jittor.contrib.loss3d.emd",
    "jittor.math_util": "jittor.contrib.math_util",
    "jittor.math_util.gamma": "jittor.contrib.math_util.gamma",
    "jittor.math_util.igamma": "jittor.contrib.math_util.igamma",
    "jittor.einops": "jittor.contrib.einops",
    "jittor.einops._backends": "jittor.contrib.einops._backends",
    "jittor.einops.einops": "jittor.contrib.einops.einops",
    "jittor.einops.parsing": "jittor.contrib.einops.parsing",
    "jittor.einops.experimental": "jittor.contrib.einops.experimental",
    "jittor.einops.experimental.indexing": "jittor.contrib.einops.experimental.indexing",
    "jittor.einops.layers": "jittor.contrib.einops.layers",
    "jittor.einops.layers._einmix": "jittor.contrib.einops.layers._einmix",
    "jittor.einops.layers.jittor": "jittor.contrib.einops.layers.jittor",
    "jittor.build.utils": "jittor_utils",
    "jittor.build.utils.auto_diff": "jittor_utils.auto_diff",
    "jittor.build.utils.backend_discovery": "jittor_utils.backend_discovery",
    "jittor.build.utils.backend_resources": "jittor_utils.backend_resources",
    "jittor.build.utils.bootstrap": "jittor_utils.bootstrap",
    "jittor.build.utils.build_config": "jittor_utils.build_config",
    "jittor.build.utils.clean_cache": "jittor_utils.clean_cache",
    "jittor.build.utils.compiler_flags": "jittor_utils.compiler_flags",
    "jittor.build.utils.config": "jittor_utils.config",
    "jittor.build.utils.cuda_wheel": "jittor_utils.cuda_wheel",
    "jittor.build.utils.env_config": "jittor_utils.env_config",
    "jittor.build.utils.env_manifest": "jittor_utils.env_manifest",
    "jittor.build.utils.install_cuda": "jittor_utils.install_cuda",
    "jittor.build.utils.install_msvc": "jittor_utils.install_msvc",
    "jittor.build.utils.load_pytorch": "jittor_utils.load_pytorch",
    "jittor.build.utils.load_pytorch_old": "jittor_utils.load_pytorch_old",
    "jittor.build.utils.lock": "jittor_utils.lock",
    "jittor.build.utils.manifest": "jittor_utils.manifest",
    "jittor.build.utils.misc": "jittor_utils.misc",
    "jittor.build.utils.preflight": "jittor_utils.preflight",
    "jittor.build.utils.probe": "jittor_utils.probe",
    "jittor.build.utils.query_cuda_cc": "jittor_utils.query_cuda_cc",
    "jittor.build.utils.ring_buffer": "jittor_utils.ring_buffer",
    "jittor.build.utils.runtime_services": "jittor_utils.runtime_services",
    "jittor.build.utils.save_pytorch": "jittor_utils.save_pytorch",
    "jittor.build.utils.student_queue": "jittor_utils.student_queue",
    "jittor.build.utils.class": "jittor_utils.class",
    "jittor.build.utils.class.setup": "jittor_utils.class.setup",
    "jittor.build.utils.class.setup_env": "jittor_utils.class.setup_env",
    "jittor.extern.acl.aclops": "jittor.backends.acl.kernels.ops",
    "jittor.extern.acl.aclops._code": "jittor.backends.acl.kernels.ops._code",
    "jittor.extern.acl.aclops.acl_data": "jittor.backends.acl.kernels.ops.acl_data",
    "jittor.extern.acl.aclops.adamw_op": "jittor.backends.acl.kernels.ops.adamw_op",
    "jittor.extern.acl.aclops.arg_reduce_op": "jittor.backends.acl.kernels.ops.arg_reduce_op",
    "jittor.extern.acl.aclops.bmm_op": "jittor.backends.acl.kernels.ops.bmm_op",
    "jittor.extern.acl.aclops.clamp_op": "jittor.backends.acl.kernels.ops.clamp_op",
    "jittor.extern.acl.aclops.concat_op": "jittor.backends.acl.kernels.ops.concat_op",
    "jittor.extern.acl.aclops.conv_op": "jittor.backends.acl.kernels.ops.conv_op",
    "jittor.extern.acl.aclops.cumsum_op": "jittor.backends.acl.kernels.ops.cumsum_op",
    "jittor.extern.acl.aclops.dropout_op": "jittor.backends.acl.kernels.ops.dropout_op",
    "jittor.extern.acl.aclops.embedding_op": "jittor.backends.acl.kernels.ops.embedding_op",
    "jittor.extern.acl.aclops.flashattention_op": "jittor.backends.acl.kernels.ops.flashattention_op",
    "jittor.extern.acl.aclops.flip_op": "jittor.backends.acl.kernels.ops.flip_op",
    "jittor.extern.acl.aclops.floor_op": "jittor.backends.acl.kernels.ops.floor_op",
    "jittor.extern.acl.aclops.gather_scatter_op": "jittor.backends.acl.kernels.ops.gather_scatter_op",
    "jittor.extern.acl.aclops.getitem_op": "jittor.backends.acl.kernels.ops.getitem_op",
    "jittor.extern.acl.aclops.index_op": "jittor.backends.acl.kernels.ops.index_op",
    "jittor.extern.acl.aclops.matmul_op": "jittor.backends.acl.kernels.ops.matmul_op",
    "jittor.extern.acl.aclops.nantonum_op": "jittor.backends.acl.kernels.ops.nantonum_op",
    "jittor.extern.acl.aclops.norms_op": "jittor.backends.acl.kernels.ops.norms_op",
    "jittor.extern.acl.aclops.pool_op": "jittor.backends.acl.kernels.ops.pool_op",
    "jittor.extern.acl.aclops.relu_op": "jittor.backends.acl.kernels.ops.relu_op",
    "jittor.extern.acl.aclops.roll_op": "jittor.backends.acl.kernels.ops.roll_op",
    "jittor.extern.acl.aclops.rope_op": "jittor.backends.acl.kernels.ops.rope_op",
    "jittor.extern.acl.aclops.setitem_op": "jittor.backends.acl.kernels.ops.setitem_op",
    "jittor.extern.acl.aclops.sigmoid_op": "jittor.backends.acl.kernels.ops.sigmoid_op",
    "jittor.extern.acl.aclops.silu_op": "jittor.backends.acl.kernels.ops.silu_op",
    "jittor.extern.acl.aclops.softmax_op": "jittor.backends.acl.kernels.ops.softmax_op",
    "jittor.extern.acl.aclops.stack_op": "jittor.backends.acl.kernels.ops.stack_op",
    "jittor.extern.acl.aclops.transpose_op": "jittor.backends.acl.kernels.ops.transpose_op",
    "jittor.extern.acl.aclops.triu_op": "jittor.backends.acl.kernels.ops.triu_op",
    "jittor.extern.acl.aclops.truth_reduce_op": "jittor.backends.acl.kernels.ops.truth_reduce_op",
    "jittor.extern.acl.aclops.upsample_op": "jittor.backends.acl.kernels.ops.upsample_op",
    "jittor.extern.acl.aclops.where_op": "jittor.backends.acl.kernels.ops.where_op",
    "jittor.attention": "jittor.nn.attention",
    "jittor.gradfunctional": "jittor.autograd",
    "jittor.gradfunctional.functional": "jittor.autograd.functional",
    "jittor.other": "jittor.nn.backends",
    "jittor.other.code_softmax": "jittor.backends.cuda.kernels.nn.softmax_cuda",
    "jittor.nn.backends.batch_norm_training_cuda": "jittor.backends.cuda.kernels.nn.batch_norm_training_cuda",
    "jittor.nn.backends.channel_bias_cuda": "jittor.backends.cuda.kernels.nn.channel_bias_cuda",
    "jittor.nn.backends.full_reduce_cuda": "jittor.backends.cuda.kernels.nn.full_reduce_cuda",
    "jittor.nn.backends.group_norm_cuda": "jittor.backends.cuda.kernels.nn.group_norm_cuda",
    "jittor.nn.backends.layer_norm_cuda": "jittor.backends.cuda.kernels.nn.layer_norm_cuda",
    "jittor.nn.backends.layer_norm_training_cuda": "jittor.backends.cuda.kernels.nn.layer_norm_training_cuda",
    "jittor.nn.backends.modulated_layer_norm_cuda": "jittor.backends.cuda.kernels.nn.modulated_layer_norm_cuda",
    "jittor.nn.backends.rms_norm_training_cuda": "jittor.backends.cuda.kernels.nn.rms_norm_training_cuda",
    "jittor.nn.backends.softmax_cuda": "jittor.backends.cuda.kernels.nn.softmax_cuda",
    "jittor.nn.rms_norm_cuda": "jittor.backends.cuda.kernels.nn.rms_norm_cuda",
    "jittor.nn.rope_cuda": "jittor.backends.cuda.kernels.nn.rope_cuda",
    "jittor.nn.swiglu_cuda": "jittor.backends.cuda.kernels.nn.swiglu_cuda",
    "jittor.nn.kv_cache_cuda": "jittor.backends.cuda.kernels.nn.kv_cache_cuda",
    "jittor.nn.packed_qkv_cuda": "jittor.backends.cuda.kernels.nn.packed_qkv_cuda",
    "jittor.nn._cuda_inference": "jittor.backends.cuda.kernels.nn._inference",
    "jittor.nn.kv_cache_acl": "jittor.backends.acl.kernels.kv_cache",
    "jittor.lr_scheduler": "jittor.optim.legacy_schedulers",
    "jittor.nn.sparse": "jittor.sparse.convolution",
    "jittor.weightnorm": "jittor.nn.utils.weight_norm",
    "jittor.depthwise_conv": "jittor.nn.modules.depthwise",
}

_PACKAGE_TARGETS = set(["jittor_utils","jittor_utils.class","jittor.contrib","jittor.contrib.ccl","jittor.contrib.loss3d","jittor.contrib.math_util","jittor.contrib.einops","jittor.contrib.einops.experimental","jittor.contrib.einops.layers","jittor.backends.acl.kernels.ops","jittor.autograd","jittor.nn.backends","jittor.sparse"])
_LAZY_PARENT_BINDINGS = set()
_IMPORT_CALLBACKS = {}
_ALIAS_PROVIDERS = {}
_FUNCTION_PARENT_ALIASES = frozenset((
    "jittor.ccl.ccl_2d", "jittor.ccl.ccl_3d", "jittor.ccl.ccl_link",
    "jittor.math_util.igamma",
))


def register_aliases(aliases, *, packages=(), lazy_parent_bindings=(), on_import=None):
    """Extend the shared loader without importing optional implementations."""
    for alias, canonical in aliases.items():
        existing = ALIASES.get(alias)
        if existing is not None and existing != canonical:
            raise RuntimeError("module alias %r already has another target" % alias)
    ALIASES.update(aliases)
    _PACKAGE_TARGETS.update(packages)
    _LAZY_PARENT_BINDINGS.update(lazy_parent_bindings)
    if on_import:
        _IMPORT_CALLBACKS.update(on_import)


def register_alias_provider(prefix, module):
    """Load an optional alias registrar only when its legacy name is requested."""
    _ALIAS_PROVIDERS[prefix] = module


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, alias, canonical):
        self.alias = alias
        self.canonical = canonical
        self.metadata = None

    def create_module(self, spec):
        module = importlib.import_module(self.canonical)
        self.metadata = (
            module.__name__,
            module.__package__,
            module.__loader__,
            module.__spec__,
        )
        return module

    def exec_module(self, module):
        module.__name__, module.__package__, module.__loader__, module.__spec__ = self.metadata
        _publish_alias(self.alias, module)
        if self.canonical in _PACKAGE_TARGETS:
            publish_loaded_aliases()
        callback = _IMPORT_CALLBACKS.get(self.alias)
        if callback is not None:
            callback(module)


class _AliasFinder(importlib.abc.MetaPathFinder):
    _jittor_compat_alias_finder = True

    def find_spec(self, fullname, path=None, target=None):
        canonical = ALIASES.get(fullname)
        if canonical is None:
            for prefix, provider in _ALIAS_PROVIDERS.items():
                if fullname == prefix or fullname.startswith(prefix + "."):
                    importlib.import_module(provider)
                    canonical = ALIASES.get(fullname)
                    break
        if canonical is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(fullname, canonical),
            is_package=canonical in _PACKAGE_TARGETS,
        )


_FINDER = _AliasFinder()


def _bind_parent(alias, module):
    # Some historical packages export a function with the child module name.
    # Publish its import/pickle alias without replacing that function object.
    if alias in _FUNCTION_PARENT_ALIASES:
        return
    if "." not in alias:
        return
    parent_name, attr = alias.rsplit(".", 1)
    # Private source aliases preserve imports/pickles, not NN facade exports.
    if parent_name == "jittor.nn" and attr.startswith("_"):
        return
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attr, module)


def _publish_alias(alias, module, bind_parent=True):
    current = sys.modules.get(alias)
    if current is not None and current is not module:
        raise RuntimeError("module alias %r already published with a different object" % alias)
    sys.modules[alias] = module
    if bind_parent:
        _bind_parent(alias, module)
    return module


def publish_loaded_aliases(root_module=None):
    for alias, canonical in ALIASES.items():
        if alias in _LAZY_PARENT_BINDINGS:
            continue
        module = sys.modules.get(canonical)
        if module is None:
            continue
        _publish_alias(alias, module)
    if root_module is not None:
        for attr, canonical in (
            ("attention", "jittor.nn.attention"),
            ("lr_scheduler", "jittor.optim.legacy_schedulers"),
            ("depthwise_conv", "jittor.nn.modules.depthwise"),
        ):
            module = sys.modules.get(canonical)
            if module is not None:
                setattr(root_module, attr, module)


def install_aliases(root_module=None):
    if not any(getattr(finder, "_jittor_compat_alias_finder", False) for finder in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)
    publish_loaded_aliases(root_module)
    return dict(ALIASES)


def import_alias(alias):
    canonical = ALIASES[alias]
    module = importlib.import_module(canonical)
    return _publish_alias(alias, module)
