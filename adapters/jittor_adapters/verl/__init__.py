"""Exact verl 0.9.1 private TensorDict type-dispatch adaptation.

The frontend's real NestedTensor is NOT a torch.Tensor. Five upstream helpers
hard-code torch.Tensor in their leaf predicates. Add the public nested type to
those predicates, without replacing a type, builtin, payload or computation.
All upstream statements/assertions and dynamic globals remain the originals.

Source: verl-project/verl commit 1876b06d0a3e4e71e06230be10af14492ca8a75b.
This is deliberately source-pinned: version alone cannot approve a fork whose
private control flow changed. Source-less packages fail explicitly.
"""
import ast
import hashlib
import inspect
from functools import update_wrapper
from pathlib import Path
import sys
import types

from .._common import (require_version, required_patch, replace,
                      replace_bound_aliases, UnsupportedAdapterVersion)


SUPPORTED_VERSIONS = frozenset(("0.9.1",))
SOURCE_SHA256 = "4dc4c321a51e256232751ddb5cb2a2be129076e7f02f36a685fcad6d0d98f5f1"
# (inspect.getsource SHA256, exact number of changed isinstance predicates)
FUNCTIONS = {
    "concat_tensordict": ("aa8cb4f745674306941253f4a4d805ea41ad8e84d8b56e23cecfd9985308ff2b", 2),
    "chunk_tensordict": ("4cdf3d25609d3db40d272b5914dcab9dbd0dad9fb8ce0c471954b68d89b0be18", 1),
    "get_tensordict": ("bad88f44b8171a0bfaebcd8b436aa813526d6c74cf6b555409e466710b89bbac", 5),
    "index_select_tensor_dict": ("ecca58fdcf80a3eda0631e7d84a4c4c7b12bc7297e2bc78c7ac50b713d2d5413", 2),
    "get": ("bafe55f84a6647eab34adfbcf6b2dc46951d34721f17fec33a5c32ebc8837933", 1),
    "contiguous": ("424b338e1f9a033d268e0ff61066cce9c950ae5646e2c4a17abe9a164888e13f", 1),
}
_CAPTURE = "_jittor_verl_nested_type"
_TENSOR = ast.dump(ast.parse("torch.Tensor", mode="eval").body)
_UNION = ast.dump(ast.parse("torch.Tensor | list | np.ndarray", mode="eval").body)


class _ExpandTensorPredicate(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Name) or node.func.id != "isinstance":
            return node
        if len(node.args) != 2 or node.keywords:
            raise UnsupportedAdapterVersion("verl isinstance predicate contract changed")
        predicate = node.args[1]
        contains_tensor = any(ast.dump(child) == _TENSOR for child in ast.walk(predicate))
        if not contains_tensor:
            return node
        spelling = ast.dump(predicate)
        if spelling == _TENSOR:
            accepted = [predicate]
        elif spelling == _UNION:
            accepted = [predicate.left.left, predicate.left.right, predicate.right]
        else:
            raise UnsupportedAdapterVersion("Unrecognized verl Tensor predicate; refusing a broad rewrite")
        node.args[1] = ast.copy_location(ast.Tuple(
            elts=accepted + [ast.Name(id=_CAPTURE, ctx=ast.Load())], ctx=ast.Load()), predicate)
        self.changed += 1
        return node


def _clone_helper(original, nested_type, expected_hash, expected_changes):
    if not isinstance(original, types.FunctionType) or original.__closure__:
        raise UnsupportedAdapterVersion("verl helper is not an unmodified module-level function")
    try:
        source_lines, source_start = inspect.getsourcelines(original)
        source = "".join(source_lines)
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl helper source is required") from error
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != expected_hash:
        raise UnsupportedAdapterVersion("verl helper source changed: " + original.__name__)
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise UnsupportedAdapterVersion("verl helper definition changed")
    function = tree.body[0]
    if function.decorator_list or any(isinstance(node, ast.Name) and node.id == _CAPTURE
                                      for node in ast.walk(function)):
        raise UnsupportedAdapterVersion("verl helper closure/decorator contract changed")
    rewrite = _ExpandTensorPredicate()
    rewrite.visit(function)
    if rewrite.changed != expected_changes:
        raise UnsupportedAdapterVersion("verl Tensor predicate count changed: " + original.__name__)
    # Do not reevaluate annotations or default expressions. Their original
    # objects are restored below; the call signature itself is unchanged.
    function.returns = None
    for argument in (list(getattr(function.args, "posonlyargs", ())) + function.args.args
                     + function.args.kwonlyargs + [function.args.vararg, function.args.kwarg]):
        if argument is not None:
            argument.annotation = None
    function.args.defaults = [ast.parse("None", mode="eval").body for _ in function.args.defaults]
    function.args.kw_defaults = [None if value is None else ast.parse("None", mode="eval").body
                                for value in function.args.kw_defaults]
    # Parsed snippets start at line 1. Preserve the upstream definition and
    # assertion locations before nesting this function inside the capture
    # factory; otherwise tracebacks point into unrelated upstream source.
    ast.fix_missing_locations(function)
    ast.increment_lineno(function, source_start - 1)
    factory = ast.parse("def _capture_nested(_jittor_verl_nested_type):\n    pass\n")
    factory.body[0].body = [function, ast.Return(value=ast.Name(id=function.name, ctx=ast.Load()))]
    compiled = compile(ast.fix_missing_locations(factory), original.__code__.co_filename, "exec")
    factory_code = next(value for value in compiled.co_consts if isinstance(value, types.CodeType))
    # Crucial: retain the exact original globals dict, not a copied snapshot.
    # Helper-to-helper calls see subsequent owned patches and later restores.
    replacement = types.FunctionType(factory_code, original.__globals__)(nested_type)
    replacement.__defaults__ = original.__defaults__
    replacement.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(replacement, original)
    replacement._jittor_verl_original = original
    replacement._jittor_verl_nested_type = nested_type
    return replacement


@required_patch
def patch_tensordict_utils(module):
    require_version("verl", SUPPORTED_VERSIONS)
    from jittor.compat.torch import nested_tensor_type

    nested_type = nested_tensor_type()
    if not isinstance(nested_type, type):
        raise UnsupportedAdapterVersion("nested_tensor_type() did not return the actual frontend type")
    try:
        source_file = inspect.getsourcefile(module)
        source = Path(source_file).read_bytes() if source_file else b""
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl module source is required") from error
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise UnsupportedAdapterVersion("verl.utils.tensordict_utils source SHA256 changed")
    prepared = []
    for name, (digest, count) in FUNCTIONS.items():
        current = vars(module).get(name)
        if getattr(current, "_jittor_verl_nested_type", None) is nested_type:
            continue
        if not isinstance(current, types.FunctionType) or current.__globals__ is not vars(module):
            raise UnsupportedAdapterVersion("verl helper ownership changed: " + name)
        replacement = _clone_helper(current, nested_type, digest, count)
        prepared.append((name, current, replacement))
    # All contracts are checked before the first mutation. The module-patcher's
    # runtime hook owns rollback/uninstall and preserves external replacements.
    for name, original, replacement in prepared:
        replace(module, name, replacement, original)
        replace_bound_aliases("verl", name, original, replacement)
    return bool(prepared)


@required_patch
def patch_verl(module):
    # verl 0.9.1 imports protocol -> transferqueue_utils -> tensordict_utils
    # before assigning root __version__. Wait for root initialization to finish;
    # never infer a version from unrelated distribution metadata or import a
    # missing helper to make a changed upstream initialization order appear OK.
    from .native_npu import enabled as native_npu_enabled, report_torch_api_version
    from .worker_npu_context import patch_dataproto_serialization

    # verl 0.9.1 gates FSDP2/DTensor imports on torch.__version__.  The Jittor
    # frontend deliberately reports its native version by default and exposes a
    # public owner API for exactly this downstream case.  Request the Torch API
    # level before any later verl trainer module evaluates those gates.
    report_torch_api_version()
    require_version("verl", SUPPORTED_VERSIONS)
    protocol_changed = False
    if native_npu_enabled():
        protocol = sys.modules.get("verl.protocol")
        if not isinstance(protocol, types.ModuleType):
            raise UnsupportedAdapterVersion("verl root did not load its expected protocol module")
        protocol_changed = patch_dataproto_serialization(protocol)
    helper = sys.modules.get("verl.utils.tensordict_utils")
    if not isinstance(helper, types.ModuleType):
        raise UnsupportedAdapterVersion("verl root did not load its expected TensorDict helper")
    helper_changed = patch_tensordict_utils(helper)
    return protocol_changed or helper_changed


def register(register_module_patch):
    from .device_probe import register as register_device_probe
    from .fsdp2_resident_reference import register as register_fsdp2_resident_reference
    from .hf_async_rollout import register as register_hf_async_rollout
    from .jagged_positions import register as register_jagged_positions
    from .native_npu import register as register_native_npu
    from .optimizer_offload import register as register_optimizer_offload
    from .valuehead_attention import register as register_valuehead_attention
    from .worker_npu_context import register as register_worker_npu_context

    register_device_probe(register_module_patch)
    register_hf_async_rollout(register_module_patch)
    register_valuehead_attention(register_module_patch)
    register_fsdp2_resident_reference(register_module_patch)
    register_native_npu(register_module_patch)
    register_optimizer_offload(register_module_patch)
    register_worker_npu_context(register_module_patch)
    register_module_patch("verl", patch_verl)
    # Run after the existing root callback, when actual verl __version__ exists.
    register_jagged_positions(register_module_patch)
