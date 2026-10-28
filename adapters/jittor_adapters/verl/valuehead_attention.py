"""Respect explicit eager attention in the pinned verl CUDA value-head loader.

Breakpoint belongs to verl's private loader, not Torch or a missing kernel:
verl 0.9.1 overrides the user's HF attention configuration at both loaders.
Only those two keyword values change. Missing/non-eager configuration retains
the original FlashAttention request, including every original TRL fallback.
No framework attributes, import availability, or computations are replaced.

Upstream: verl-project/verl 1876b06d0a3e4e71e06230be10af14492ca8a75b.
Candidate uses the owned module-patch lifecycle; the isolated official oracle
uses prepare() directly, which never imports Jittor. Host tests are not CUDA
acceptance. Revalidate on upstream helper/config or compat module-patcher changes.
"""
import ast
from functools import update_wrapper
import hashlib
import inspect
import os
import types

from .._common import (require_version, required_patch, replace,
                      replace_bound_aliases, UnsupportedAdapterVersion)


SUPPORTED_VERSIONS = frozenset(("0.9.1",))
MODULE = "verl.utils.model"
FUNCTION = "load_valuehead_model"
FUNCTION_SHA256 = "2371ca280654b53bc2f25101ff2eb1898aafef14570f7ed94883d0b4a4ac825b"
EXPECTED_CHANGES = 2
POLICY = "verl_v091_explicit_eager_valuehead_cuda_hf_async_v1"


def enabled():
    return (os.environ.get("VERL_DEVICE") == "cuda"
            and os.environ.get("JITTOR_VERL_HF_ASYNC_ROLLOUT") == "1")


class _RespectEagerAttention(ast.NodeTransformer):
    def __init__(self):
        self.changed = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "from_pretrained":
            return node
        for keyword in node.keywords:
            if (keyword.arg == "attn_implementation"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == "flash_attention_2"):
                # Public HF config construction stores attn_implementation in
                # this downstream property. Read it, never mutate the config.
                expression = ast.parse(
                    "'eager' if getattr(model_config, '_attn_implementation', None) == 'eager' "
                    "else 'flash_attention_2'", mode="eval").body
                keyword.value = ast.copy_location(expression, keyword.value)
                self.changed += 1
        return node


def _clone(original):
    if not isinstance(original, types.FunctionType) or original.__closure__:
        raise UnsupportedAdapterVersion("verl value-head loader must be a module-level function")
    try:
        lines, start = inspect.getsourcelines(original)
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("verl value-head loader source is required") from error
    source = "".join(lines)
    if hashlib.sha256(source.encode()).hexdigest() != FUNCTION_SHA256:
        raise UnsupportedAdapterVersion("verl value-head loader source SHA256 changed")
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != FUNCTION or tree.body[0].decorator_list):
        raise UnsupportedAdapterVersion("verl value-head loader definition changed")
    rewrite = _RespectEagerAttention()
    rewrite.visit(tree)
    if rewrite.changed != EXPECTED_CHANGES:
        raise UnsupportedAdapterVersion("verl value-head attention keyword count changed")
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, start - 1)
    compiled = compile(tree, original.__code__.co_filename, "exec")
    code = next(item for item in compiled.co_consts
                if isinstance(item, types.CodeType) and item.co_name == FUNCTION)
    replacement = types.FunctionType(code, original.__globals__, original.__name__, original.__defaults__)
    replacement.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(replacement, original)
    replacement._jittor_verl_original = original
    replacement._jittor_verl_valuehead_attention = POLICY
    return replacement


def prepare(module):
    """Prepare one exact helper for candidate or oracle without importing Jittor."""
    if not enabled():
        raise UnsupportedAdapterVersion("value-head attention policy requires explicit CUDA HF async scope")
    require_version("verl", SUPPORTED_VERSIONS)
    if module.__name__ != MODULE:
        raise UnsupportedAdapterVersion("unexpected verl value-head module")
    current = vars(module).get(FUNCTION)
    if (not isinstance(current, types.FunctionType) or current.__module__ != MODULE
            or current.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("verl value-head loader ownership changed")
    if getattr(current, "_jittor_verl_valuehead_attention", None) == POLICY:
        original = getattr(current, "_jittor_verl_original", None)
        if (not isinstance(original, types.FunctionType) or original.__module__ != MODULE
                or original.__globals__ is not vars(module)):
            raise UnsupportedAdapterVersion("verl value-head original ownership changed")
        _clone(original)  # An idempotent call must still validate upstream source.
        return current, current
    return current, _clone(current)


@required_patch
def patch_valuehead_attention(module):
    if not enabled():
        return False
    original, replacement = prepare(module)
    if original is replacement:
        return False
    replace(module, FUNCTION, replacement, original)
    replace_bound_aliases("verl", FUNCTION, original, replacement)
    return True


def register(register_module_patch):
    if enabled():
        register_module_patch(MODULE, patch_valuehead_attention)
