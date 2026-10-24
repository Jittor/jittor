"""Pinned private TD pickle restoration; opt-in named verl four-axis positions.

All storage/metadata reconstruction remains upstream. The default path retains
explicit public axis 1. Only root position_ids under the explicit verl profile
uses axis 2, with a complete companion input_ids contract. No Torch defaults,
private Torch fields, framework math or transport format are changed.
Review on TD helper, public nested construction, or owned-patcher changes.
"""
import ast
from functools import update_wrapper
import hashlib
import inspect
import os
import sys
import types

from .._common import (require_version, required_patch,
                      replace_bound_aliases, UnsupportedAdapterVersion)
from .jagged_rebuild import _check_owned_handle

SUPPORTED_VERSIONS = frozenset(("0.10.0",))
VERL_VERSIONS = frozenset(("0.9.1",))
MODULE = "tensordict._reductions"
FUNCTION = "_rebuild_tensordict_files_consolidated"
FUNCTION_SHA256 = "e9d2b2053ce228c7ec04e5ef43eb6e09836b75d91deff6d25a1b6fb6379ca17f"
POLICY = "tensordict_v010_public_consolidated_pickle_v2"
FLAG = "VERL_CORRECT_JAGGED_POSITIONS"


def enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise UnsupportedAdapterVersion(FLAG + " must be absent, 0 or 1")
    return value == "1"


def _require_profile():
    if not enabled():
        raise UnsupportedAdapterVersion("four-axis restore profile changed after installation")


def _ragged(size, torch):
    if isinstance(size, int):
        return type(size) is int and size == -1
    return isinstance(size, torch.SymInt)


def _offsets(torch, values, offsets, *, companion=False):
    # Only verl input_ids accepts int32 cu_seqlens from unpad_input. Positions
    # retain the pinned tensorlist helper's int64 contract. Do not cast either
    # storage view: serialized dtype, buffer aliasing and complete ranges stay
    # upstream-owned, and the caller still compares every offset value.
    dtypes = (torch.int32, torch.int64) if companion else (torch.int64,)
    if (offsets.dim() != 1 or offsets.dtype not in dtypes
            or offsets.requires_grad is not False or offsets.device != values.device):
        dtype_contract = "int32/int64" if companion else "int64"
        raise ValueError("four-axis restore offsets require rank-one " + dtype_contract
                         + " non-grad on the values device")
    result = offsets.detach().cpu().tolist()
    if (len(result) < 2 or any(type(x) is not int for x in result)
            or result[0] != 0 or any(a > b for a, b in zip(result, result[1:]))):
        raise ValueError("four-axis restore offsets must be zero-based nondecreasing integers")
    return result


def _construct(torch, key, prefix, values, offsets, lengths):
    if key != "<NJT_OFFSETS>position_ids":
        return torch.nested.nested_tensor_from_jagged(
            values.detach(), offsets=offsets, lengths=lengths, jagged_dim=1)
    require_version("verl", VERL_VERSIONS)
    if prefix is not None:
        raise ValueError("four-axis restore supports only root position_ids")
    if (values.dim() != 2 or values.shape[0] != 4 or values.dtype != torch.int64
            or values.requires_grad is not False or lengths is not None):
        raise ValueError("four-axis restore positions require [4,tokens] int64 non-grad without holes")
    offs = _offsets(torch, values, offsets)
    if offs[-1] != values.shape[1]:
        raise ValueError("four-axis restore offsets must span the full token axis")
    result = torch.nested.nested_tensor_from_jagged(
        values.detach(), offsets=offsets, lengths=None, jagged_dim=2)
    if (result.is_nested is not True or result.dim() != 3
            or result.shape[0] != len(offs) - 1 or result.shape[1] != 4
            or not _ragged(result.shape[2], torch)
            or result.device != values.device or result.dtype != torch.int64
            or result.requires_grad is not False or result.lengths() is not None
            or result.values().data_ptr() != values.data_ptr()
            or result.offsets() is not offsets):
        raise ValueError("four-axis public constructor did not preserve the validated layout/buffers")
    return result


def _validate(torch, data, cls_metadata, prefix):
    if "position_ids" not in data:
        return
    if prefix is not None:
        raise ValueError("four-axis restore supports only root position_ids")
    positions = data["position_ids"]
    if "input_ids" not in data:
        raise ValueError("four-axis restore requires complete companion input_ids")
    ids = data["input_ids"]
    if (positions.is_nested is not True or positions.dim() != 3
            or ids.is_nested is not True or ids.dim() != 2):
        raise ValueError("four-axis restore requires jagged positions/input_ids")
    pos_values, id_values = positions.values(), ids.values()
    if (pos_values.dim() != 2 or pos_values.shape[0] != 4 or id_values.dim() != 1
            or pos_values.dtype != torch.int64 or id_values.dtype != torch.int64
            or positions.dtype != torch.int64 or ids.dtype != torch.int64
            or positions.requires_grad is not False or ids.requires_grad is not False
            or pos_values.requires_grad is not False or id_values.requires_grad is not False
            or positions.lengths() is not None or ids.lengths() is not None
            or pos_values.device != id_values.device or positions.device != ids.device):
        raise ValueError("four-axis restore companion dtype/device/shape/gradient contract changed")
    offs = _offsets(torch, pos_values, positions.offsets())
    id_offs = _offsets(torch, id_values, ids.offsets(), companion=True)
    batch = cls_metadata.get("batch_size")
    if (offs != id_offs or offs[-1] != pos_values.shape[1] or offs[-1] != id_values.shape[0]
            or positions.shape[0] != len(offs) - 1 or ids.shape[0] != len(offs) - 1
            or positions.shape[1] != 4 or batch is None or len(batch) != 1
            or not _ragged(positions.shape[2], torch) or not _ragged(ids.shape[1], torch)
            or type(batch[0]) is not int or batch[0] != len(offs) - 1):
        raise ValueError("four-axis restore companion offsets/token counts/batch contract changed")


class _PublicJagged(ast.NodeTransformer):
    def __init__(self, four_axis=False):
        self.imports = self.calls = self.validators = self.initializers = self.local_contexts = self.axis_marks = 0
        self.four_axis = four_axis

    def visit_ImportFrom(self, node):
        if node.module == "torch.nested._internal.nested_tensor":
            if node.level != 0 or [(x.name, x.asname) for x in node.names] != [("NestedTensor", None)]:
                raise UnsupportedAdapterVersion("TensorDict private nested import changed")
            self.imports += 1
            return None
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "NestedTensor":
            expected = ast.parse("NestedTensor(nested_values, offsets=offsets, lengths=nested_lengths)").body[0].value
            if ast.dump(node) != ast.dump(expected):
                raise UnsupportedAdapterVersion("TensorDict private nested call changed")
            self.calls += 1
            expression = ("_td_construct(torch, key, prefix, nested_values, offsets, nested_lengths)"
                          if self.four_axis else
                          "torch.nested.nested_tensor_from_jagged(nested_values.detach(), "
                          "offsets=offsets, lengths=nested_lengths, jagged_dim=1)")
            return ast.copy_location(ast.parse(expression, mode="eval").body, node)
        return node

    def visit_Assign(self, node):
        constructor = ast.parse("value = NestedTensor(nested_values, offsets=offsets, lengths=nested_lengths)").body[0]
        mark_axis = self.four_axis and ast.dump(node) == ast.dump(constructor)
        node = self.generic_visit(node)
        if mark_axis:
            self.axis_marks += 1
            mark = ast.parse("_td_axis2 = _td_axis2 or key == '<NJT_OFFSETS>position_ids'").body[0]
            return [ast.copy_location(mark, node), node]
        expected = ast.parse("result = cls._from_dict_validated(d, **cls_metadata)").body[0]
        if self.four_axis and ast.dump(node) == ast.dump(expected):
            self.validators += 1
            validate = ast.parse("if _td_axis2:\n    _td_validate(torch, d, cls_metadata, prefix)").body[0]
            return [ast.copy_location(validate, node), node]
        return node

    def visit_FunctionDef(self, node):
        node = self.generic_visit(node)
        if self.four_axis and node.name == FUNCTION:
            self.initializers += 1
            node.body.insert(0, ast.copy_location(ast.parse("_td_require_profile()").body[0], node.body[0]))
        elif self.four_axis and node.name == "from_metadata":
            self.local_contexts += 1
            node.body.insert(0, ast.copy_location(ast.parse("_td_axis2 = False").body[0], node.body[0]))
        return node


def _rewrite(original, source, four_axis=False):
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != FUNCTION or tree.body[0].decorator_list):
        raise UnsupportedAdapterVersion("TensorDict consolidated definition changed")
    if any(isinstance(n, ast.Name) and n.id.startswith("_td_") for n in ast.walk(tree)):
        raise UnsupportedAdapterVersion("TensorDict reserved local ownership changed")
    rewrite = _PublicJagged(four_axis)
    tree = rewrite.visit(tree)
    if ((rewrite.imports, rewrite.calls) != (1, 1)
            or (rewrite.validators, rewrite.initializers, rewrite.local_contexts, rewrite.axis_marks) !=
               ((1, 1, 1, 1) if four_axis else (0, 0, 0, 0))):
        raise UnsupportedAdapterVersion("TensorDict consolidated helper AST changed")
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, original.__code__.co_firstlineno - 1)
    if four_axis:
        factory = ast.parse("def _td_factory(_td_construct, _td_validate, _td_require_profile):\n    pass\n")
        factory.body[0].body = tree.body + [ast.Return(ast.Name(id=FUNCTION, ctx=ast.Load()))]
        compiled = compile(ast.fix_missing_locations(factory), original.__code__.co_filename, "exec")
        code = next(c for c in compiled.co_consts if isinstance(c, types.CodeType))
        replacement = types.FunctionType(code, original.__globals__)(_construct, _validate, _require_profile)
    else:
        code = compile(tree, original.__code__.co_filename, "exec")
        functions = [item for item in code.co_consts if isinstance(item, types.CodeType)
                     and item.co_name == original.__name__]
        if len(functions) != 1:
            raise UnsupportedAdapterVersion("TensorDict consolidated helper code shape changed")
        replacement = types.FunctionType(functions[0], original.__globals__, original.__name__)
    replacement.__defaults__ = original.__defaults__
    replacement.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(replacement, original)
    replacement._jittor_tensordict_transport = POLICY
    replacement._jittor_tensordict_original = original
    replacement._jittor_tensordict_four_axis = four_axis
    return replacement


def prepare(module):
    """Shared framework-neutral source-owned plan; never import downstream/Jittor."""
    four_axis = enabled()
    require_version("tensordict", SUPPORTED_VERSIONS)
    if module.__name__ != MODULE or sys.modules.get(MODULE) is not module:
        raise UnsupportedAdapterVersion("unexpected TensorDict consolidated module ownership")
    current = vars(module).get(FUNCTION)
    patched = getattr(current, "_jittor_tensordict_transport", None) == POLICY
    original = getattr(current, "_jittor_tensordict_original", None) if patched else current
    if (not isinstance(original, types.FunctionType) or original.__closure__
            or original.__name__ != FUNCTION or original.__module__ != MODULE
            or original.__globals__ is not vars(module)):
        raise UnsupportedAdapterVersion("TensorDict consolidated helper ownership changed")
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("TensorDict consolidated helper source required") from error
    if hashlib.sha256(source.encode()).hexdigest() != FUNCTION_SHA256:
        raise UnsupportedAdapterVersion("TensorDict consolidated helper source SHA256 changed")
    expected = _rewrite(original, source, four_axis)
    if patched:
        if (current.__code__ != expected.__code__ or current.__defaults__ != expected.__defaults__
                or current.__kwdefaults__ != expected.__kwdefaults__
                or current.__globals__ is not original.__globals__
                or getattr(current, "_jittor_tensordict_four_axis", None) is not four_axis
                or tuple(c.cell_contents for c in (current.__closure__ or ())) !=
                tuple(c.cell_contents for c in (expected.__closure__ or ()))):
            raise UnsupportedAdapterVersion("owned TensorDict consolidated replacement changed")
        return []
    return [(FUNCTION, current, expected)]


@required_patch
def patch_transport(module):
    plans = prepare(module)
    if not plans:
        current = vars(module)[FUNCTION]
        _check_owned_handle(current, module, FUNCTION, current._jittor_tensordict_original)
        return False
    from jittor.compat.module_patcher import patch_method
    for name, original, replacement in plans:
        replacement._jittor_tensordict_patch_handle = patch_method(module, name, replacement, expected=original)
        replace_bound_aliases("tensordict", name, original, replacement)
    return True


@required_patch
def patch_deferred_transport(module):
    root = sys.modules.get("tensordict")
    if root is not None and "__version__" not in vars(root):
        return False
    return patch_transport(module)


def patch_loaded_transport():
    module = sys.modules.get(MODULE)
    return False if module is None else patch_transport(module)
