"""Opt-in public jagged-position construction for exact verl 0.9.1.

Downstream-private ownership: padding.py constructs multi-axis positions using
as_nested_tensor's first ragged axis; BaseEngine subsequently invokes a helper
that changes only private metadata. This policy constructs the requested axis
through public Torch APIs instead and replaces that workaround with validation and narrowly proven transport reconstruction.
It never mutates framework objects or installs an operator. Candidate owned
patches and the independent native oracle use the same prepare() output.

Upstream: verl-project/verl 1876b06d0a3e4e71e06230be10af14492ca8a75b.
Default off. Host contracts are not device or PPO numerical acceptance.
Revalidate when upstream helpers, public nested APIs, or module-patcher change.
"""
import ast
from functools import update_wrapper
import hashlib
import inspect
import os
from pathlib import Path
import sys
import types

from .._common import (require_version, required_patch, replace,
                      replace_bound_aliases, UnsupportedAdapterVersion)


SUPPORTED_VERSIONS = frozenset(("0.9.1",))
FLAG = "VERL_CORRECT_JAGGED_POSITIONS"
POLICY = "verl_v091_public_jagged_position_construction_v2"
TU_MODULE = "verl.utils.tensordict_utils"
PADDING_MODULE = "verl.workers.utils.padding"
MODULE_SHA256 = {
    TU_MODULE: "4dc4c321a51e256232751ddb5cb2a2be129076e7f02f36a685fcad6d0d98f5f1",
    PADDING_MODULE: "4e29a4af7d6f7a2182cbd45536df5c3423745e925d0cd3bb151440dbfba8fbcc",
}
FUNCTION_SHA256 = {
    "nested_tensor_from_tensor_list": "470ebbd5276a24fdceebc62d093619d0054f2eb63f9c3209128f2e1bd18cd8a0",
    "maybe_fix_3d_position_ids": "dc8fa98190805633010a81a926e8d296da2a0b136f5db61160bb4c39d2de9397",
    "left_right_2_no_padding": "8b09e2ef9bb544fa95661223f7d002e6cd8ec76b250b0ca6335916d5082acf87",
}


def enabled():
    value = os.environ.get(FLAG, "0")
    if value not in ("0", "1"):
        raise UnsupportedAdapterVersion(FLAG + " must be absent, 0 or 1")
    return value == "1"


def _offset_list(tensor):
    offsets = tensor.offsets()
    if offsets.device != tensor.values().device:
        raise ValueError("jagged position offsets and values must share a device")
    if offsets.dim() != 1:
        raise ValueError("jagged position offsets must be one-dimensional")
    result = offsets.detach().cpu().tolist()
    if (len(result) < 2 or any(type(value) is not int for value in result)
            or result[0] != 0 or any(a > b for a, b in zip(result, result[1:]))):
        raise ValueError("jagged position offsets must be zero-based nondecreasing integers")
    return result


def _ragged_dimension(size, torch):
    # Public shape representations: native Torch SymInt; current frontend -1.
    # Never infer a ragged axis from an arbitrary mismatching integer dimension.
    if isinstance(size, int):
        return type(size) is int and size == -1
    return isinstance(size, torch.SymInt)


def _contiguous_lengths(tensor, offsets):
    lengths = tensor.lengths()
    if lengths is not None:
        if lengths.dim() != 1 or lengths.device != tensor.values().device:
            raise ValueError("jagged position lengths must be one-dimensional and share the values device")
        observed = lengths.detach().cpu().tolist()
        if (any(type(value) is not int for value in observed)
                or observed != [b - a for a, b in zip(offsets, offsets[1:])]):
            raise ValueError("jagged positions must have contiguous lengths matching offsets")


def _validate_positions(data, torch):
    """Validate normal layout; reconstruct only proven four-axis TD transport loss.

    The narrow stale public shape is (B, ragged, packed_T), with unchanged
    integer values (4, packed_T) and full token offsets independently confirmed
    by input_ids. Replacement is a new public nested view; the old object and
    buffers are never mutated. This is not a generic axis-guessing repair.
    """
    if "position_ids" not in data.keys():
        return
    positions = data["position_ids"]
    if not positions.is_nested or positions.dim() != 3:
        return
    values = positions.values()
    offsets = _offset_list(positions)
    shape = positions.shape
    if (values.dim() != 2 or len(shape) != 3 or type(shape[0]) is not int
            or shape[0] != len(offsets) - 1 or offsets[-1] != values.shape[1]):
        raise ValueError("multi-axis positions require public jagged_dim=2 with consistent shape/values/offsets")
    _contiguous_lengths(positions, offsets)
    normal = (type(shape[1]) is int and shape[1] == values.shape[0]
              and _ragged_dimension(shape[2], torch))
    if normal:
        if "input_ids" in data.keys() and data["input_ids"].is_nested:
            ids = data["input_ids"]
            if ids.dim() != 2 or _offset_list(ids) != offsets:
                raise ValueError("jagged position token offsets differ from input_ids")
        return
    # Known TensorDict consolidated-pickle reconstruction defaults to axis 1.
    # Packed positions still have four fixed axes and packed_T tokens. A wrong
    # original axis-1 constructor (e.g. values[8,38], offsets end=8) fails above.
    if (not _ragged_dimension(shape[1], torch) or type(shape[2]) is not int
            or shape[2] != values.shape[1] or values.shape[0] != 4):
        raise ValueError("multi-axis positions require public jagged_dim=2; unsupported stale public shape")
    integer_dtypes = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    if (positions.dtype not in integer_dtypes or values.dtype != positions.dtype
            or positions.requires_grad is not False or values.requires_grad is not False
            or not values.is_contiguous()):
        raise ValueError("jagged stale positions require contiguous integer values without gradients")
    if "input_ids" not in data.keys():
        raise ValueError("jagged stale positions require full input_ids token-offset contract")
    ids = data["input_ids"]
    if not ids.is_nested or ids.dim() != 2:
        raise ValueError("jagged stale positions require nested rank-two input_ids")
    id_values, id_shape = ids.values(), ids.shape
    if (id_values.dim() != 1 or len(id_shape) != 2 or type(id_shape[0]) is not int
            or id_shape[0] != shape[0] or not _ragged_dimension(id_shape[1], torch)
            or _offset_list(ids) != offsets or id_values.shape[0] != offsets[-1]
            or ids.dtype not in integer_dtypes or id_values.dtype != ids.dtype
            or ids.requires_grad is not False or id_values.requires_grad is not False
            or ids.device != positions.device or not id_values.is_contiguous()):
        raise ValueError("jagged stale position token contract differs from input_ids")
    _contiguous_lengths(ids, offsets)
    replacement = torch.nested.nested_tensor_from_jagged(
        values, offsets=positions.offsets(), lengths=positions.lengths(), jagged_dim=2)
    # A bad public implementation must fail before changing the downstream TD.
    new_shape = replacement.shape
    if (replacement.device != positions.device or replacement.dtype != positions.dtype
            or replacement.requires_grad is not False or replacement.dim() != 3
            or type(new_shape[0]) is not int or new_shape[0] != shape[0]
            or type(new_shape[1]) is not int or new_shape[1] != 4
            or not _ragged_dimension(new_shape[2], torch)
            or replacement.values().data_ptr() != values.data_ptr()
            or replacement.offsets().data_ptr() != positions.offsets().data_ptr()):
        raise ValueError("jagged public reconstruction did not preserve the validated buffers and layout")
    data["position_ids"] = replacement


class _Rewrite(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name
        self.calls = 0
        self.assignments = 0

    def visit_Call(self, node):
        self.generic_visit(node)
        spelling = ast.unparse(node.func)
        if (self.name == "nested_tensor_from_tensor_list"
                and spelling == "torch.nested.nested_tensor_from_jagged"):
            if any(keyword.arg == "jagged_dim" for keyword in node.keywords):
                raise UnsupportedAdapterVersion("upstream jagged_dim already specified")
            node.keywords.append(ast.keyword(arg="jagged_dim", value=ast.Name(id="ragged_idx", ctx=ast.Load())))
            self.calls += 1
        elif (self.name == "left_right_2_no_padding"
              and spelling == "torch.nested.as_nested_tensor"):
            expected = ast.parse("torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)", mode="eval").body
            if ast.dump(node) != ast.dump(expected):
                raise UnsupportedAdapterVersion("position constructor expression changed")
            node = ast.copy_location(ast.parse(
                "tu.nested_tensor_from_tensor_list(position_ids_list, ragged_idx=position_ids_list[0].dim())",
                mode="eval").body, node)
            self.calls += 1
        return node

    def visit_Assign(self, node):
        if self.name == "nested_tensor_from_tensor_list":
            expected = ast.parse("nested_tensor._ragged_idx = ragged_idx").body[0]
            if ast.dump(node) == ast.dump(expected):
                self.assignments += 1
                return None
        return self.generic_visit(node)


def _clone(original):
    if not isinstance(original, types.FunctionType) or original.__closure__:
        raise UnsupportedAdapterVersion("jagged helper must be an unmodified module-level function")
    try:
        lines, start = inspect.getsourcelines(original)
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("jagged helper source is required") from error
    source = "".join(lines)
    name = original.__name__
    if hashlib.sha256(source.encode()).hexdigest() != FUNCTION_SHA256[name]:
        raise UnsupportedAdapterVersion("jagged helper source SHA256 changed: " + name)
    tree = ast.parse(source)
    if (len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef)
            or tree.body[0].name != name or tree.body[0].decorator_list):
        raise UnsupportedAdapterVersion("jagged helper definition changed")
    function = tree.body[0]
    if name == "maybe_fix_3d_position_ids":
        assignments = [node for node in ast.walk(function)
                       if isinstance(node, ast.Assign) and ast.dump(node) ==
                       ast.dump(ast.parse('data["position_ids"]._ragged_idx = 2').body[0])]
        if len(assignments) != 1:
            raise UnsupportedAdapterVersion("position workaround assignment count changed")
        function.body = ast.parse("_jittor_validate_positions(data, torch)").body
    else:
        rewrite = _Rewrite(name)
        rewrite.visit(function)
        expected = (1, 1) if name == "nested_tensor_from_tensor_list" else (1, 0)
        if (rewrite.calls, rewrite.assignments) != expected:
            raise UnsupportedAdapterVersion("jagged constructor rewrite count changed")
    # Do not reevaluate upstream annotations/default objects during cloning.
    function.returns = None
    for argument in (list(function.args.posonlyargs) + function.args.args
                     + function.args.kwonlyargs + [function.args.vararg, function.args.kwarg]):
        if argument is not None:
            argument.annotation = None
    function.args.defaults = [ast.Constant(None) for _ in function.args.defaults]
    function.args.kw_defaults = [None if item is None else ast.Constant(None)
                                for item in function.args.kw_defaults]
    ast.fix_missing_locations(function)
    ast.increment_lineno(function, start - 1)
    factory = ast.parse("def _jittor_jagged_factory(_jittor_validate_positions):\n    pass\n")
    factory.body[0].body = [function, ast.Return(ast.Name(id=name, ctx=ast.Load()))]
    compiled = compile(ast.fix_missing_locations(factory), original.__code__.co_filename, "exec")
    code = next(item for item in compiled.co_consts if isinstance(item, types.CodeType))
    replacement = types.FunctionType(code, original.__globals__)(_validate_positions)
    replacement.__defaults__ = original.__defaults__
    replacement.__kwdefaults__ = original.__kwdefaults__
    update_wrapper(replacement, original)
    replacement._jittor_verl_original = original
    replacement._jittor_verl_jagged_positions = POLICY
    return replacement


def prepare(module):
    """Return fully validated (name, current, replacement) plans; no mutation/JT import."""
    if not enabled():
        raise UnsupportedAdapterVersion("jagged positions policy requires explicit opt-in")
    require_version("verl", SUPPORTED_VERSIONS)
    if module.__name__ not in MODULE_SHA256 or sys.modules.get(module.__name__) is not module:
        raise UnsupportedAdapterVersion("unexpected jagged-position module ownership")
    try:
        filename = inspect.getsourcefile(module)
        content = Path(filename).read_bytes() if filename else b""
    except (OSError, TypeError) as error:
        raise UnsupportedAdapterVersion("jagged-position module source is required") from error
    if hashlib.sha256(content).hexdigest() != MODULE_SHA256[module.__name__]:
        raise UnsupportedAdapterVersion("jagged-position module source SHA256 changed")
    if module.__name__ == TU_MODULE:
        names = ("nested_tensor_from_tensor_list", "maybe_fix_3d_position_ids")
    else:
        helper = vars(module).get("tu")
        if (not isinstance(helper, types.ModuleType)
                or helper is not sys.modules.get(TU_MODULE) or prepare(helper)):
            raise UnsupportedAdapterVersion("corrected TensorDict helper must precede padding patch")
        names = ("left_right_2_no_padding",)
    plans = []
    for name in names:
        current = vars(module).get(name)
        if (not isinstance(current, types.FunctionType) or current.__module__ != module.__name__
                or current.__globals__ is not vars(module) or current.__name__ != name):
            raise UnsupportedAdapterVersion("jagged helper ownership changed: " + name)
        if getattr(current, "_jittor_verl_jagged_positions", None) == POLICY:
            original = getattr(current, "_jittor_verl_original", None)
            if (not isinstance(original, types.FunctionType) or original.__module__ != module.__name__
                    or original.__globals__ is not vars(module) or original.__name__ != name):
                raise UnsupportedAdapterVersion("original jagged helper ownership changed: " + name)
            expected = _clone(original)
            if (current.__code__ != expected.__code__ or current.__defaults__ != expected.__defaults__
                    or tuple(cell.cell_contents for cell in (current.__closure__ or ())) !=
                    tuple(cell.cell_contents for cell in (expected.__closure__ or ()))):
                raise UnsupportedAdapterVersion("owned jagged replacement changed: " + name)
        else:
            plans.append((name, current, _clone(current)))
    return plans


@required_patch
def patch_module(module):
    if not enabled():
        return False
    plans = prepare(module)
    for name, original, replacement in plans:
        replace(module, name, replacement, original)
        replace_bound_aliases("verl", name, original, replacement)
    return bool(plans)


@required_patch
def patch_root(module):
    if not enabled():
        return False
    if module.__name__ != "verl" or sys.modules.get("verl") is not module:
        raise UnsupportedAdapterVersion("unexpected verl root ownership")
    require_version("verl", SUPPORTED_VERSIONS)
    helper = sys.modules.get(TU_MODULE)
    if not isinstance(helper, types.ModuleType):
        raise UnsupportedAdapterVersion("verl root did not load its TensorDict helper")
    return patch_module(helper)


def register(register_module_patch):
    # The helper loads inside verl root before root __version__ exists. Apply
    # only after root completion; never substitute distribution metadata.
    if enabled():
        register_module_patch("verl", patch_root)
        register_module_patch(PADDING_MODULE, patch_module)
