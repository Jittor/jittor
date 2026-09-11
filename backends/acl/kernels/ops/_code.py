"""Assembled CodeOp programs for the ACL runners.

Everything a CodeOp carries besides its Vars -- the C++ source, the gradient
programs and the encoded attribute payload -- is a pure function of the runner
name, the input and output counts and the attribute values. None of it depends
on which Vars get wired up, yet it used to be rebuilt for every graph node:
string concatenation, a dict copy per data lane and a payload merge, all to
produce bytes identical to the previous construction.

:func:`acl_program` performs that assembly once and memoises it;
:func:`acl_emit` does the per-call part, which is a single ``jt.code``.
:func:`acl_code` is the two together and keeps the old signature. A builder
that can name its program with a couple of scalars -- matmul with its
transpose flag, conv with its stride tuple -- should call the pair directly
and skip re-deriving the key from freshly built attribute dicts.
"""

from jittor._core.dtypes import dtype_name as _jittor_dtype_name
import jittor as jt

from ._attributes import (
    AttributeCode,
    _cache_key as _value_key,
    attribute_data,
    attribute_payloads,
    code_program,
)

# The float dtypes the ACL kernels accept. This is the set adamw_op.py and
# getitem_op.py already declare, and the norm kernels in this directory are
# literally named grouped_bfloat16_rms_norm, so bf16 already reaches ACL
# unconverted from several other ops here.
ACL_FLOAT_DTYPES = ("float16", "bfloat16", "float32")

#: The same set, keyed for a membership test on the *raw* spelling. A Var's
#: dtype is a native NanoString that prints its canonical name, so a guard is
#: one ``str`` plus one set lookup and only an unusual spelling has to reach
#: ``dtype_name``.
ACL_FLOAT_DTYPE_SET = frozenset(ACL_FLOAT_DTYPES)

_ATTRIBUTE_INCLUDE = '\n#include "aclops/acl_code_attributes.h"\n'
_ACLOPS_INCLUDE = """
    #include "aclops/aclops.h"
    """


def code_with_attributes(*args, **kwargs):
    """Forward structural CodeOp programs and merge their typed attribute data."""
    fragments = []
    attribute_sets = kwargs.pop("attribute_sets", None)
    for key in ("cuda_src", "cpu_src"):
        value = kwargs.get(key)
        if isinstance(value, AttributeCode):
            fragments.append(value)
            kwargs[key] = value.source
    for key in ("cuda_grad_src", "cpu_grad_src"):
        values = kwargs.get(key)
        if values:
            fragments.extend(value for value in values if isinstance(value, AttributeCode))
            kwargs[key] = [
                value.source if isinstance(value, AttributeCode) else value for value in values
            ]
    if fragments:
        merged = code_program(fragments).data
        data = dict(kwargs.get("data") or {})
        if set(data) & set(merged):
            raise ValueError("CodeOp data conflicts with typed ACL attributes")
        data.update(merged)
        kwargs["data"] = data
        kwargs["cuda_header"] = kwargs.get("cuda_header", "") + _ATTRIBUTE_INCLUDE
    if attribute_sets:
        data = dict(kwargs.get("data") or {})
        data.update(attribute_payloads(attribute_sets))
        kwargs["data"] = data
        kwargs["cuda_header"] = kwargs.get("cuda_header", "") + _ATTRIBUTE_INCLUDE
    return jt.code(*args, **kwargs)


#: Canonical names by the raw dtype name ``dtype_name`` would read. That
#: function is a pure function of that one string and rebuilds its alias table
#: from a literal on every call, while this check runs once per activation,
#: norm and product. `jittor._runtime.dispatch` keeps the same memo for kernel
#: selection; this module cannot borrow it, because the ACL structure tests
#: load these op modules on their own, with no jittor runtime behind them.
_DTYPE_NAMES = {}


def canonical_dtype_name(dtype):
    """``dtype_name(dtype)``, memoised on the raw name it would have read."""
    raw = getattr(dtype, "name", None)
    if raw.__class__ is not str:
        raw = getattr(dtype, "__name__", None)
        if raw.__class__ is not str:
            raw = str(dtype)
    name = _DTYPE_NAMES.get(raw)
    if name is None:
        name = _jittor_dtype_name(dtype)
        _DTYPE_NAMES[raw] = name
    return name


def scalar_key(value):
    """Dict-key form of a scalar that keeps -0.0 apart from 0.0.

    Python hashes and compares 0.0, -0.0 and 0 alike, but they do not all
    reach a kernel as the same double, so a falsy scalar carries its repr.
    """
    return value if value else repr(value)


def check_acl_float_dtype(x, op_name):
    """Reject an unsupported dtype instead of quietly widening it. 6.B11.

    Six ops used to open with ``x = x.float32()``. That is not a conversion for
    the kernel's benefit: the result var keeps the promoted dtype, so a bf16 or
    fp16 model silently became fp32 at that point and stayed fp32 for the rest
    of the graph -- disagreeing with torch, costing bandwidth, and reported
    nowhere. Declaring what is supported and failing on the rest is the
    behaviour the other 28 op files in this directory already have.
    """
    if str(x.dtype) in ACL_FLOAT_DTYPE_SET:
        return x
    dtype = canonical_dtype_name(x.dtype)
    if dtype not in ACL_FLOAT_DTYPE_SET:
        raise TypeError(
            "{} on ACL supports {}, got {}".format(
                op_name, "/".join(ACL_FLOAT_DTYPES), dtype
            )
        )
    return x


GRAD_ATTRIBUTE_PREFIX = "acl_grad_attr."


def _insert_before_run(program, injection):
    """Place generated code ahead of a runner program's `op.run();` call.

    A runner reads `op_attr` inside run(), so appending the attribute
    application after the program would launch the operator with no attributes
    at all. Every ACL gradient program ends in that call.
    """
    source = program.source if isinstance(program, AttributeCode) else str(program)
    marker = "op.run();"
    index = source.rfind(marker)
    if index < 0:
        raise ValueError("ACL gradient program does not call op.run()")
    head = AttributeCode(source[:index], dict(getattr(program, "data", {}) or {}))
    return code_program([head, injection, source[index:]])


class AclProgram:
    """One ACL operator's CodeOp source and data, assembled once.

    An instance is shared between every construction of the same operator, so
    nothing on it may be mutated after :func:`acl_program` hands it out. Both
    containers go straight to ``jt.code``, which copies them into the C++
    ``vector<string>`` and ``unordered_map<string,double>`` it owns.
    """

    __slots__ = ("cuda_header", "cuda_src", "cuda_grad_src", "data")

    def __init__(self, cuda_header, cuda_src, cuda_grad_src, data):
        self.cuda_header = cuda_header
        self.cuda_src = cuda_src
        self.cuda_grad_src = cuda_grad_src
        self.data = data


#: Assembled programs keyed by everything they are derived from. Bounded so a
#: model whose attributes track a changing shape -- slice bounds, `Range` ends
#: -- cannot grow it without limit; dropping entries only costs a rebuild.
_ACL_PROGRAMS = {}
_ACL_PROGRAM_LIMIT = 4096

#: Marks an argument that was not supplied, distinct from any fragment key.
_ABSENT = "\0absent"


def _source_key(value):
    """Identity of one source fragment, or None when it cannot be memoised."""
    if value.__class__ is str:
        return value
    if value.__class__ is AttributeCode:
        return value.key
    return None


def _mapping_key(mapping):
    """Hashable canonical form of an attribute mapping, or None if there is none."""
    if not isinstance(mapping, dict):
        return None
    fields = []
    for field in sorted(mapping):
        value = _value_key(mapping[field])
        if value is None:
            return None
        fields.append((field, value))
    return tuple(fields)


def _program_key(name, input_count, output_count, attr_code, attr_header, extra_data,
                 cuda_grad_src, multi_grad_src, multi_grad_output,
                 multi_grad_input_count, attributes, multi_grad_attributes,
                 attribute_sets):
    """Everything :func:`_build_program` reads, or None when it is not a value.

    A None component means the program cannot be identified by value, and the
    caller then rebuilds it. Returning a key that omitted one of these inputs
    would hand back a program assembled from different attributes.
    """
    attr_key = _source_key(attr_code)
    if attr_key is None:
        return None
    if cuda_grad_src:
        parts = []
        for value in cuda_grad_src:
            part = _source_key(value)
            if part is None:
                return None
            parts.append(part)
        grad_key = tuple(parts)
    else:
        grad_key = _ABSENT
    if multi_grad_src is None:
        multi_key = _ABSENT
    else:
        multi_key = _source_key(multi_grad_src)
        if multi_key is None:
            return None
    if attributes is None:
        attributes_key = _ABSENT
    else:
        attributes_key = _mapping_key(attributes)
        if attributes_key is None:
            return None
    if multi_grad_attributes is None:
        grad_attributes_key = _ABSENT
    else:
        grad_attributes_key = _mapping_key(multi_grad_attributes)
        if grad_attributes_key is None:
            return None
    if attribute_sets:
        if not isinstance(attribute_sets, dict):
            return None
        parts = []
        for slot in sorted(attribute_sets):
            entry = attribute_sets[slot]
            if not isinstance(entry, (tuple, list)) or len(entry) != 2:
                return None
            set_key = _mapping_key(entry[1])
            if set_key is None:
                return None
            parts.append((slot, entry[0], set_key))
        sets_key = tuple(parts)
    else:
        sets_key = _ABSENT
    if extra_data:
        extra_key = _mapping_key(extra_data)
        if extra_key is None:
            return None
    else:
        extra_key = _ABSENT
    return (name, input_count, output_count, attr_header, attr_key, grad_key, multi_key,
            multi_grad_output, multi_grad_input_count, attributes_key,
            grad_attributes_key, sets_key, extra_key)


def acl_program(
    name,
    input_count,
    output_count,
    attr_code="",
    attr_header="",
    extra_data=None,
    cuda_grad_src=None,
    multi_grad_src=None,
    multi_grad_output=0,
    multi_grad_input_count=None,
    attributes=None,
    multi_grad_attributes=None,
    attribute_sets=None,
):
    """Assemble, once per distinct operator, the CodeOp source and data."""
    key = _program_key(name, input_count, output_count, attr_code, attr_header,
                       extra_data, cuda_grad_src, multi_grad_src, multi_grad_output,
                       multi_grad_input_count, attributes, multi_grad_attributes,
                       attribute_sets)
    if key is not None:
        program = _ACL_PROGRAMS.get(key)
        if program is not None:
            return program
    program = _build_program(name, input_count, output_count, attr_code, attr_header,
                             extra_data, cuda_grad_src, multi_grad_src, multi_grad_output,
                             multi_grad_input_count, attributes, multi_grad_attributes,
                             attribute_sets)
    if key is not None:
        if len(_ACL_PROGRAMS) >= _ACL_PROGRAM_LIMIT:
            _ACL_PROGRAMS.clear()
        _ACL_PROGRAMS[key] = program
    return program


def _build_program(name, input_count, output_count, attr_code, attr_header, extra_data,
                   cuda_grad_src, multi_grad_src, multi_grad_output,
                   multi_grad_input_count, attributes, multi_grad_attributes,
                   attribute_sets):
    attr_header = "\nnamespace jittor{" + attr_header + "}\n"
    cuda_header = _ACLOPS_INCLUDE

    input_code = "".join("op.add(in{}, true);\n".format(index) for index in range(input_count))
    output_code = "".join("op.add(out{}, false);\n".format(index) for index in range(output_count))
    data = dict(extra_data or {})
    if attribute_sets:
        if any(key.startswith("acl_payload.") for key in data):
            raise ValueError("extra_data uses the reserved ACL payload namespace")
        data.update(attribute_payloads(attribute_sets))
        cuda_header += _ATTRIBUTE_INCLUDE
    if attributes is not None:
        if attr_code:
            raise ValueError("ACL attributes and generated attr_code are mutually exclusive")
        if any(key.startswith("acl_attr.") for key in data):
            raise ValueError("extra_data uses the reserved ACL attribute namespace")
        data.update(attribute_data(name, attributes))
        cuda_header += _ATTRIBUTE_INCLUDE
        attr_code = 'apply_acl_code_attributes(op, data, "acl_attr.", "' + name + '");'
    if multi_grad_attributes is not None:
        if not multi_grad_src:
            raise ValueError("multi_grad_attributes requires multi_grad_src")
        backward_name = name + "Backward"
        # Forward and gradient programs share one CodeOp data map, so the two
        # records need separate namespaces. Under a single prefix the second
        # encode overwrites `version`/`fields`/`field.N.*` and leaves a second
        # `op.<name>` marker that the first program's decoder then rejects as
        # an unknown key.
        if any(key.startswith(GRAD_ATTRIBUTE_PREFIX) for key in data):
            raise ValueError("extra_data uses the reserved ACL gradient attribute namespace")
        data.update(attribute_data(backward_name, multi_grad_attributes,
                                   prefix=GRAD_ATTRIBUTE_PREFIX))
        cuda_header += _ATTRIBUTE_INCLUDE
        multi_grad_src = _insert_before_run(multi_grad_src, code_program([
            '\n            apply_acl_code_attributes(op, data, "' + GRAD_ATTRIBUTE_PREFIX + '", "',
            backward_name,
            '");\n            ',
        ]))
    if multi_grad_src:
        if cuda_grad_src:
            raise ValueError("ACL code cannot combine multi_grad_src with cuda_grad_src")
        cuda_grad_src = [multi_grad_src]
        data.update({"multi_grad": 1, "multi_grad_output": multi_grad_output})
        if multi_grad_input_count is not None:
            data["multi_grad_input_count"] = multi_grad_input_count

    if isinstance(attr_code, AttributeCode):
        cuda_src = code_program(
            [
                "\n// aclop\n" + name + "OpRunner op;\n",
                input_code,
                output_code,
                attr_code,
                "\nop.run();",
            ]
        )
    else:
        cuda_src = (
            "\n// aclop\n"
            + name
            + "OpRunner op;\n"
            + input_code
            + output_code
            + attr_code
            + "\nop.run();"
        )

    # The merge code_with_attributes performs, done here so its result lands in
    # the memoised program instead of being redone per construction.
    fragments = []
    if isinstance(cuda_src, AttributeCode):
        fragments.append(cuda_src)
        cuda_src = cuda_src.source
    grad_sources = []
    for value in cuda_grad_src or ():
        if isinstance(value, AttributeCode):
            fragments.append(value)
            grad_sources.append(value.source)
        else:
            grad_sources.append(value)
    if fragments:
        merged = code_program(fragments).data
        if set(data) & set(merged):
            raise ValueError("CodeOp data conflicts with typed ACL attributes")
        data.update(merged)
        cuda_header += _ATTRIBUTE_INCLUDE

    return AclProgram(attr_header + cuda_header, cuda_src, grad_sources, data)


def acl_emit(program, inputs, output_dtypes=None, output_shapes=None, outputs=None):
    """Construct the graph node for an already assembled ACL program."""
    if outputs is not None:
        return jt.code(
            outputs=outputs,
            inputs=inputs,
            backend="acl",
            cuda_header=program.cuda_header,
            cuda_src=program.cuda_src,
            cuda_grad_src=program.cuda_grad_src,
            data=program.data,
        )
    return jt.code(
        output_shapes,
        output_dtypes,
        inputs,
        backend="acl",
        cuda_header=program.cuda_header,
        cuda_src=program.cuda_src,
        cuda_grad_src=program.cuda_grad_src,
        data=program.data,
    )


def acl_code(
    name,
    inputs,
    output_dtypes=None,
    output_shapes=None,
    attr_code="",
    attr_header="",
    outputs=None,
    extra_data=None,
    cuda_grad_src=None,
    multi_grad_src=None,
    multi_grad_output=0,
    multi_grad_input_count=None,
    attributes=None,
    multi_grad_attributes=None,
    attribute_sets=None,
):
    if outputs is not None:
        output_count = len(outputs)
    else:
        if output_dtypes is None or output_shapes is None:
            raise ValueError("ACL code requires output_dtypes and output_shapes")
        if len(output_dtypes) != len(output_shapes):
            raise ValueError("ACL code output dtypes and shapes must have equal length")
        output_count = len(output_shapes)
    program = acl_program(
        name,
        len(inputs),
        output_count,
        attr_code=attr_code,
        attr_header=attr_header,
        extra_data=extra_data,
        cuda_grad_src=cuda_grad_src,
        multi_grad_src=multi_grad_src,
        multi_grad_output=multi_grad_output,
        multi_grad_input_count=multi_grad_input_count,
        attributes=attributes,
        multi_grad_attributes=multi_grad_attributes,
        attribute_sets=attribute_sets,
    )
    return acl_emit(program, inputs, output_dtypes, output_shapes, outputs)
