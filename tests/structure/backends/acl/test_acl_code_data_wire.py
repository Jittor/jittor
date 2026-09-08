"""Python encoding is consumed by the actual C++ decoder, without CANN/JIT."""

import copy
import importlib.util
import math
from pathlib import Path
import shutil
import struct
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[4]
SCHEMA = {
    "a_i": {"type": "int64"},
    "b_f": {"type": "float64"},
    "c_b": {"type": "bool"},
    "d_is": {"type": "int64[]"},
    "e_fs": {"type": "float64[]"},
    "f_bs": {"type": "bool[]"},
    "z_default": {"type": "int64", "default": -7},
}


@pytest.fixture(scope="module")
def codec():
    spec = importlib.util.spec_from_file_location(
        "acl_wire_python", ROOT / "backends/acl/kernels/ops/acl_data.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = r"""
#include "backends/acl/include/aclops/acl_code_data.h"
#include <cstring>
#include <iostream>
#include <unordered_map>
using namespace jittor::acl_data;
uint64_t read_number(unsigned bytes) {
    uint64_t value = 0;
    for (unsigned i = 0; i < bytes; ++i) {
        char byte;
        if (!std::cin.get(byte)) throw std::runtime_error("truncated probe input");
        value |= uint64_t(static_cast<unsigned char>(byte)) << (i * 8);
    }
    return value;
}
void write_number(uint64_t value, unsigned bytes) {
    for (unsigned i = 0; i < bytes; ++i) std::cout.put(char(value >> (i * 8)));
}
void dump_float(double value) {
    uint64_t bits;
    static_assert(sizeof(value) == sizeof(bits), "requires float64");
    std::memcpy(&bits, &value, sizeof(bits));
    write_number(bits, 8);
}
int main(int argc, char** argv) {
    try {
        std::unordered_map<std::string, double> data;
        uint32_t count = read_number(4);
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t size = read_number(4);
            std::string name(size, '\0');
            std::cin.read(&name[0], size);
            uint64_t bits = read_number(8);
            double value;
            std::memcpy(&value, &bits, sizeof(value));
            data.emplace(name, value);
        }
        auto before = data;
        AclAttrSchema schema;
        const char* names[] = {"a_i", "b_f", "c_b", "d_is", "e_fs", "f_bs"};
        for (int i = 0; i < 6; ++i) {
            AclAttrField field;
            field.type = static_cast<AclDataType>(i);
            schema.emplace(names[i], field);
        }
        AclAttrField defaulted;
        defaulted.type = AclDataType::int64;
        defaulted.required = false;
        defaulted.has_default = true;
        defaulted.default_value = AclDataValue::int64_value(-7);
        if (argc > 3) defaulted.default_value = AclDataValue::bool_value_of(true);
        schema.emplace("z_default", defaulted);
        auto decoded = decode_code_data(data, argv[1], schema, argv[2]);
        for (const auto& item : before) {
            auto found = data.find(item.first);
            if (found == data.end() || std::memcmp(&item.second, &found->second, sizeof(double))) return 5;
        }
        write_number(decoded.fields.size(), 4);
        for (const auto& item : decoded.fields) {
            write_number(item.first.size(), 4);
            std::cout.write(item.first.data(), item.first.size());
            const auto& value = item.second;
            write_number(unsigned(value.type), 4);
            switch (value.type) {
                case AclDataType::int64: write_number(uint64_t(value.int_value), 8); break;
                case AclDataType::float64: dump_float(value.float_value); break;
                case AclDataType::boolean: write_number(value.bool_value, 8); break;
                case AclDataType::int64_vector:
                    write_number(value.int_values.size(), 4);
                    for (auto number : value.int_values) write_number(uint64_t(number), 8);
                    break;
                case AclDataType::float64_vector:
                    write_number(value.float_values.size(), 4);
                    for (auto number : value.float_values) dump_float(number);
                    break;
                case AclDataType::bool_vector:
                    write_number(value.bool_values.size(), 4);
                    for (bool number : value.bool_values) write_number(number, 8);
                    break;
            }
        }
        return 0;
    } catch (const jittor::UserError& error) {
        std::cerr << "USER:" << error.what(); return 2;
    } catch (const jittor::InternalInvariantError& error) {
        std::cerr << "INTERNAL:" << error.what(); return 3;
    }
}
"""


@pytest.fixture(scope="module")
def decoder(tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("backend prerequisite: C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("acl-wire")
    source, binary = directory / "wire.cc", directory / "wire"
    source.write_text(DRIVER)
    result = subprocess.run(
        [
            compiler,
            "-std=c++14",
            "-I",
            str(ROOT),
            "-I",
            str(ROOT / "src"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return binary


def _record(integer=0, floating=1.25, empty=False):
    return {
        "schema_version": 1,
        "op": "Wire",
        "fields": {
            "a_i": {"type": "int64", "value": integer},
            "b_f": {"type": "float64", "value": floating},
            "c_b": {"type": "bool", "value": True},
            "d_is": {
                "type": "int64[]",
                "value": [] if empty else [-(1 << 63), (1 << 63) - 1, 2**53 + 1, -1],
            },
            "e_fs": {"type": "float64[]", "value": [] if empty else [-0.0, 1e300, -1e-300]},
            "f_bs": {"type": "bool[]", "value": [] if empty else [True, False]},
        },
    }


def _decode(binary, wire, op="Wire", prefix="acl_attr.", invalid_default=False):
    payload = struct.pack("<I", len(wire))
    for key, value in wire.items():
        encoded = key.encode()
        payload += struct.pack("<I", len(encoded)) + encoded + struct.pack("<d", value)
    arguments = [str(binary), op, prefix] + (["invalid-default"] if invalid_default else [])
    return subprocess.run(arguments, input=payload, capture_output=True)


def _unpack(payload):
    offset = 0

    def number(format):
        nonlocal offset
        (value,) = struct.unpack_from(format, payload, offset)
        offset += struct.calcsize(format)
        return value

    result = {}
    for _ in range(number("<I")):
        length = number("<I")
        name = payload[offset : offset + length].decode()
        offset += length
        tag = number("<I")
        format = ("<q", "<d", "<Q")[tag % 3]
        values = [number(format) for _ in range(number("<I"))] if tag >= 3 else number(format)
        if tag % 3 == 2:
            values = [bool(value) for value in values] if tag >= 3 else bool(values)
        result[name] = values
    assert offset == len(payload)
    return result


@pytest.mark.parametrize("integer", [-(1 << 63), (1 << 63) - 1, -(2**53 + 1), 2**53 + 1, 0])
@pytest.mark.parametrize("empty", [False, True])
def test_python_to_cpp_roundtrip_preserves_all_scalar_and_vector_types(
    codec, decoder, integer, empty
):
    record = _record(integer, -0.0, empty)
    original = copy.deepcopy(record)
    wire = codec.encode_code_data(record, expected_op="Wire", schema=SCHEMA)
    assert record == original and all(type(value) is float for value in wire.values())
    wire.update({"outside.nan": float("nan"), "other": 123.0})
    decoded = _decode(decoder, wire)
    assert decoded.returncode == 0, decoded.stderr
    values = _unpack(decoded.stdout)
    expected = codec.validate_acl_data(record, schema=SCHEMA)["fields"]
    for name, entry in expected.items():
        assert values[name] == entry["value"]
    assert struct.pack("<d", values["b_f"]) == struct.pack("<d", -0.0)
    if not empty:
        assert struct.pack("<d", values["e_fs"][0]) == struct.pack("<d", -0.0)


def test_custom_prefix_operator_and_cpp_defaults(codec, decoder):
    record = _record()
    record["op"] = "Wire/算子"
    wire = codec.encode_code_data(record, prefix="custom.")
    decoded = _decode(decoder, wire, op=record["op"], prefix="custom.")
    assert decoded.returncode == 0, decoded.stderr
    assert _unpack(decoded.stdout)["z_default"] == -7


@pytest.mark.parametrize(
    "type_tag,value",
    [("int64", 1 << 63), ("int64", -(1 << 63) - 1), ("int64[]", [1 << 63]), ("float64", 10**1000)],
)
def test_python_rejects_out_of_range_values_and_classifies_bad_defaults(codec, type_tag, value):
    record = {
        "schema_version": 1,
        "op": "Wire",
        "fields": {"x": {"type": type_tag, "value": value}},
    }
    with pytest.raises(codec.AclDataUserError):
        codec.validate_acl_data(record)
    with pytest.raises(codec.AclDataInternalError, match="default"):
        codec.validate_acl_data(
            {"schema_version": 1, "op": "Wire"}, schema={"x": {"type": type_tag, "default": value}}
        )


def test_python_field_tag_must_match_schema(codec):
    with pytest.raises(codec.AclDataUserError, match="type.*schema"):
        codec.validate_acl_data(
            {"schema_version": 1, "op": "Wire", "fields": {"x": {"type": "float64", "value": 1.0}}},
            schema={"x": {"type": "int64"}},
        )


def test_python_schema_and_header_errors_keep_their_error_domain(codec):
    with pytest.raises(codec.AclDataInternalError, match="both required and defaulted"):
        codec.validate_acl_data(
            {"schema_version": 1, "op": "Wire"},
            schema={"x": {"type": "int64", "required": True, "default": 0}},
        )
    with pytest.raises(codec.AclDataUserError, match="version"):
        codec.encode_code_data({"schema_version": True, "op": "Wire"})
    with pytest.raises(codec.AclDataInternalError, match="prefix"):
        codec.encode_code_data(_record(), prefix="")


@pytest.mark.parametrize(
    "kind",
    [
        "unknown",
        "missing",
        "type",
        "order",
        "version",
        "op",
        "fraction",
        "wide_lane",
        "negative_lane",
        "boolean",
        "nan",
        "vector_length",
    ],
)
def test_cpp_rejects_malformed_wire_payloads(codec, decoder, kind):
    wire = codec.encode_code_data(_record(), schema=SCHEMA)
    if kind == "unknown":
        wire["acl_attr.surprise"] = 1.0
    elif kind == "missing":
        del wire["acl_attr.field.0.hi"]
    elif kind == "type":
        wire["acl_attr.field.0.name.615f69"] = 1.0
    elif kind == "order":
        left, right = "acl_attr.field.0.", "acl_attr.field.1."
        wire = {
            (
                right + key[len(left) :]
                if key.startswith(left)
                else left + key[len(right) :]
                if key.startswith(right)
                else key
            ): value
            for key, value in wire.items()
        }
    elif kind == "version":
        wire["acl_attr.version"] = 2.0
    elif kind == "op":
        wire["acl_attr.op.4f74686572"] = wire.pop("acl_attr.op.57697265")
    elif kind == "fraction":
        wire["acl_attr.field.0.lo"] = 0.5
    elif kind == "wide_lane":
        wire["acl_attr.field.0.hi"] = float(1 << 32)
    elif kind == "negative_lane":
        wire["acl_attr.field.0.lo"] = -1.0
    elif kind == "boolean":
        wire["acl_attr.field.2.value"] = 2.0
    elif kind == "nan":
        wire["acl_attr.field.1.value"] = math.nan
    elif kind == "vector_length":
        wire["acl_attr.field.3.length"] = 4294967295.0
    result = _decode(decoder, wire)
    assert result.returncode == 2, result.stderr
    assert result.stderr.startswith(b"USER:")
    if kind == "order":
        assert b"order" in result.stderr


def test_cpp_schema_default_errors_are_internal(codec, decoder):
    result = _decode(
        decoder, codec.encode_code_data(_record(), schema=SCHEMA), invalid_default=True
    )
    assert result.returncode == 3 and result.stderr.startswith(b"INTERNAL:")
