"""Host-only compile and behavior contract for the ACL data-channel boundary."""

import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "python/jittor/extern/acl/aclops/acl_data_channel.h"
SRC_INCLUDE = ROOT / "python/jittor/src"


def _compile(source, output=None):
    with tempfile.TemporaryDirectory(prefix="jittor-acl-data-") as directory:
        probe = Path(directory) / "probe.cc"
        probe.write_text(source, encoding="utf-8")
        command = [
            os.environ.get("CXX", "g++"), "-std=c++14", "-I", str(ROOT),
            "-I", str(SRC_INCLUDE),
        ]
        if output is None:
            command += ["-fsyntax-only"]
        else:
            command += ["-o", str(output)]
        command.append(str(probe))
        subprocess.run(command, check=True, stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, text=True)


def test_acl_data_channel_header_is_cann_free_and_compilable():
    text = HEADER.read_text(encoding="utf-8")
    assert "#include <acl/" not in text
    assert "AclDataRecord" in text
    assert "AclDecodedData decode_acl_data" in text
    _compile('#include "python/jittor/extern/acl/aclops/acl_data_channel.h"\n')


def test_acl_data_channel_decodes_defaults_and_has_stable_key():
    source = r'''
#include "python/jittor/extern/acl/aclops/acl_data_channel.h"
#include <string>
int main() {
    using namespace jittor::acl_data;
    AclAttrSchema schema;
    AclAttrField dim;
    dim.type = AclDataType::int64;
    schema.emplace("dim", dim);
    AclAttrField keep;
    keep.type = AclDataType::boolean;
    keep.required = false;
    keep.has_default = true;
    keep.default_value = AclDataValue::bool_value_of(false);
    schema.emplace("keepdim", keep);
    AclDataRecord record;
    record.op = "Softmax";
    record.fields.emplace("dim", AclDataValue::int64_value(-1));
    std::string key;
    auto result = decode_acl_data(record, "Softmax", schema, key);
    if (result.fields.at("keepdim").bool_value) return 1;
    if (key != result.cache_key) return 2;
    if (key.find("0x") != std::string::npos) return 3;
    try {
        record.fields.emplace("unknown", AclDataValue::int64_value(1));
        decode_acl_data(record, "Softmax", schema, key);
        return 4;
    } catch (const jittor::UserError&) {
        return 0;
    }
}
'''
    with tempfile.TemporaryDirectory(prefix="jittor-acl-data-run-") as directory:
        binary = Path(directory) / "probe"
        _compile(source, binary)
        subprocess.run([str(binary)], check=True)

