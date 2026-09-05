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
    assert "class AclDataOwner" in text
    assert "class AclDataView" in text
    assert "class AclAttrRunnerContract" in text
    assert "AclAttrBinding" in text
    assert "void consume(const AclDataRecord& record" in text
    assert "const AclAttrSchema& schema() const" in text
    assert "ACL data owner name must be non-empty" in text
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


def test_acl_data_owner_binds_identity_and_schema_for_future_registry():
    source = r'''
#include "python/jittor/extern/acl/aclops/acl_data_channel.h"
int main() {
    using namespace jittor::acl_data;
    AclAttrSchema schema;
    AclAttrField axis;
    axis.type = AclDataType::int64;
    schema.emplace("axis", axis);
    AclAttrField keep;
    keep.type = AclDataType::boolean;
    keep.required = false;
    keep.has_default = true;
    keep.default_value = AclDataValue::bool_value_of(true);
    schema.emplace("keepdim", keep);
    AclDataOwner owner("Softmax", schema);
    if (owner.op() != "Softmax" || owner.schema().size() != 2) return 1;
    AclDataRecord record;
    record.op = "Softmax";
    record.fields.emplace("axis", AclDataValue::int64_value(-1));
    std::string key;
    auto decoded = owner.decode(record, key);
    if (!decoded.fields.at("keepdim").bool_value) return 2;
    if (key != decoded.cache_key || key.find("0x") != std::string::npos) return 3;
    record.op = "Triu";
    try {
        owner.decode(record, key);
        return 4;
    } catch (const jittor::UserError&) {
    }
    try {
        AclDataOwner invalid("", AclAttrSchema());
        return 5;
    } catch (const jittor::InternalInvariantError&) {
        return 0;
    }
}
'''
    with tempfile.TemporaryDirectory(prefix="jittor-acl-owner-run-") as directory:
        binary = Path(directory) / "probe"
        _compile(source, binary)
        subprocess.run([str(binary)], check=True)


def test_acl_data_owner_exposes_validated_read_only_consumer_view():
    source = r'''
#include "python/jittor/extern/acl/aclops/acl_data_channel.h"
#include <vector>
int main() {
    using namespace jittor::acl_data;
    AclAttrSchema schema;
    AclAttrField dim;
    dim.type = AclDataType::int64;
    schema.emplace("dim", dim);
    AclAttrField axes;
    axes.type = AclDataType::int64_vector;
    schema.emplace("axes", axes);
    AclAttrField keep;
    keep.type = AclDataType::boolean;
    keep.required = false;
    keep.has_default = true;
    keep.default_value = AclDataValue::bool_value_of(false);
    schema.emplace("keepdim", keep);
    AclDataOwner owner("Softmax", schema);
    AclDataRecord record;
    record.op = "Softmax";
    record.fields.emplace("dim", AclDataValue::int64_value(-1));
    record.fields.emplace("axes", AclDataValue::int64_vector({1, 3}));
    std::string key;
    int callback_count = 0;
    owner.consume(record, key, [&](const AclDataView& attrs) {
        ++callback_count;
        if (attrs.op() != "Softmax" || attrs.schema_version() != 1) return;
        if (attrs.int64("dim") != -1) return;
        if (attrs.int64_vector("axes") != std::vector<int64_t>({1, 3})) return;
        if (!attrs.has("keepdim") || attrs.boolean("keepdim")) return;
        if (attrs.cache_key() != key || key.find("0x") != std::string::npos) return;
        try {
            attrs.float64("dim");
            return;
        } catch (const jittor::InternalInvariantError&) {
            ++callback_count;
        }
    });
    return callback_count == 2 ? 0 : 1;
}
'''
    with tempfile.TemporaryDirectory(prefix="jittor-acl-consumer-run-") as directory:
        binary = Path(directory) / "probe"
        _compile(source, binary)
        subprocess.run([str(binary)], check=True)


def test_acl_attr_runner_contract_freezes_bindings_before_consumer():
    source = r'''
#include "python/jittor/extern/acl/aclops/acl_data_channel.h"
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
    AclAttrRunnerContract contract("Softmax", schema, {
        {"dim", AclDataType::int64},
        {"keepdim", AclDataType::boolean},
    });
    AclDataRecord record;
    record.op = "Softmax";
    record.fields.emplace("dim", AclDataValue::int64_value(-1));
    std::string key;
    int calls = 0;
    contract.consume(record, key, [&](const AclDataView& attrs) {
        if (attrs.int64("dim") == -1 && !attrs.boolean("keepdim"))
            ++calls;
    });
    if (calls != 1)
        return 1;
    try {
        AclAttrRunnerContract duplicate("Softmax", schema, {
            {"dim", AclDataType::int64}, {"dim", AclDataType::int64},
        });
        return 2;
    } catch (const jittor::InternalInvariantError&) {
    }
    try {
        AclAttrRunnerContract wrong_type("Softmax", schema, {
            {"dim", AclDataType::boolean},
        });
        return 3;
    } catch (const jittor::InternalInvariantError&) {
    }
    try {
        AclAttrRunnerContract missing("Softmax", schema, {
            {"axis", AclDataType::int64},
        });
        return 4;
    } catch (const jittor::InternalInvariantError&) {
        return 0;
    }
}
'''
    with tempfile.TemporaryDirectory(prefix="jittor-acl-runner-contract-") as directory:
        binary = Path(directory) / "probe"
        _compile(source, binary)
        subprocess.run([str(binary)], check=True)
