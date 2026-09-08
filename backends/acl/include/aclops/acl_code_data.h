#pragma once

// Typed ACL attributes transported through CodeOp's string -> double DataMap.
// No ACL/CANN or Python dependency; semantic validation stays in decode_acl_data.
#include <limits>
#include "acl_data_channel.h"

namespace jittor {
namespace acl_data {
namespace code_data_detail {

static_assert(static_cast<unsigned>(AclDataType::int64) == 0 &&
              static_cast<unsigned>(AclDataType::float64) == 1 &&
              static_cast<unsigned>(AclDataType::boolean) == 2 &&
              static_cast<unsigned>(AclDataType::int64_vector) == 3 &&
              static_cast<unsigned>(AclDataType::float64_vector) == 4 &&
              static_cast<unsigned>(AclDataType::bool_vector) == 5,
              "changing ACL wire type codes requires a schema-version migration");

inline std::string hex_name(const std::string& name) {
    static const char digits[] = "0123456789abcdef";
    std::string result;
    result.reserve(name.size() * 2);
    for (unsigned char byte : name) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}

inline unsigned hex_digit(char value) {
    if (value >= '0' && value <= '9') return unsigned(value - '0');
    if (value >= 'a' && value <= 'f') return unsigned(value - 'a' + 10);
    user_error("ACL code-data name is not canonical hexadecimal");
    return 0;
}

inline std::string unhex_name(const std::string& text) {
    if (text.empty() || text.size() % 2)
        user_error("ACL code-data field name is invalid");
    std::string result;
    result.reserve(text.size() / 2);
    for (size_t i = 0; i < text.size(); i += 2)
        result.push_back(char((hex_digit(text[i]) << 4) | hex_digit(text[i + 1])));
    return result;
}

template<class Map>
class Reader {
public:
    const Map& data;
    const std::string& prefix;
    std::set<std::string> consumed;

    Reader(const Map& data, const std::string& prefix) : data(data), prefix(prefix) {}

    double take(const std::string& key) {
        const auto found = data.find(key);
        if (found == data.end()) user_error("missing ACL code-data key: " + key);
        const double value = found->second;
        if (!std::isfinite(value)) user_error("non-finite ACL code-data value: " + key);
        consumed.insert(key);
        return value;
    }

    uint32_t integer(const std::string& key, uint32_t maximum = UINT32_MAX) {
        const double value = take(key);
        if (value < 0 || value > maximum || std::floor(value) != value)
            user_error("invalid unsigned ACL code-data lane: " + key);
        return static_cast<uint32_t>(value);
    }

    std::pair<std::string, AclDataType> field(const std::string& base) {
        const std::string head = base + "name.";
        std::string key;
        for (const auto& item : data) {
            if (item.first.compare(0, head.size(), head) != 0) continue;
            if (!key.empty()) user_error("multiple ACL code-data names for " + base);
            key = item.first;
        }
        if (key.empty()) user_error("missing ACL code-data field name: " + base);
        auto name = unhex_name(key.substr(head.size()));
        auto type = static_cast<AclDataType>(integer(key, 5));
        return {std::move(name), type};
    }

    int64_t signed_integer(const std::string& base) {
        const uint64_t low = integer(base + "lo");
        const uint64_t high = integer(base + "hi");
        const uint64_t bits = (high << 32) | low;
        if (bits <= uint64_t(std::numeric_limits<int64_t>::max()))
            return static_cast<int64_t>(bits);
        // Avoid an implementation-defined uint64 -> int64 conversion.
        return -1 - static_cast<int64_t>(std::numeric_limits<uint64_t>::max() - bits);
    }

    AclDataValue value(const std::string& base, AclDataType type) {
        AclDataValue result;
        result.type = type;
        if (type == AclDataType::int64) result.int_value = signed_integer(base);
        else if (type == AclDataType::float64) result.float_value = take(base + "value");
        else if (type == AclDataType::boolean) result.bool_value = integer(base + "value", 1) != 0;
        else {
            const uint32_t count = integer(base + "length");
            if (count > data.size()) user_error("ACL code-data vector length exceeds payload");
            for (uint32_t i = 0; i < count; ++i) {
                const std::string item = base + "item." + std::to_string(i) + ".";
                if (type == AclDataType::int64_vector) result.int_values.push_back(signed_integer(item));
                else if (type == AclDataType::float64_vector) result.float_values.push_back(take(item + "value"));
                else result.bool_values.push_back(integer(item + "value", 1) != 0);
            }
        }
        return result;
    }

    void finish() const {
        for (const auto& item : data)
            if (item.first.compare(0, prefix.size(), prefix) == 0 && !consumed.count(item.first))
                user_error("unknown ACL code-data key: " + item.first);
    }
};

} // namespace code_data_detail

template<class Map>
AclDecodedData decode_code_data(const Map& data, const std::string& expected_op,
                                const AclAttrSchema& schema,
                                const std::string& prefix = "acl_attr.") {
    if (prefix.empty() || expected_op.empty())
        internal_error("ACL code-data owner and prefix must be non-empty");
    validate_schema(schema);
    code_data_detail::Reader<Map> reader(data, prefix);
    AclDataRecord record;
    record.schema_version = reader.integer(prefix + "version");
    if (record.schema_version != kSchemaVersion) user_error("unsupported ACL data schema version");
    const auto marker = reader.integer(prefix + "op." + code_data_detail::hex_name(expected_op), 1);
    if (marker != 1) user_error("invalid ACL code-data operator marker");
    record.op = expected_op;
    const uint32_t count = reader.integer(prefix + "fields");
    if (count > schema.size()) user_error("ACL code-data has more fields than its schema");
    std::string previous;
    for (uint32_t i = 0; i < count; ++i) {
        const std::string base = prefix + "field." + std::to_string(i) + ".";
        auto field = reader.field(base);
        if (i && field.first <= previous) user_error("ACL code-data field order is not canonical");
        previous = field.first;
        auto declaration = schema.find(field.first);
        if (declaration == schema.end()) user_error("unknown ACL data field: " + field.first);
        if (declaration->second.type != field.second)
            user_error("ACL code-data field type disagrees with schema: " + field.first);
        record.fields.emplace(field.first, reader.value(base, field.second));
    }
    reader.finish();
    std::string canonical;
    return decode_acl_data(record, expected_op, schema, canonical);
}

} // namespace acl_data
} // namespace jittor
