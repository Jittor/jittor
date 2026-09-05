#pragma once

// Host-side ACL attribute channel.  Keep this header independent of ACL/CANN
// so schema and decoder changes can be compiled and tested on a CPU host.
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <initializer_list>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "utils/log.h"

namespace jittor {
namespace acl_data {

constexpr uint32_t kSchemaVersion = 1;

enum class AclDataType : uint8_t {
    int64,
    float64,
    boolean,
    int64_vector,
    float64_vector,
    bool_vector,
};

inline const char* type_name(AclDataType type) {
    switch (type) {
        case AclDataType::int64: return "int64";
        case AclDataType::float64: return "float64";
        case AclDataType::boolean: return "bool";
        case AclDataType::int64_vector: return "int64[]";
        case AclDataType::float64_vector: return "float64[]";
        case AclDataType::bool_vector: return "bool[]";
    }
    return "<invalid>";
}

inline bool is_vector(AclDataType type) {
    return type == AclDataType::int64_vector ||
           type == AclDataType::float64_vector ||
           type == AclDataType::bool_vector;
}

inline bool is_valid_type(AclDataType type) {
    const auto value = static_cast<unsigned>(type);
    return value <= static_cast<unsigned>(AclDataType::bool_vector);
}

// A value owns its storage.  In particular, no pointer or Python object id
// can leak into a generated JIT key or survive across the decoder boundary.
struct AclDataValue {
    AclDataType type = AclDataType::int64;
    int64_t int_value = 0;
    double float_value = 0;
    bool bool_value = false;
    std::vector<int64_t> int_values;
    std::vector<double> float_values;
    std::vector<bool> bool_values;

    static AclDataValue int64_value(int64_t value) {
        AclDataValue result;
        result.type = AclDataType::int64;
        result.int_value = value;
        return result;
    }
    static AclDataValue float64_value(double value) {
        AclDataValue result;
        result.type = AclDataType::float64;
        result.float_value = value;
        return result;
    }
    static AclDataValue bool_value_of(bool value) {
        AclDataValue result;
        result.type = AclDataType::boolean;
        result.bool_value = value;
        return result;
    }
    static AclDataValue int64_vector(std::vector<int64_t> value) {
        AclDataValue result;
        result.type = AclDataType::int64_vector;
        result.int_values = std::move(value);
        return result;
    }
    static AclDataValue float64_vector(std::vector<double> value) {
        AclDataValue result;
        result.type = AclDataType::float64_vector;
        result.float_values = std::move(value);
        return result;
    }
    static AclDataValue bool_vector(std::vector<bool> value) {
        AclDataValue result;
        result.type = AclDataType::bool_vector;
        result.bool_values = std::move(value);
        return result;
    }
};

using AclDataMap = std::map<std::string, AclDataValue>;

struct AclAttrField {
    AclDataType type = AclDataType::int64;
    bool required = true;
    bool has_default = false;
    AclDataValue default_value;
};

using AclAttrSchema = std::map<std::string, AclAttrField>;

struct AclDataRecord {
    uint32_t schema_version = kSchemaVersion;
    std::string op;
    AclDataMap fields;
};

struct AclDecodedData {
    uint32_t schema_version = kSchemaVersion;
    std::string op;
    AclDataMap fields;
    std::string cache_key;
};

// Descriptor identity is deliberately separate from the attribute record.
// ACL descriptors are shape/layout/device specific; reusing one by operator
// or attribute key alone can silently attach a tensor to the wrong device.
// This value object is CANN-free so the registry can validate cache identity
// before an eventual runner creates an aclTensor descriptor.
constexpr uint32_t kDescriptorKeyVersion = 1;

struct AclDescriptorKey {
    std::string attribute_key;
    std::vector<int64_t> shape;
    std::string dtype;
    std::string layout;
    std::string device;
};

inline void internal_error(const std::string& message);

// Read-only view passed to a future OpAttr/ACL consumer.  Keeping field
// lookup here prevents each launcher from reimplementing type checks or
// reaching into the decoder's mutable map.  The view borrows the decoded
// record and therefore must not outlive the consume() call below.
class AclDataView {
public:
    const std::string& op() const {
        return decoded_->op;
    }

    uint32_t schema_version() const {
        return decoded_->schema_version;
    }

    const std::string& cache_key() const {
        return decoded_->cache_key;
    }

    bool has(const std::string& name) const {
        if (allowed_fields_ && allowed_fields_->find(name) == allowed_fields_->end())
            return false;
        return decoded_->fields.find(name) != decoded_->fields.end();
    }

    const AclDataValue& value(const std::string& name) const {
        if (allowed_fields_ && allowed_fields_->find(name) == allowed_fields_->end())
            internal_error("ACL attribute consumer requested unbound field: " + name);
        auto field = decoded_->fields.find(name);
        if (field == decoded_->fields.end())
            internal_error("ACL attribute consumer requested absent field: " + name);
        auto declaration = schema_->find(name);
        if (declaration == schema_->end())
            internal_error("ACL attribute consumer requested undeclared field: " + name);
        if (field->second.type != declaration->second.type)
            internal_error("ACL decoded field type disagrees with schema: " + name);
        return field->second;
    }

    int64_t int64(const std::string& name) const {
        const auto& item = value(name);
        require_type(name, item, AclDataType::int64);
        return item.int_value;
    }

    double float64(const std::string& name) const {
        const auto& item = value(name);
        require_type(name, item, AclDataType::float64);
        return item.float_value;
    }

    bool boolean(const std::string& name) const {
        const auto& item = value(name);
        require_type(name, item, AclDataType::boolean);
        return item.bool_value;
    }

    const std::vector<int64_t>& int64_vector(const std::string& name) const {
        const auto& item = value(name);
        require_type(name, item, AclDataType::int64_vector);
        return item.int_values;
    }

    const std::vector<double>& float64_vector(const std::string& name) const {
        const auto& item = value(name);
        require_type(name, item, AclDataType::float64_vector);
        return item.float_values;
    }

    const std::vector<bool>& bool_vector(const std::string& name) const {
        const auto& item = value(name);
        require_type(name, item, AclDataType::bool_vector);
        return item.bool_values;
    }

private:
    friend class AclDataOwner;
    friend class AclAttrRunnerContract;

    AclDataView(const AclDecodedData& decoded, const AclAttrSchema& schema,
                const std::set<std::string>* allowed_fields = nullptr)
        : decoded_(&decoded), schema_(&schema), allowed_fields_(allowed_fields) {}

    static void require_type(const std::string& name,
                             const AclDataValue& value,
                             AclDataType expected) {
        if (value.type != expected)
            internal_error("ACL attribute consumer requested " +
                           std::string(type_name(expected)) + " field " + name +
                           ", got " + type_name(value.type));
    }

    const AclDecodedData* decoded_;
    const AclAttrSchema* schema_;
    const std::set<std::string>* allowed_fields_;
};

inline void user_error(const std::string& message) {
    throw UserError(message);
}

inline void internal_error(const std::string& message) {
    throw InternalInvariantError(message);
}

inline void validate_value(const std::string& name,
                           AclDataType expected,
    const AclDataValue& value,
                           bool schema_default = false) {
    if (!is_valid_type(expected))
        internal_error("ACL schema contains an invalid type tag for " + name);
    if (value.type != expected) {
        const std::string message = "ACL data field " + name + " has type " +
            type_name(value.type) + ", expected " + type_name(expected);
        if (schema_default)
            internal_error(message);
        user_error(message);
    }
    if (expected == AclDataType::float64) {
        if (!std::isfinite(value.float_value)) {
            if (schema_default)
                internal_error("ACL schema default " + name + " must be finite");
            user_error("ACL data field " + name + " must be finite");
        }
    } else if (expected == AclDataType::float64_vector) {
        for (double item : value.float_values) {
            if (!std::isfinite(item)) {
                if (schema_default)
                    internal_error("ACL schema default " + name + " contains a non-finite value");
                user_error("ACL data field " + name + " contains a non-finite value");
            }
        }
    }
}

inline void validate_schema(const AclAttrSchema& schema) {
    for (const auto& item : schema) {
        const std::string& name = item.first;
        const AclAttrField& field = item.second;
        if (name.empty())
            internal_error("ACL schema field names must be non-empty");
        // Validate the declaration even when it has no default.  Otherwise an
        // invalid enum value can sit in a required field and only surface as
        // a misleading missing-value/user-data error during decode.
        if (!is_valid_type(field.type))
            internal_error("ACL schema contains an invalid type tag for " + name);
        if (field.has_default)
            validate_value(name, field.type, field.default_value, true);
        if (field.required && field.has_default)
            internal_error("ACL schema field cannot be both required and defaulted: " + name);
    }
}

inline void append_length_prefixed(std::ostringstream& key, const std::string& value) {
    key << value.size() << ':' << value;
}

inline void append_value(std::ostringstream& key, const AclDataValue& value) {
    key << type_name(value.type) << '=';
    switch (value.type) {
        case AclDataType::int64:
            key << value.int_value;
            break;
        case AclDataType::float64:
            key << std::setprecision(17) << value.float_value;
            break;
        case AclDataType::boolean:
            key << (value.bool_value ? "true" : "false");
            break;
        case AclDataType::int64_vector:
            key << '[';
            for (size_t i = 0; i < value.int_values.size(); ++i)
                key << (i ? "," : "") << value.int_values[i];
            key << ']';
            break;
        case AclDataType::float64_vector:
            key << '[' << std::setprecision(17);
            for (size_t i = 0; i < value.float_values.size(); ++i)
                key << (i ? "," : "") << value.float_values[i];
            key << ']';
            break;
        case AclDataType::bool_vector:
            key << '[';
            for (size_t i = 0; i < value.bool_values.size(); ++i)
                key << (i ? "," : "") << (value.bool_values[i] ? "true" : "false");
            key << ']';
            break;
    }
}

inline std::string canonical_cache_key(uint32_t schema_version,
                                       const std::string& op,
                                       const AclDataMap& fields) {
    std::ostringstream key;
    // Cache keys cross process/device boundaries; never inherit a caller's
    // locale, whose decimal separator would otherwise change float keys.
    key.imbue(std::locale::classic());
    key << schema_version << "|op=";
    append_length_prefixed(key, op);
    for (const auto& item : fields) {
        key << "|field=";
        append_length_prefixed(key, item.first);
        key << ':';
        append_value(key, item.second);
    }
    return key.str();
}

inline void validate_descriptor_key(const AclDescriptorKey& descriptor) {
    if (descriptor.attribute_key.empty())
        internal_error("ACL descriptor key requires an attribute cache key");
    if (descriptor.dtype.empty() || descriptor.layout.empty() || descriptor.device.empty())
        internal_error("ACL descriptor key requires dtype, layout, and device");
    for (int64_t dimension : descriptor.shape) {
        if (dimension < 0)
            user_error("ACL descriptor shape dimensions must be non-negative");
    }
}

inline std::string canonical_descriptor_key(const AclDescriptorKey& descriptor) {
    validate_descriptor_key(descriptor);
    std::ostringstream key;
    key.imbue(std::locale::classic());
    key << "v" << kDescriptorKeyVersion << "|attrs=";
    append_length_prefixed(key, descriptor.attribute_key);
    key << "|shape=" << descriptor.shape.size() << ':';
    for (size_t i = 0; i < descriptor.shape.size(); ++i)
        key << (i ? "," : "") << descriptor.shape[i];
    key << "|dtype=";
    append_length_prefixed(key, descriptor.dtype);
    key << "|layout=";
    append_length_prefixed(key, descriptor.layout);
    key << "|device=";
    append_length_prefixed(key, descriptor.device);
    return key.str();
}

inline AclDescriptorKey make_descriptor_key(const AclDecodedData& decoded,
                                            std::vector<int64_t> shape,
                                            std::string dtype,
                                            std::string layout,
                                            std::string device) {
    AclDescriptorKey result;
    result.attribute_key = decoded.cache_key;
    result.shape = std::move(shape);
    result.dtype = std::move(dtype);
    result.layout = std::move(layout);
    result.device = std::move(device);
    validate_descriptor_key(result);
    return result;
}

// Host-only cache shell.  The value is intentionally a template: a CANN
// runner can later provide its descriptor handle, while CPU tests use a
// trivial value.  The cache never manufactures or aliases runtime handles.
template <typename Descriptor>
class AclDescriptorCache {
public:
    template <typename Builder>
    Descriptor& get_or_create(const AclDescriptorKey& key, Builder&& builder) {
        const std::string canonical = canonical_descriptor_key(key);
        auto found = entries_.find(canonical);
        if (found != entries_.end())
            return found->second;
        auto inserted = entries_.emplace(
            canonical, builder(key));
        return inserted.first->second;
    }

    bool contains(const AclDescriptorKey& key) const {
        return entries_.find(canonical_descriptor_key(key)) != entries_.end();
    }

    size_t size() const {
        return entries_.size();
    }

    void clear() {
        entries_.clear();
    }

private:
    std::map<std::string, Descriptor> entries_;
};

// Shared host-side decoder contract.  It is intentionally not wired into an
// ACL runner yet: the first attribute owner must migrate schema, generated
// OpAttr construction, and JIT/cache key in one atomic change.
inline AclDecodedData decode_acl_data(const AclDataRecord& record,
                                      const std::string& expected_op,
                                      const AclAttrSchema& schema,
                                      std::string& canonical_key) {
    if (record.schema_version != kSchemaVersion)
        user_error("unsupported ACL data schema version");
    if (record.op.empty() || record.op != expected_op)
        user_error("ACL data operator does not match registered owner");
    validate_schema(schema);

    for (const auto& item : record.fields) {
        if (schema.find(item.first) == schema.end())
            user_error("unknown ACL data field: " + item.first);
    }

    AclDecodedData result;
    result.schema_version = record.schema_version;
    result.op = record.op;
    for (const auto& item : schema) {
        const auto value = record.fields.find(item.first);
        if (value == record.fields.end()) {
            if (item.second.has_default) {
                result.fields.emplace(item.first, item.second.default_value);
            } else if (item.second.required) {
                user_error("missing required ACL data field: " + item.first);
            }
            continue;
        }
        validate_value(item.first, item.second.type, value->second);
        result.fields.emplace(value->first, value->second);
    }
    result.cache_key = canonical_cache_key(result.schema_version, result.op, result.fields);
    canonical_key = result.cache_key;
    return result;
}

// The owner is the C++ boundary that a future ACL registry entry will hold.
// It owns the operator identity and schema, so callers cannot accidentally
// decode one operator with another operator's fields or a temporary schema.
// Keeping this object ACL-free lets registry and attribute tests run without
// CANN while preserving the same decode/key contract used on device.
class AclDataOwner {
public:
    AclDataOwner(std::string op, AclAttrSchema schema)
        : op_(std::move(op)), schema_(std::move(schema)) {
        if (op_.empty())
            internal_error("ACL data owner name must be non-empty");
        validate_schema(schema_);
    }

    const std::string& op() const {
        return op_;
    }

    const AclAttrSchema& schema() const {
        return schema_;
    }

    AclDecodedData decode(const AclDataRecord& record,
                          std::string& canonical_key) const {
        return decode_acl_data(record, op_, schema_, canonical_key);
    }

    // Decode once and expose only a validated, read-only view to the consumer.
    // A future ACL owner can use this to construct OpAttr values without
    // parsing generated source text.  No ACL/CANN object is created here.
    template <typename Consumer>
    void consume(const AclDataRecord& record,
                 std::string& canonical_key,
                 Consumer&& consumer) const {
        AclDecodedData decoded = decode(record, canonical_key);
        AclDataView view(decoded, schema_);
        consumer(view);
    }

private:
    std::string op_;
    AclAttrSchema schema_;
};

// A registry entry can describe the attributes its launcher consumes without
// depending on ACL/CANN.  The binding list is deliberately owned by the
// contract: a launcher cannot accidentally read a field that was omitted from
// its registered schema, and a schema edit cannot silently change the type a
// generated consumer expects.
struct AclAttrBinding {
    std::string name;
    AclDataType type = AclDataType::int64;
};

using AclAttrBindings = std::vector<AclAttrBinding>;

class AclAttrRunnerContract {
public:
    AclAttrRunnerContract(std::string op,
                          AclAttrSchema schema,
                          AclAttrBindings bindings)
        : owner_(std::move(op), std::move(schema)),
          bindings_(std::move(bindings)) {
        validate_bindings();
    }

    AclAttrRunnerContract(std::string op,
                          AclAttrSchema schema,
                          std::initializer_list<AclAttrBinding> bindings)
        : AclAttrRunnerContract(std::move(op), std::move(schema),
                                AclAttrBindings(bindings)) {}

    const AclDataOwner& owner() const {
        return owner_;
    }

    const AclAttrBindings& bindings() const {
        return bindings_;
    }

    // Decode exactly once, then invoke the generated/static consumer with a
    // bounded read-only view.  No ACL object is allocated on this boundary;
    // the eventual launcher must copy values while the callback is active.
    template <typename Consumer>
    void consume(const AclDataRecord& record,
                 std::string& canonical_key,
                 Consumer&& consumer) const {
        // Decode once, then expose only the fields registered by this runner.
        // This keeps a generated consumer from accidentally depending on a
        // schema field that is not part of its binding contract.
        AclDecodedData decoded = owner_.decode(record, canonical_key);
        AclDataView attrs(decoded, owner_.schema(), &bound_names_);
        for (const auto& binding : bindings_) {
            if (!attrs.has(binding.name))
                internal_error("ACL runner binding has no decoded field: " +
                              binding.name);
            // value() rechecks the schema/type pair immediately before the
            // generated consumer sees the view.
            const auto& decoded_field = attrs.value(binding.name);
            if (decoded_field.type != binding.type)
                internal_error("ACL runner binding type disagrees with decoded field: " +
                              binding.name);
        }
        consumer(attrs);
    }

private:
    void validate_bindings() const {
        std::set<std::string> names;
        const auto& schema = owner_.schema();
        for (const auto& binding : bindings_) {
            if (binding.name.empty())
                internal_error("ACL runner binding names must be non-empty");
            if (!names.insert(binding.name).second)
                internal_error("duplicate ACL runner binding: " + binding.name);
            bound_names_.insert(binding.name);
            const auto field = schema.find(binding.name);
            if (field == schema.end())
                internal_error("ACL runner binding is not declared in schema: " +
                              binding.name);
            if (field->second.type != binding.type)
                internal_error("ACL runner binding type disagrees with schema: " +
                              binding.name);
        }
    }

    AclDataOwner owner_;
    AclAttrBindings bindings_;
    mutable std::set<std::string> bound_names_;
};

} // namespace acl_data
} // namespace jittor
