#pragma once
// ACL default-stream bookkeeping only, with no driver or Python dependency.
// The native provider owns one instance, serialized by ExecutorEntryScope.
#include <cstdint>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace jittor {

struct AclRngSnapshot { int64_t seed = 0, offset = 0; };

class AclRngStreams {
    int64_t default_seed_ = 0;
    std::map<int, AclRngSnapshot> streams_;
public:
    // CANN 9 uniform takes uint64_t; normal and RandomAttr take int64_t.
    // Their common nonnegative domain is lossless. Torch's unsigned upper half
    // and negative-seed remapping are explicitly not implemented.
    static void validate_seed(int64_t seed) {
        if (seed < 0)
            throw std::invalid_argument("NPU manual_seed supports 0..INT64_MAX only");
    }
    static void validate(const AclRngSnapshot& value) {
        // Preserve existing native jt.set_seed(int) negative seeds. Only that
        // legacy int32 domain may appear as a negative seed in a saved state.
        if (value.seed < std::numeric_limits<int32_t>::min())
            throw std::invalid_argument("Unsupported negative native ACL RNG seed");
        if (value.offset < 0) throw std::invalid_argument("Negative ACL RNG offset");
    }
    static std::string encode(const AclRngSnapshot& value) {
        validate(value);
        std::ostringstream out;
        out.imbue(std::locale::classic());
        out << "JITTOR_ACL_RNG_V1\nnumel_v1\n" << value.seed << '\n' << value.offset;
        return out.str();
    }
    static AclRngSnapshot decode(const std::string& blob) {
        const std::string header = "JITTOR_ACL_RNG_V1\nnumel_v1\n";
        if (blob.size() > 256 || blob.compare(0, header.size(), header) != 0)
            throw std::invalid_argument("Invalid or incompatible Jittor ACL RNG state");
        std::istringstream input(blob.substr(header.size()));
        input.imbue(std::locale::classic());
        AclRngSnapshot result;
        if (!(input >> result.seed >> result.offset))
            throw std::invalid_argument("Malformed Jittor ACL RNG state");
        validate(result);
        if (encode(result) != blob)
            throw std::invalid_argument("Noncanonical Jittor ACL RNG state");
        return result;
    }
    AclRngSnapshot get(int device) {
        if (device < 0) throw std::invalid_argument("Negative ACL RNG device");
        auto found = streams_.find(device);
        if (found == streams_.end()) found = streams_.emplace(device, AclRngSnapshot{default_seed_, 0}).first;
        return found->second;
    }
    void set(int device, const AclRngSnapshot& value) {
        validate(value);
        if (device < 0) throw std::invalid_argument("Negative ACL RNG device");
        streams_[device] = value;
    }
    void set_all(const std::vector<AclRngSnapshot>& values) {
        std::map<int, AclRngSnapshot> replacement;
        for (size_t i = 0; i < values.size(); ++i) {
            validate(values[i]);
            replacement.emplace(static_cast<int>(i), values[i]);
        }
        streams_.swap(replacement);
    }
    void seed_all(int64_t seed) {
        validate_seed(seed);
        default_seed_ = seed;
        streams_.clear();
    }
    void native_seed_all(int seed) {
        default_seed_ = seed;
        streams_.clear();
    }
    AclRngSnapshot prepare(int device, int64_t count) {
        auto value = get(device);
        if (count < 0 || value.offset > std::numeric_limits<int64_t>::max() - count)
            throw std::overflow_error("ACL RNG offset overflow");
        return value;
    }
    void advance(int device, int64_t count) {
        auto value = prepare(device, count);
        value.offset += count;
        set(device, value);
    }
};

} // namespace jittor
