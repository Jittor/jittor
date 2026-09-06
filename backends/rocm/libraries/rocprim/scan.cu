#include "scan.h"
#include <cstdint>
#include <rocprim/device/device_scan.hpp>
#include <rocprim/iterator/reverse_iterator.hpp>

namespace jittor {
namespace {
template<class T>
hipError_t scan_typed(void* workspace, size_t& bytes, const void* input,
                      void* output, size_t count, bool reverse, hipStream_t stream) {
    auto source = static_cast<const T*>(input);
    auto target = static_cast<T*>(output);
    if (reverse) {
        rocprim::reverse_iterator<const T*> first(source + count);
        rocprim::reverse_iterator<T*> destination(target + count);
        return rocprim::inclusive_scan(workspace, bytes, first, destination, count,
                                       rocprim::plus<T>(), stream);
    }
    return rocprim::inclusive_scan(workspace, bytes, source, target, count,
                                   rocprim::plus<T>(), stream);
}
} // namespace

hipError_t rocprim_scan(void* workspace, size_t& workspace_bytes,
                       const void* input, void* output, size_t count,
                       RocprimScanType dtype, bool reverse, hipStream_t stream) {
    switch (dtype) {
        case RocprimScanType::Float32:
            return scan_typed<float>(workspace, workspace_bytes, input, output, count, reverse, stream);
        case RocprimScanType::Float64:
            return scan_typed<double>(workspace, workspace_bytes, input, output, count, reverse, stream);
        case RocprimScanType::Int32:
            return scan_typed<int32_t>(workspace, workspace_bytes, input, output, count, reverse, stream);
        case RocprimScanType::Int64:
            return scan_typed<int64_t>(workspace, workspace_bytes, input, output, count, reverse, stream);
    }
    return hipErrorInvalidValue;
}

} // namespace jittor
