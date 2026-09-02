#pragma once
#include <cstdint>
#include <mutex>
#include <optional>
#include <tuple>
#include <type_traits>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/PhiloxCudaState.h>
#include <ATen/cuda/detail/UnpackRaw.cuh>

namespace jittor {
int get_seed();
extern int64_t current_offset;
}

namespace at {

struct CUDAGeneratorImpl {
    std::mutex mutex_;
    PhiloxCudaState philox_cuda_state(uint64_t increment) {
        auto state = PhiloxCudaState(
            static_cast<uint64_t>(jittor::get_seed()),
            static_cast<uint64_t>(jittor::current_offset));
        jittor::current_offset += static_cast<int64_t>(increment);
        return state;
    }
};

// Torch's Generator is a handle over a generator implementation, not the implementation itself:
// a caller locks `mutex()` for the duration of a draw and reaches the implementation through
// `get<T>()`. Extensions are written against that shape -- flash-attn's dropout path does
// `gen.mutex()` and `gen.get<at::CUDAGeneratorImpl>()` -- so keep it, even though CUDA is the
// only implementation this shim has.
class Generator {
public:
    Generator() = default;
    explicit Generator(CUDAGeneratorImpl* impl) : impl_(impl) {}

    std::mutex& mutex() const { return impl_->mutex_; }

    template <typename T>
    T* get() const {
        static_assert(std::is_same<T, CUDAGeneratorImpl>::value,
                      "the Jittor torch shim only provides at::CUDAGeneratorImpl");
        return impl_;
    }

    explicit operator bool() const { return impl_ != nullptr; }

private:
    CUDAGeneratorImpl* impl_ = nullptr;
};

namespace cuda { namespace detail {
inline const Generator& getDefaultCUDAGenerator() {
    static CUDAGeneratorImpl impl;
    static Generator generator(&impl);
    return generator;
}
}} // namespace cuda::detail

template <typename T>
T* get_generator_or_default(const std::optional<Generator>& gen,
                            const Generator& default_gen) {
    return (gen.has_value() && static_cast<bool>(*gen)) ? gen->template get<T>()
                                                        : default_gen.get<T>();
}

} // namespace at
