#pragma once
#include <torch/extension.h>

namespace torch { namespace nn { namespace functional {

struct PadFuncOptions {
    std::vector<int64_t> pad_;
    explicit PadFuncOptions(std::vector<int64_t> pad) : pad_(std::move(pad)) {}
    explicit PadFuncOptions(std::initializer_list<int64_t> pad) : pad_(pad) {}
};

inline torch::Tensor pad(const torch::Tensor&, const PadFuncOptions&) {
    throw std::runtime_error("torch::nn::functional::pad is not implemented in the Jittor torch shim");
}

}}} // namespace torch::nn::functional
