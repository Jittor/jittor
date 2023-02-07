#include "CUDAUtils.h"

namespace jittor {
    namespace at {
        namespace cuda {
            bool check_device(std::vector<torch::Tensor>) {
                return true;
            }
        }
    }
}