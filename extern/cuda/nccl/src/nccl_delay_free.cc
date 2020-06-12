#include "event_queue.h"
#include "nccl_delay_free.h"

namespace jittor {
NcclDelayFree nccl_delay_free;

static void free_caller() {
    nccl_delay_free.wait_for_free.pop_front();
}

void to_free_nccl_allocation(CUDA_HOST_FUNC_ARGS) {
    event_queue.push(free_caller);
}

NcclDelayFree::~NcclDelayFree() {
    cudaDeviceSynchronize();
    event_queue.flush();
}
}