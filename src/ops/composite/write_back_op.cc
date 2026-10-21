#include "ops/composite/write_back_op.h"
#include "core/var.h"
#include <algorithm>

namespace jittor {

#ifndef JIT
WriteBackOp::WriteBackOp(vector<Var*>&& targets, vector<Var*>&& values)
    : targets(targets), values(values) {
    USER_CHECKop(targets.size(),>,0);
    USER_CHECKop(targets.size(),==,values.size());
    set_flag(OpFlags::_cuda);
    set_flag(OpFlags::_manual_set_vnbb);
    for (uint i=0; i<targets.size(); ++i) {
        USER_CHECK(targets[i]->dtype() == values[i]->dtype() && targets[i]->shape == values[i]->shape)
            << "write_back needs matching dtype and shape at index" << i;
        written.push_back(create_output(nullptr, targets[i]->dtype()));
    }
}

void WriteBackOp::infer_shape() {
    for (uint i=0; i<targets.size(); ++i) {
        written[i]->set_shape(targets[i]->shape);
        written[i]->share_with(targets[i]);
    }
    if (!ordered) {
        ordered = true;
        order_after_readers(this, targets);
    }
}

void WriteBackOp::jit_prepare(JK& jk) {
    jk << "«N=" << targets.size();
}

#else // JIT
#ifdef JIT_cuda
namespace {
constexpr int kEntries = 120;
constexpr int kThreads = 256;
constexpr int64 kPerBlock = (int64)kThreads * 16;

struct WriteBackLaunch {
    char* dst[kEntries];
    const char* src[kEntries];
    int64 bytes[kEntries];
    int first_block[kEntries + 1];
    int count;
};

// Each block copies 4 KB of one entry, 16 bytes a thread: a uint4 where both
// sides are aligned, byte by byte where not.
__global__ void write_back_kernel(WriteBackLaunch launch) {
    const int total = launch.first_block[launch.count];
    int t = 0;
    for (int block = blockIdx.x; block < total; block += gridDim.x) {
        while (t + 1 < launch.count && launch.first_block[t + 1] <= block) t++;
        const int64 offset = (int64)(block - launch.first_block[t]) * kPerBlock
            + (int64)threadIdx.x * 16;
        const int64 bytes = launch.bytes[t];
        if (offset >= bytes) continue;
        char* dst = launch.dst[t] + offset;
        const char* src = launch.src[t] + offset;
        if (offset + 16 <= bytes && (((size_t)dst | (size_t)src) & 15) == 0) {
            *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
        } else {
            const int64 n = bytes - offset < 16 ? bytes - offset : 16;
            for (int64 k = 0; k < n; k++) dst[k] = src[k];
        }
    }
}
} // namespace

void WriteBackOp::jit_run() {
    WriteBackLaunch launch;
    launch.count = 0;
    int blocks = 0;
    int device = 0, sms = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    const int grid = std::max(sms, 1) * 8;
    auto flush = [&]() {
        if (!launch.count) return;
        launch.first_block[launch.count] = blocks;
        write_back_kernel<<<std::min(blocks, grid), kThreads>>>(launch);
        launch.count = 0;
        blocks = 0;
    };
    for (uint i = 0; i < targets.size(); ++i) {
        const int64 bytes = targets[i]->size;
        if (!bytes || written[i]->mem_ptr == values[i]->mem_ptr) continue;
        const int64 need = (bytes + kPerBlock - 1) / kPerBlock;
        if (launch.count == kEntries || blocks + need > (int64)(1 << 30)) flush();
        int c = launch.count++;
        launch.dst[c] = (char*)written[i]->mem_ptr;
        launch.src[c] = (const char*)values[i]->mem_ptr;
        launch.bytes[c] = bytes;
        launch.first_block[c] = blocks;
        blocks += (int)need;
    }
    flush();
}
#else
void WriteBackOp::jit_run() {
    USER_ERROR << "write_back is only available through a mapped backend";
}
#endif // JIT_cuda
#endif // JIT

} // jittor
