#include "ops/composite/external_kernel_op.h"
#include "core/var.h"
#include "runtime/graph_capture.h"

namespace jittor {
ExternalKernelOp::ExternalKernelOp(vector<NanoVector>&& shapes,
        vector<NanoString>&& dtypes, vector<Var*>&& inputs,
        KernelFunc&& launch, string backend)
    : read_buffers(inputs), launch(move(launch)), backend(move(backend)) {
    USER_CHECK(this->backend == "cpu" || this->backend == "cuda")
        << "external_kernel: supported execution backends are cpu and cuda";
    USER_CHECK(!read_buffers.empty()) << "external_kernel needs an input for device placement";
    USER_CHECK(!shapes.empty() && shapes.size() == dtypes.size())
        << "external_kernel needs matching nonempty output shapes and dtypes";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);
    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int j = 0; j < shapes[i].size(); ++j)
            USER_CHECK(shapes[i][j] >= 0) << "external_kernel requires concrete output shapes";
        write_buffers.push_back(create_output(shapes[i], dtypes[i]));
    }
}

void ExternalKernelOp::run() {
    USER_CHECK(!graph_capture_active())
        << "external_kernel: device graph capture is not supported; use ordinary execution";
    Device device = allocation_device(write_buffers[0]->allocator);
    USER_CHECK((backend == "cuda" && device.backend == BackendId::Cuda)
            || (backend == "cpu" && device.backend == BackendId::Cpu))
        << "external_kernel: input placement does not match the requested backend";
    for (Var* v : read_buffers) {
        Device d = allocation_device(v->allocator);
        USER_CHECK(d.backend == device.backend && d.index == device.index)
            << "external_kernel: all buffers must be on the same device";
    }
    int64 stream = device.backend == BackendId::Cpu ? 0
        : reinterpret_cast<int64>(backend_stream(device, BackendStreamKind::Compute).handle);
    launch.callback({read_buffers, write_buffers, device, stream});
}

VarPtr ExternalKernelOp::grad(Var*, Var*, Var*, int) {
    LOGf << "external_kernel has no implicit derivative; define backward with jittor.Function";
    return nullptr;
}
} // namespace jittor
