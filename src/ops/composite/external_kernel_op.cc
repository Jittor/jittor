#include "ops/composite/external_kernel_op.h"
#include "core/var.h"
#include "runtime/device.h"
#include "runtime/graph_capture.h"
#include "runtime/tensor_placement.h"

namespace jittor {
ExternalKernelOp::ExternalKernelOp(vector<NanoVector>&& shapes,
        vector<NanoString>&& dtypes, vector<Var*>&& inputs,
        KernelFunc&& launch, string backend)
    : read_buffers(inputs), launch(move(launch)), backend(move(backend)) {
    USER_CHECK(this->backend == "cpu" || this->backend == "cuda")
        << "external_kernel: supported execution backends are cpu and cuda";
    USER_CHECK(!shapes.empty() && shapes.size() == dtypes.size())
        << "external_kernel needs matching nonempty output shapes and dtypes";
    set_flag(OpFlags::_cpu);
    set_flag(OpFlags::_cuda);

    TensorPlacement zero_input_placement;
    int zero_input_device = -1;
    if (read_buffers.empty()) {
        if (this->backend == "cpu") {
            zero_input_placement = TensorPlacement({BackendId::Cpu, 0});
            zero_input_device = -1;
        } else {
            USER_CHECK(accelerator_backend_id() == BackendId::Cuda)
                << "external_kernel: CUDA backend is not available";
            auto current = current_tensor_placement();
            int dev = (current.explicit_backend && current.device.backend == BackendId::Cuda)
                ? current.device.index : current_device();
            USER_CHECK(dev >= 0) << "external_kernel: no active CUDA device";
            zero_input_placement = TensorPlacement({BackendId::Cuda, dev});
            zero_input_device = dev;
        }
    }

    for (size_t i = 0; i < shapes.size(); ++i) {
        for (int j = 0; j < shapes[i].size(); ++j)
            USER_CHECK(shapes[i][j] >= 0) << "external_kernel requires concrete output shapes";
        Var* out = create_output(shapes[i], dtypes[i]);
        if (read_buffers.empty()) {
            out->placement = zero_input_placement;
            out->device_id = zero_input_device;
        }
        write_buffers.push_back(out);
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
