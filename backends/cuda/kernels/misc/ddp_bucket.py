"""Pack dense DDP gradients on the compute stream before NCCL consumes them."""

import jittor as jt


def pack_gradients(gradients):
    """One output allocation and device copies, preserving the input dtype.

    DDP supplies one device and dtype per bucket. ``contiguous`` also handles
    gradient views; empty inputs require no copy. NCCL's compute-to-comm event
    orders these copies before the collective. This operator has no backward:
    optimizer gradients have already been detached from the training graph.
    """
    inputs = [gradient.contiguous() for gradient in gradients]
    copies = []
    offset = 0
    for index, gradient in enumerate(inputs):
        count = int(gradient.numel())
        if count:
            copies.append(
                "checkCudaErrors(cudaMemcpyAsync(out0_p + {offset}, in{index}_p, "
                "{count} * sizeof(out0_type), cudaMemcpyDeviceToDevice, "
                "cudaStreamPerThread));".format(
                    offset=offset, index=index, count=count))
        offset += count
    return jt.code(
        (offset,), inputs[0].dtype, inputs,
        cuda_header='#include "helper_cuda.h"',
        cuda_src="// Empty buckets need no device copy.\n" + "\n".join(copies),
    ).stop_grad()


def unpack_gradients(reduced, gradients, world_size):
    """Scale after the NCCL group closes, then copy to detached dense outputs.

    A single multi-output operator avoids one slice/reshape graph per gradient.
    Outputs own their storage: optimizer updates and zero_grad can rebind their
    holders without modifying the collective output or another gradient.
    """
    averaged = reduced / world_size
    copies = []
    offset = 0
    for index, gradient in enumerate(gradients):
        count = int(gradient.numel())
        if count:
            copies.append(
                "checkCudaErrors(cudaMemcpyAsync(out{index}_p, in0_p + {offset}, "
                "{count} * sizeof(in0_type), cudaMemcpyDeviceToDevice, "
                "cudaStreamPerThread));".format(
                    offset=offset, index=index, count=count))
        offset += count
    outputs = jt.code(
        [gradient.shape for gradient in gradients],
        [gradient.dtype for gradient in gradients], [averaged],
        cuda_header='#include "helper_cuda.h"',
        cuda_src="// Empty outputs need no device copy.\n" + "\n".join(copies),
    )
    return [output.stop_grad() for output in outputs]
