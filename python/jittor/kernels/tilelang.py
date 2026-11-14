"""TileLang's public executable export and TVM-FFI DLPack execution."""


def executable_for(kernel):
    if getattr(kernel, "execution_backend", None) != "tvm_ffi":
        raise ValueError("compile TileLang kernels with execution_backend='tvm_ffi'")
    if kernel.target.kind.name != "cuda":
        raise ValueError("tilelang_call currently requires a CUDA compilation target")
    if kernel.out_idx:
        raise ValueError("compile with out_idx=[] and declare Jittor TensorSpec output arguments")
    # TileLang's export API avoids its Torch allocation/call wrapper entirely.
    return kernel.adapter.get_exportable_executable()


def launch(executable, args, context):
    import tvm_ffi
    device = tvm_ffi.device("cuda", context.device[1])
    with tvm_ffi.use_raw_stream(device, context.stream):
        # TVM-FFI's generic importer calls __dlpack__ without a stream.
        # Export explicitly on the executor's stream before importing the
        # owning capsule; do not synchronize or switch to the default stream.
        values = [tvm_ffi.from_dlpack(arg.__dlpack__(stream=context.stream or 1))
                  if hasattr(arg, "__dlpack__") else arg for arg in args]
        executable(*values)
