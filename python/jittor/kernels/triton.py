"""Triton compilation and launch; no Torch shim or private CUDA launcher."""


def launch(kernel, args, grid, options, context):
    from triton.runtime import driver

    active = driver.active
    previous = active.get_current_device()
    device = context.device[1]
    try:
        active.set_current_device(device)
        if active.get_current_target().backend != "cuda":
            raise RuntimeError("triton_call requires a CUDA Triton driver")
        # Keep upstream binding, specialization, validation and cache keys.
        compiled = kernel.warmup(*args, grid=grid, **options)
        signature = kernel.signature
        bound = signature.bind(*args, **{k: v for k, v in options.items() if k in signature.parameters})
        bound.apply_defaults()
        launch_grid = grid(bound.arguments) if callable(grid) else grid
        launch_grid = tuple(launch_grid)
        if not 1 <= len(launch_grid) <= 3 or any(type(n) is not int or n < 0 for n in launch_grid):
            raise ValueError("Triton grid must contain one to three nonnegative integers")
        # Scratch allocation defaults to Torch in upstream Triton. Reject such
        # kernels until workspace allocation can be owned by this graph node.
        if any(getattr(compiled.metadata, field, 0) for field in
               ("global_scratch_size", "profile_scratch_size")):
            raise NotImplementedError("Triton kernels requiring global scratch are not supported yet")
        compiled[launch_grid + (1,) * (3 - len(launch_grid))](
            *bound.arguments.values(), stream=context.stream)
    finally:
        active.set_current_device(previous)
