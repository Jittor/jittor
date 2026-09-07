"""CPU-side library sources, owned here rather than under the Python package.

The CPU backend itself is native and always present, so this package declares
no ``configure``/``install_extern``/``post_process`` provider the way
``backends/cuda`` and ``backends/rocm`` do. It exists so that CPU math
libraries sit next to the accelerator ones in the same shape --
``backends/<backend>/libraries/<library>/`` -- and so that
``backend_root(jittor_path, "cpu")`` resolves in both a checkout and an
installed tree.

``libraries/mkl/`` holds the oneDNN operator sources that used to live in
``python/jittor/extern/mkl/ops/``; ``compile_extern.setup_mkl`` builds them and
already declared ``backend="cpu"`` for them before the move.
"""
