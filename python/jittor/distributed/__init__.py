"""jittor.distributed: MPI-free multi-device launching (torchrun-style).

See ``jittor.distributed.launch`` (run as ``python -m jittor.distributed.launch``).
The collective bootstrap is env/file based (JT_{NCCL,HCCL}_*), so multi-card data
parallel works on both NVIDIA (NCCL) and Ascend (HCCL) without mpirun.
"""
