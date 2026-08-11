# Distributed Tools

`tmpi` is the GPLv2 tmux/OpenMPI helper retained from the original repository.
It validates `tmux` and `mpirun`, creates its coordination file below
`$JITTOR_LAB_ROOT/_state/tools/tmpi`, and propagates optional `HOSTS_ARGS` and
`MPI_ARGS` strings. Its tmux server socket is also created in that state
directory rather than the system temporary directory.

```bash
tools/distributed/tmpi 4 python train.py
```
