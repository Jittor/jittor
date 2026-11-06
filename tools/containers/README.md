# Development containers

`Containerfile.cuda` installs the current checkout's core and compat distributions,
plus test dependencies. Build it with the repository root as the context:

```bash
podman build -t jittor-cuda:local -f tools/containers/Containerfile.cuda .
podman run --rm -it --device nvidia.com/gpu=all jittor-cuda:local bash
```

GPU access requires the host's NVIDIA driver and Podman CDI configuration. Docker
users can build with the same `-f` path and run with `--gpus all` instead.
The recipe is not evidence of device validation; choose a base image matching your
toolkit and workload with `--build-arg FROM_IMAGE=...`.

The root `Dockerfile` serves a different purpose: it builds the notebook image used
by the container workflow. Build instructions and CI choices should retain that
root entry point.
