# Legacy Installers

These scripts exist for reproducibility of old provisioning workflows. New
systems should use the installation commands in the root README and their
distribution's supported compiler packages.

- `install.sh` provisions a Python interpreter and runs the installed
  `jittor.selftest` CPU check, plus a CUDA check when `with_cuda=1`.
- `install_llvm.sh` supports only obsolete LLVM 8-10 distributions and requires
  the explicit `--acknowledge-legacy` argument. Do not pipe it from the network.
- `install_mkl.sh` installs the old oneDNN 1.0.2 asset. Its optional argument is
  an output directory; the default is the runtime asset location resolved from
  the repository root.

Every script is fail-closed when a required command or requested compiler is
unavailable.
