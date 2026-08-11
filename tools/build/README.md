# Build Tools

`build_aarch64_mkl.sh` cross-compiles the legacy oneDNN 2.2 asset from an
existing oneDNN source checkout. It does not download source code and writes to
`$JITTOR_LAB_ROOT/_state/tools/build-aarch64-mkl` by default.

```bash
JITTOR_BUILD_JOBS=8 tools/build/build_aarch64_mkl.sh /path/to/oneDNN
```

Override `CC_AARCH64`, `CXX_AARCH64`, or pass a second output-directory
argument when the default toolchain or state location is unsuitable.
