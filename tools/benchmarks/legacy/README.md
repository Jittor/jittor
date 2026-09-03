# Legacy Benchmarks

`inference_perf.py` is retained only to reproduce an older Jittor/PyTorch CUDA
model comparison. It imports neither framework until `main()` runs.

`bench_klo.py` reports CUDA kernel-launch overhead in nanoseconds. It came out
of `python/jittor/utils/` (task 5.25), where it was shipped inside the wheel and
ran its measurement at *import* time; it now follows the same rule as the script
above and needs a CUDA build of Jittor.

```bash
python tools/benchmarks/legacy/bench_klo.py 100000
```

Use `python -m nox -s benchmark` for maintained, repeatable performance gates.
The legacy script requires CUDA builds of Jittor and PyTorch plus torchvision:

```bash
python tools/benchmarks/legacy/inference_perf.py --batch-sizes 1,2,4
```
