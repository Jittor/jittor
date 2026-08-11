# Legacy Benchmarks

`inference_perf.py` is retained only to reproduce an older Jittor/PyTorch CUDA
model comparison. It imports neither framework until `main()` runs.

Use `python -m nox -s benchmark` for maintained, repeatable performance gates.
The legacy script requires CUDA builds of Jittor and PyTorch plus torchvision:

```bash
python tools/benchmarks/legacy/inference_perf.py --batch-sizes 1,2,4
```
