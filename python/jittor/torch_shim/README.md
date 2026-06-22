# torch_shim — run torch code on jittor

`import torch` → jittor, so torch-targeted libraries (transformers, peft, trl,
accelerate, LlamaFactory) run **unmodified** on the jittor / Ascend-NPU backend.

The torch-style API itself is supplied by `jittor/torch_compat.py` (installed
onto the jittor module at import). This package only re-exports it as `torch`
and wires the `torch.<submodule>` paths libraries expect.

## Layout

- `torch__init__.py` — the shim package body. Deploy as
  `<site-packages>/torch/__init__.py`.
- `stubs/{torchvision,torchaudio,torchdata}/__init__.py` — permissive stub
  packages (meta-path finders fabricating submodules on demand) so import-time
  `import torchvision...` in transformers/LlamaFactory succeeds. Deploy each as
  `<site-packages>/<pkg>/__init__.py`.
- `torch_dist_info/METADATA` — fake `torch-2.11.0.dist-info` metadata so
  `importlib.metadata.version("torch")` (used by accelerate) resolves.

## Deploy

```bash
SP=$(python -c "import site; print(site.getsitepackages()[0])")
cp torch__init__.py            "$SP/torch/__init__.py"
for p in torchvision torchaudio torchdata; do
  mkdir -p "$SP/$p"; cp "stubs/$p/__init__.py" "$SP/$p/__init__.py"
done
mkdir -p "$SP/torch-2.11.0.dist-info"
cp torch_dist_info/METADATA "$SP/torch-2.11.0.dist-info/METADATA"
printf 'torch\n' > "$SP/torch-2.11.0.dist-info/top_level.txt"
```

After editing `torch_compat.py` (the real API surface), only `torch_compat.py`
needs reloading — it is imported live from the jittor source tree. After editing
`torch__init__.py`, re-copy it to site-packages.

## NPU dispatch (performance — read this)

jittor only runs ops on the Ascend NPU when `jt.flags.use_cuda == 1`; otherwise
everything executes on **CPU** (a 2048³ matmul is ~20s on CPU vs ~2ms on NPU, so
training appears to "work" but is ~1000× too slow). torch users never set this —
they call `.cuda()`/`.to(device)`, which here are device-agnostic no-ops. So
`torch_compat.install()` now forces `jt.flags.use_cuda = 1` whenever
`jt.compiler.has_acl` is true, making `import jittor as torch` dispatch to the
NPU by default.

**Sanity check it's really on the NPU:** during training `npu-smi info` must show
**GB-scale HBM** for the python process. If HBM sits at the ~100 MB context
baseline, ops are on CPU — something reset `use_cuda`.

Note: ACL has no float64 kernels, so `torch.tensor([1.0])` yields float32 (torch's
default) and `cumsum`/`cumprod` cast bool→int64 before the native op.

For throughput, prefer **fixed-length sequences** (`packing: true`): each distinct
input shape triggers a fresh JIT compile (~tens of seconds for the backward+optim
graph), so variable-length batches recompile every step. Packing compiles once.

## Running transformers / LlamaFactory

Required environment:

- Python 3.10+ env (transformers needs `types.UnionType`); we use conda env
  `jt-torch` (py3.11).
- `HF_DEACTIVATE_ASYNC_LOAD=1` — forces single-thread weight materialization so
  jittor Vars are created on the thread that holds the ACL context (worker
  threads have a null ACL context).
- `HF_HUB_OFFLINE=1`, `DISABLE_VERSION_CHECK=1` for offline runs.
- `trl<=0.24.0` for LlamaFactory (newer trl drops
  `AutoModelForCausalLMWithValueHead`). Install trl with `--no-deps` or it will
  pull real PyTorch and clobber this shim.

**Do NOT let pip install the real `torch` package into this env** — it overwrites
`torch/__init__.py` and the dist-info. If that happens: `pip uninstall torch`,
delete the real `torch/` dir + `torch-*.dist-info`, then redeploy this shim.

Validated end-to-end: real Qwen3-0.6B load + generate, and a full LlamaFactory
Qwen3 LoRA SFT (loss decreases, grads flow, LR schedule applied, adapter saved
and reloadable via `PeftModel.from_pretrained`). See
`jittor/test/test_torch_compat.py` for regression coverage.
