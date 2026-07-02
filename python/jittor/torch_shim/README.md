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

## graphdeco gaussian-splatting

The shim can run the original `graphdeco-inria/gaussian-splatting` checkout
from a normal Python entrypoint. Add this generic bootstrap before the first
`import torch` in the entry script:

```python
from jittor.torch_shim import enable as _enable_torch_shim
_enable_torch_shim(project_root=__file__)
```

Then run the stock command:

```bash
CUDA_VISIBLE_DEVICES=1 \
  JITTOR_TORCH_RUNTIME_ROOT=/path/to/gaussian-splatting/.jittor_torch_runtime \
  python train.py -s /path/to/data -m /path/to/output --disable_viewer
```

When using the `jittor.torch_shim` entrypoint, set
`JITTOR_TORCH_RUNTIME_ROOT` before starting Python so Jittor's core cache,
temporary files and CUDA discovery are configured before `jittor` finishes
importing.

`enable()` is not gaussian-splatting-specific. It scans the local project tree
for native extension build roots (`setup.py`, `pyproject.toml`, `CMakeLists.txt`,
and declared `.cu/.cpp` sources), registers Jittor as `torch`, prepends the
discovered extension roots to `sys.path`, and builds setuptools extensions with
`setup.py build_ext --inplace` when their in-place `.so` outputs are missing,
older than their source inputs, or stamped for a different shim/toolchain.
Set `JITTOR_TORCH_SKIP_EXT_BUILD=1` to skip this check on warm runs.

For PyTorch numerical parity, the bootstrap disables fast-math for Jittor JIT
CUDA kernels (`--fmad=false --prec-div=true --prec-sqrt=true`) unless
`JITTOR_TORCH_KEEP_FAST_MATH=1` is set. PyTorch-style project extensions keep
the same nvcc math policy as `torch.utils.cpp_extension`: the shim passes arch
and include/link flags, but does not inject Jittor JIT math flags into project
CUDA files unless the project requested them.

Runtime state defaults to `.jittor_torch_runtime` under the project root:

- `HOME`, `JITTOR_HOME`, `TORCH_HOME`, `TMPDIR`, `XDG_CACHE_HOME`,
  `CUDA_CACHE_PATH`
- the deployed shim `site-packages`
- `JITTOR_TORCH_EXTENSIONS_DIR`

Set `JITTOR_TORCH_RUNTIME_ROOT` to choose another project-local runtime
directory. If Jittor's bundled CUDA is installed at
`~/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux`, the bootstrap exports `JTCUDA`,
`CUDA_HOME`, `nvcc_path`, `PATH` and `LD_LIBRARY_PATH` for it automatically.

The older helper script is still available when you want a no-edit wrapper. It
deploys `import torch` into a per-project site-packages, builds the original
PyTorch-style CUDA extensions, and keeps runtime artifacts under the
gaussian-splatting checkout:

```bash
CUDA_VISIBLE_DEVICES=1 \
  python/jittor/torch_shim/scripts/run_gaussian_splatting.sh \
  /path/to/gaussian-splatting \
  train.py -s /path/to/data -m /path/to/output --disable_viewer
```

Useful subcommands:

```bash
# Show the conda Python, deployed torch shim path, Jittor cache and CUDA state.
python/jittor/torch_shim/scripts/run_gaussian_splatting.sh /path/to/gaussian-splatting env

# Build the stock submodule extensions in-place:
#   submodules/simple-knn
#   submodules/diff-gaussian-rasterization
#   submodules/fused-ssim
python/jittor/torch_shim/scripts/run_gaussian_splatting.sh /path/to/gaussian-splatting build_ext

# Smoke-test the three extensions plus rasterizer backward.
JITTOR_GS_SKIP_EXT_BUILD=1 \
  python/jittor/torch_shim/scripts/run_gaussian_splatting.sh \
  /path/to/gaussian-splatting \
  python/jittor/torch_shim/scripts/check_gaussian_splatting.py
```

By default the script chooses the active conda Python, or
`~/miniconda3/envs/jt311/bin/python` when no conda env is active. Runtime state
defaults to `/path/to/gaussian-splatting/.jittor_gs_runtime`:

- `HOME`, `JITTOR_HOME`, `TORCH_HOME`, `TMPDIR`, `XDG_CACHE_HOME`,
  `CUDA_CACHE_PATH`
- the deployed shim `site-packages`
- `JITTOR_TORCH_EXTENSIONS_DIR`

Set `JITTOR_GS_RUNTIME_ROOT` to choose a different project-local runtime
directory. If Jittor's bundled CUDA is installed at
`~/.cache/jittor/jtcuda/cuda12.2_cudnn8_linux`, the script exports `JTCUDA`,
`CUDA_HOME`, `nvcc_path`, `PATH` and `LD_LIBRARY_PATH` for it automatically.

The generic bootstrap was validated with direct `python train.py` runs on a
clean checkout after only adding the two bootstrap lines above. It scanned the
three native extension roots (`simple-knn`, `diff-gaussian-rasterization`,
`fused-ssim`), rebuilt stale in-place `.so` outputs when the shim/toolchain stamp
changed, skipped them on the next warm run, and completed
`train.py --iterations 5 --eval`, producing `chkpnt5.pth`,
`point_cloud/iteration_5/point_cloud.ply`, `cameras.json`, `input.ply` and
`exposure.json`.

Validated on a clean gaussian-splatting worktree with the original submodule
`setup.py` files and original `import torch` sources:

```bash
run_gaussian_splatting.sh /home/zy/projects/gs-jittor-clean build_ext
run_gaussian_splatting.sh /home/zy/projects/gs-jittor-clean \
  train.py -s data/tiny_blender -m output/tiny_jittor_eval_smoke \
  --iterations 3 --test_iterations 3 --save_iterations 3 \
  --checkpoint_iterations 3 --disable_viewer --quiet --eval
run_gaussian_splatting.sh /home/zy/projects/gs-jittor-clean \
  render.py -m output/tiny_jittor_eval_smoke --quiet
run_gaussian_splatting.sh /home/zy/projects/gs-jittor-clean \
  metrics.py -m output/tiny_jittor_eval_smoke
```

The metrics run produced `results.json` and `per_view.json` with SSIM, PSNR and
LPIPS values, exercising torchvision transforms/save_image, `torch.hub`
downloads, legacy PyTorch `.pth` loading, VGG16 feature extraction, and the
three gaussian-splatting CUDA extensions.

Additional CUDA validation on `/home/zy/projects/gs-parity-work`:

- A same-input PyTorch/Jittor parity script compared 3DGS math helpers,
  `simple_knn._C.distCUDA2`, `fused_ssim` forward/backward, and
  `diff_gaussian_rasterization` wrapper/direct forward/backward. The report
  `/home/zy/projects/gs-parity-work/parity_report.json` passed; rasterizer image,
  depth, radii and gradients matched within the recorded tolerances.
- One-step train tracing with original GS inputs produced bitwise-identical
  scaling, opacity, SH DC features, rendered image, and image gradient. The
  remaining direct rasterizer backward deltas were at native CUDA noise level:
  xyz grad max abs `2.0463630789890885e-12`; rotation grad max abs
  `9.316762720470648e-15`.
- A 5-step checkpoint comparison against PyTorch 2.1.2 matched all saved model
  and optimizer fields under the validation tolerance except `model.rotation`.
  The Jittor-vs-PyTorch rotation max abs was `0.0036064726300537586`; two
  independent PyTorch-vs-PyTorch runs of the unmodified GS CUDA rasterizer
  differed by `0.004010355216450989` on the same field. The residual rotation
  drift is therefore inside the original rasterizer's own nondeterministic
  boundary rather than a shim-specific mismatch.
- The default one-click command, without `JITTOR_GS_SKIP_EXT_BUILD`, invoked the
  stock `setup.py build_ext --inplace` in all three submodules and completed
  `train.py --iterations 5 --eval`, producing `chkpnt5.pth`, `point_cloud.ply`,
  `cameras.json`, `input.ply` and `exposure.json`.
- With warm Jittor caches, the same 5-iteration tiny-scene training run took
  `real 3.80s` under the Jittor shim versus `real 3.72s` under PyTorch 2.1.2
  on the same RTX 4090 and dataset. A cold first run includes one-time Jittor
  core/kernel compilation and extension build time.
- `render.py` and `metrics.py` on the Jittor output generated train/test PNGs,
  `results.json` and `per_view.json`. The `ours_5` metrics were:
  SSIM `0.009542092680931091`, PSNR `6.742209434509277`, LPIPS
  `0.6262948513031006`.

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

## mmdetection / computer-vision ops

The shim also covers the **torch operator surface used by mmdetection** (analysed
against v3.3.0): `torch.*`, `torch.nn.*`, `torch.nn.functional.*` and `Tensor`
methods. The vast majority were already supported by `torch_compat.py`; the
detection-specific gaps that were filled:

- **torch.\***: `mm`, `masked_select`, `split_with_sizes`, `nan_to_num_`,
  `_shape_as_tensor`, and `sparse_coo_tensor` (+ `torch.sparse.sum` / `.to_dense()`,
  a dense-backed hybrid-COO used by the free-anchor head).
- **nn / F**: `nn.SyncBatchNorm` (+ `convert_sync_batchnorm`, a single-device
  no-op), `F._Reduction.get_enum`, `F.adaptive_max_pool2d`, `F.relu_`,
  `F.upsample_bilinear`.
- **Tensor methods**: `.relu`, `.eq/.ne/.gt/.ge/.lt/.le`, `.clamp_min/.clamp_max`,
  `.neg`, `.reciprocal`, `.bmm/.mm`, `.diff`, `.fliplr/.flipud`, `.fmod/.remainder`,
  `.softplus`.
- **import-time submodule paths** mmdet pulls layer internals from:
  `torch.nn.modules.{utils,batchnorm,normalization,activation}` (e.g.
  `from torch.nn.modules.utils import _pair`, `from torch.nn.modules.batchnorm
  import _BatchNorm`), plus resolution stubs for the *modules that contain the
  operators* (`torch.onnx.is_in_onnx_export`, `torch.multiprocessing`,
  `torch._utils._flatten_dense_tensors`, `torch.hub`).

See `jittor/test/test_mmdet_ops.py` for regression coverage (existence sweep over
the full extracted surface + numeric/identity correctness checks, CPU and CUDA).

### Out of scope: mmcv native ops (a SEPARATE package, not in this repo)

mmdetection's heavy custom ops come from **`mmcv.ops`** — NMS / `batched_nms`,
`RoIAlign` / `RoIPool`, `DeformConv` / `ModulatedDeformConv`, `(Modulated)DeformRoIPool`,
`MaskedConv`, `CARAFE`, `CornerPool`, `MultiScaleDeformableAttention`,
`point_sample`, `sigmoid_focal_loss`, … These are **not** torch operators (they are
C++/CUDA kernels mmcv compiles against libtorch's ABI), so they are **intentionally
NOT part of jittor**. The jittor-native reimplementations live in a separate
adapter package, `mmcv_jittor` (`mmcv_compat.py` + `mmcv_ops_jt/`), kept outside
this repo — call `mmcv_compat.install()` before `import mmdet`.
