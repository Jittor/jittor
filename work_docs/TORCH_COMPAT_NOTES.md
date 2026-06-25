# jittor-as-torch compatibility — progress notes

Goal (TODO.md #2,#3,#4): `import jittor as torch` runs unmodified PyTorch code;
run transformers + LlamaFactory (Qwen3-8B) on jittor/Ascend.

## Environment
- conda env `jt-torch` (Python 3.11) -- transformers needs >=3.10, LlamaFactory >=3.11.
  Activate: `source ~/jt_torch_env.sh` (conda activate jt-torch + CANN env).
- jittor installed editable from /home/yizhang/projects/jittor.
- `torch` shim package deployed to the env site-packages
  (.../jt-torch/lib/python3.11/site-packages/torch/__init__.py). A reproducible
  copy is committed at python/jittor/torch_shim/torch__init__.py — redeploy with:
    cp python/jittor/torch_shim/torch__init__.py <site-packages>/torch/__init__.py
- transformers run via PYTHONPATH=/home/yizhang/projects/transformers/src, HF_HUB_OFFLINE=1.

## Architecture
- python/jittor/torch_compat.py: augments the jittor module namespace with
  torch API (dtypes, device, Tensor/tensor, no_grad decorator, constructors
  wrapped for device=/dtype objs, nn.functional, sdpa, nn.init torch names,
  Module.forward<->execute bridge + torch Module methods, in-place tensor ops,
  finfo/iinfo, cuda shim, ...). Installed at end of jittor/__init__.py.
- torch_shim: the `torch` package; re-exports jittor + wires torch.<submodule>
  (nn/optim/utils.data/distributed/distributions/compiler/library/jit/_dynamo/
   amp/autograd/fx/version/_C ... mostly stubs for import-time references).

## Status
- DONE: transformers 5.13.0.dev0 imports; LlamaModel forward works; LlamaForCausalLM
  TRAINS (loss decreases, grads flow, optimizer updates).
- KEY BUG FIXED: in-place init (.assign) adopted source's stop_grad and froze all
  params -> zero grad. Now grad-preserving. (commit 7611b6e6)
- Core C++ change: NanoString PyObject converter accepts str subclasses
  (py_converter.h) so torch dtype objects work as jittor dtype strings.

## Next
- Test model.generate(); load a real HF checkpoint (safetensors).
- Then LlamaFactory Qwen3-8B finetune (TODO#4).
- TODO#1 (max model on 8 cards) deferred: cards busy with another user's
  llama-mpi-serve (~30GB/card); revisit when free (npu-smi).
- Keep adding missing torch ops as models surface them; add unit tests.
