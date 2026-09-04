# torch.vmap Owner Plan

## Current owner

`python/jittor/compat/torch/installers/numerical.py:1557-1647` currently owns
the compatibility implementation inside `install(ctx)`. The implementation has
two coupled pieces:

- `_vectorized_getitem_vmap` handles nested pointwise maps used by transformer
  mask construction. It depends on `jt.Var` shape/dtype checks and the runtime
  flag `g._transform_getitem_to_index_depth`.
- `_vmap` builds nested mapping specifications, applies the vectorized fast path,
  or loops over mapped axes and stacks the results. Its wrapped callable stores
  `_jittor_vmap_base` and `_jittor_vmap_specs` metadata.

There is no native `jt.vmap` owner. The public binding is currently the
`_alias("vmap", _vmap)` call in the same installer.

## Migration boundary

Do not extract only `_vmap`: the vectorized helper and runtime flag access are
part of the same contract. The eventual module-level owner should receive an
explicit context object (or a narrow callback for the transform-depth probe)
instead of reading a module-global `g`. `install(ctx)` should only bind that
stable callable to `ctx.jittor_module.vmap`.

Keep the existing mapping semantics unchanged: `in_dims`/`out_dims`, nested
spec metadata, scalar-to-singleton shape normalization, and the bool broadcast
fast path. Do not introduce a no-op fallback or a process-global runtime state.

## Focused acceptance

Before broad tests, add CPU-only nodes covering:

1. module identity and fidelity registration (`torch.vmap` is the module-level
   owner and is marked approximate with the context limitation documented);
2. a simple mapped unary function with `in_dims=0` and an explicit `out_dims`;
3. nested mapped boolean mask construction exercising the vectorized helper.

Run these nodes with an isolated Jittor cache. GPU/NPU behavior remains deferred
until a backend with the required transform integration is available.
