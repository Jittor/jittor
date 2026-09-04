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

## Verifiable contract

The module-level owner should satisfy these static/runtime checks before any
behavioral optimization:

- `torch.vmap is numerical.vmap`, with `__module__` and `__name__` identifying
  the stable owner; fidelity is registered as `approximate` and documents the
  injected context dependency.
- `install(ctx)` performs binding only. It must not create a new closure per
  install or retain the context in a process-global variable. Repeated installs
  must leave the same callable identity in place.
- The context callback exposes only the transform-depth probe. The callable
  must remain usable when that probe is false or absent, taking the ordinary
  loop/stack path.
- For every mapped argument, `in_dims=None` leaves the argument unchanged;
  mapped dimensions preserve their extent, and `out_dims` is applied exactly
  once. Nested calls append one mapping spec and retain the existing metadata
  attributes.
- Unsupported combinations (mismatched mapped extents, non-zero nested helper
  dimensions, or non-bool vectorized results) return to the ordinary path or
  raise a documented `RuntimeError`; they must never silently become a no-op.

Static checks should assert that no `g`/`ctx` object is captured by a module
global and that install contains only the stable binding. These checks are
independent of backend availability and can run without JIT compilation.
