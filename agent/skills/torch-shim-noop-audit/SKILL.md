---
name: torch-shim-noop-audit
description: Decide whether a torch API in Jittor's compatibility layer really works or is a signature-complete no-op, then convert it into an implementation or an explicit NotImplementedError with a negative test. Use when auditing, fixing, or extending `python/jittor/compat/` — especially when an API "runs fine" but the numbers, the log file, the checkpoint or the speed-up are missing.
---

# Is this torch API real, or just a shape?

The compat layer's most expensive defect class is not a crash. It is an API that
accepts every argument the real one accepts, returns a plausible value, and does
nothing — `torch.autocast` that trains in fp32, `load_state_dict` that reports no
missing keys for a checkpoint with every key wrong, `dcp.save()` that writes zero
bytes and returns success. The program finishes. The numbers are wrong. Nothing
in the output says so.

This is the method used to work through that list (task 7.01). It is written for
the next person adding an API to it.

## 1. Three-question test

For any `torch.X` in `python/jittor/compat/`:

1. **Does the body read every argument it accepts?** An argument stored on
   `self` and never read again is the signature of a stub. `DataLoader.__init__`
   assigned `self.num_workers`, `self.prefetch_factor`, `self.worker_init_fn`
   and then `__iter__` built `_SingleProcessDataLoaderIter` unconditionally.
2. **Does its return value depend on its inputs?** `return _IncompatibleKeys([], [])`
   as the last line, `lambda state_dict=None, *a, **k: state_dict`,
   `lambda *a, **k: False`, `lambda t, *a, **k: t` — a constant or an identity
   with a rich signature is the tell.
3. **Would a user notice it did nothing?** This is the priority sort. Rank
   *silently wrong numbers* (autocast, DTensor.full_tensor, backward(gradient=))
   above *silently lost data* (dcp.save, SummaryWriter) above *inaccurate query
   interfaces* (is_autocast_enabled, current_device). Fix in that order.

Fast greps that find candidates:

```bash
cd python/jittor/compat
grep -rn "lambda \*a, \*\*k: \(None\|False\|True\|0\.0\)" --include=*.py
grep -rn "lambda [a-z_]*, \*a, \*\*k: \1" --include=*.py      # identity
grep -rn "^\s*pass$" --include=*.py -B3 | grep -i "class .*Iter\|class .*Store"
grep -rn "(ignored)\|no-op\|best-effort" --include=*.py
```

Then confirm at runtime rather than by reading — see §4 for how to import the
right tree.

## 2. Implement, or refuse. Never leave it silent.

Order of preference:

1. **Implement** when the machinery already exists somewhere in the tree. It
   usually does, one file away: `tree_map` was `lambda f,x: f(x)` sitting six
   lines below a real recursive `_tree_flatten`/`_tree_unflatten`;
   `torch.autocast` mapped one-to-one onto jittor's existing `jt.flags.amp_reg`;
   `load_state_dict`'s key diff is `model.state_dict().keys()` minus the given
   keys; `Tensor.backward(gradient=v)` is `(y*v).sum().backward()`.
2. **Implement the exact special case, refuse the rest.** An all-reduce on one
   rank IS the identity; a `Replicate()` DTensor's `full_tensor()` IS the local
   tensor; `set_device(0)` IS a no-op. Keep those exact and raise only for the
   inputs that would actually be wrong (`world_size > 1`, `Shard(...)`,
   `set_device(i != 0)`). This is what keeps single-process test suites green
   while closing the real hole.
3. **`unimplemented(api, effect, hint)`** otherwise — from
   `jittor/compat/stub_policy.py`. `effect` completes the sentence "running it
   as a no-op would ..." and must name the *damage*, not the gap: not "is not
   supported" but "write no bytes at all while reporting a successful save".
4. **`degraded(api, difference, hint)`** for APIs that do work but not torch's
   way (DataLoader workers are threads, not processes; `checkpoint()` recomputes
   nothing). Warns once, never raises.

`allow_stub` is the escape hatch, and its design matters: **off by default**,
opened by `JITTOR_TORCH_ALLOW_STUB=1` or `torch.compat_allow_stub(True)`, and
when open it warns **once per API** and returns the caller-supplied
`stub_result` — so old behaviour is exactly recoverable, and the recovery is on
the record. Every refusal message names the switch, so the error itself tells
the user how to get their old (wrong) run back.

## 3. The negative test is the deliverable

One test per API, in `tests/compat/torch/test_torch_compat_unimplemented.py`.
Two shapes only:

- **Refused**: `assertRefuses(fn, "torch.X", "<damage phrase>")` — asserts
  `NotImplementedError`, that the message names the API and the consequence, and
  that it documents the env var. Pair it with `assertStubFallback(fn)`, which
  asserts the hatch restores the old value *and warns*.
- **Really takes effect**: assert the observable that was missing, never that
  the call returned. `autocast` → `(a @ b).dtype == "float16"`;
  `dirac_` → the convolution reproduces its input; `update_bn` → `running_mean`
  moved from 99.0 to the data mean; DataLoader workers → the batches carry a
  *thread id different from the caller's*; `Event.elapsed_time` → `> 5.0` after
  a 20 ms sleep.

Both shapes must **fail before the fix**. A test that asserts
`self.assertIsNone(writer.add_scalar(...))` passes against the stub and is
worthless.

Expect to find existing tests asserting the *wrong* behaviour, and change them
in the same commit, saying so in the message.

## 4. Traps specific to this layer (each cost real time)

- **`torch is jittor`.** Importing the wrong tree means testing a different
  shim than the one you edited, and every conclusion inverts. In a worktree,
  `pytest` is correct (`tests/conftest.py` puts this checkout first) but any
  hand-run `python -c` / script / subprocess needs
  `PYTHONPATH=<worktree>/python` explicitly, because the conda env has jittor
  installed editable against the *main* tree. Self-check before believing
  anything: `python -c "import jittor,os;print(os.path.dirname(jittor.__file__))"`.
- **The Write/Edit tools may not reach the box's filesystem** for paths outside
  the primary project tree. Create and patch files with Bash heredocs, and
  verify with `ls` before assuming an edit landed.
- **Never `cd` into `compat/torch/` to run python**: `types.py` there shadows the
  stdlib `types` module and the interpreter dies during startup. Run patch
  scripts from a neutral directory with absolute paths.
- **`torch.dtype` is a `str` subclass** whose `__str__` returns *itself* and
  whose `.split(".")` is special-cased to `["torch", name]`. Normalise with
  `str(d).split(".")[-1]`, and remember it pickles as a class reference — an
  allowlisting unpickler must permit `jittor.compat.torch.types.dtype`.
- **A `dict`/`set` of "safe" pickle globals is per-format**: our own
  `torch.save` output is a plain pickle, real `.pt` files are zip archives with
  persistent-id storages, and legacy `.pth` is a third path. Wire `weights_only`
  through all of them or one path stays unguarded.
- **Flags are process-global.** `jt.flags.amp_reg`, `jt.flags.use_cuda`,
  `jt.world_size` — save and restore them in `setUp`/`tearDown` or the next
  test in the session inherits your setting and fails somewhere unrelated.

## 5. Checklist for one API

```
[ ] confirmed at runtime (not by reading) that it is a no-op, in the right tree
[ ] ranked: wrong numbers > lost data > inaccurate query
[ ] implemented, or exact-case implemented + refused elsewhere, or unimplemented()
[ ] message names the API and the damage, not just "unsupported"
[ ] negative test added; verified it FAILS without the fix
[ ] stub-fallback test added for refusals
[ ] existing tests that asserted the old silent behaviour updated, and called
    out in the commit message
[ ] full tests/compat/torch/ run compared against a pre-change baseline
```
