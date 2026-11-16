# JIT operator source contract

[Section contents](../index.en.md) · [中文](jit-operator-source.md)

For contributors writing or changing C++ JIT operators. Read [operator registration and dispatch](operator-dispatch.en.md) first; this page covers source syntax, while [JIT cache](../testing/jit-cache.en.md) covers compilation reuse.

An operator has a host implementation and a kernel template. The ordinary C++ build
compiles construction, shape inference, gradients, and `jit_prepare`; `OpCompiler`
preprocesses the kernel template, extracts `jit_run`, renames operator-local identifiers,
and parses the fused body as `KernelIR`. Therefore **the code inside `jit_run` must satisfy
both the C++ and the KernelIR rules**.

## File organization and incremental builds

For `xxx_op.h`, registration prefers a sibling `xxx_op.jit.cc` as the kernel template.
`xxx_op.cc` retains the host implementation. `.jit.cc` files are excluded from host
compilation and from runtime flag and C++ test declaration scans. Unary, binary,
reduction, broadcast, and ternary operators use this layout.

For custom operators, `compile_custom_ops([header, source])` also discovers the sibling
template; callers may explicitly include it in the file list. Without a companion,
registration uses the original `.cc`, including the legacy `#ifndef JIT` / `#else` form.
After adding or removing a template, reload the operator library in a new process.

Fused custom operators retain each constituent's registered include, macro-definition,
and link options; those options also participate in compilation-cache validation. Use
qualified names such as `::function(...)` for global helpers declared in headers so they
are not renamed as operator-local identifiers.

Adding or removing host-source lines no longer changes the kernel's `#line`, allowing
unchanged kernels to be reused. Templates must still include the headers they need;
changes to actual templates, headers, compiler options, or generated code still invalidate
the relevant cache entries. Do not remove source locations or dependency checks to obtain
cache reuse. `tests/build/test_jit_companion_sources.py` covers these boundaries.

## Supported forms

- **Headers and reusable declarations go outside `jit_run`.** An include is copied into the
  fused translation unit, but **an include is not a statement in the generated kernel body**.
- Use ordinary C++ quoted strings and character literals. Escapes are supported; their
  contents -- including `%`, braces, parentheses and identifier-looking text -- are copied
  verbatim and are not subject to identifier renaming.
- `_Pragma("...")` immediately preceding a statement is supported. KernelIR keeps the pragma
  as raw source and parses the following loop or statement normally.
- **CPU-only or CUDA-only instructions go after `JIT_cpu` / `JIT_cuda`.** An instruction one
  compiler accepts is not automatically valid for the other.
- Operator-local names may be renamed during fusion. **A global helper type or function must
  be declared by an included header, or registered as a reserved identifier in
  `op_compiler.cc`.**

**Do not hide unmatched braces or parentheses in preprocessor tricks.** When a construct
cannot be expressed as an ordinary statement or loop, declare a small helper function outside
`jit_run` instead.

## Diagnostics

The fused source has a `#line` directive before every extracted operator body, so C++
compiler diagnostics **point back at the operator source file and line** instead of only
naming the generated cache file. A KernelIR parse failure reports the line within the fragment
it was parsing at the time; a malformed `_Pragma` directive explicitly reports the line that
directive is on.
