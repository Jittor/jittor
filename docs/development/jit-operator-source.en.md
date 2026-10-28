# JIT operator source contract

Jittor compiles an operator source file **twice**. The ordinary C++ build takes the non-`JIT`
branch; `OpCompiler` preprocesses the `JIT` branch, extracts the `jit_run` function body from
it, renames operator-local identifiers, and parses the fused function body as `KernelIR`.
Therefore **the code inside `jit_run` must satisfy both the C++ and the KernelIR rules**.

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
