# JIT operator source contract

Jittor compiles an operator source file twice. The ordinary C++ build uses the
non-`JIT` branch, while `OpCompiler` preprocesses the `JIT` branch, extracts its
`jit_run` body, renames operator-local identifiers, and parses the fused body as
`KernelIR`. Code in `jit_run` must therefore follow both C++ and KernelIR rules.

## Supported source

- Put headers and reusable declarations outside `jit_run`. Includes are copied
  to the fused translation unit, but an include is not a statement in the
  generated kernel body.
- Use normal C++ quoted strings and character literals. Escapes are supported;
  their contents, including `%`, braces, parentheses, and identifier-like text,
  are copied without identifier renaming.
- `_Pragma("...")` is supported immediately before a statement. KernelIR keeps
  the pragma as raw source and parses the following loop or statement normally.
- Keep CPU-only and CUDA-only directives behind `JIT_cpu` / `JIT_cuda`. A
  directive accepted by one compiler is not automatically valid for the other.
- Operator-local names may be renamed when fused. Global helper types and
  functions must be declared by an included header or registered as reserved
  identifiers in `op_compiler.cc`.

Do not hide unmatched braces or parentheses in preprocessor tricks. Prefer a
small helper declared outside `jit_run` when a construct cannot be represented
as a regular statement or loop.

## Diagnostics

The fused source contains a `#line` directive before each extracted operator
body. C++ compiler diagnostics therefore point back to the operator source file
and line instead of only naming the generated cache file. KernelIR parser
failures report the line within the source fragment they were parsing; malformed
`_Pragma` directives report the directive line explicitly.
