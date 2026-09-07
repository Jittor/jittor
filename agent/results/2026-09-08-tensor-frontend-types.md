# Tensor frontend types

- Status: native subtype and return-type boundary implemented; 7.12 remains open
- Baseline: `15b1cef72`
- Date: 2026-09-08
- Owner: coord
- Review when: pyjt object allocation, frontend installation or callbacks change

Native Var now permits real Python subclasses. Input conversion, NumPy array
conversion, RingBuffer serialization and gradient callbacks accept those
subclasses without changing the VarHolder payload or the unified Var/Op graph.
The generated heap-base destructor releases the actual instance type, and GC
visits dynamic subtype references, including cycles through class attributes.

A shared binding implementation owns a ContextVar selecting the Python output
type. `_set_tensor_frontend_type(type)` returns a reset token and
`_reset_tensor_frontend_type(token)` restores it. Factories can select a type
explicitly; generated tensor bindings infer it from self, positional arguments,
keyword arguments and a single list/tuple level for tensor-vector signatures.
Explicit selection wins; otherwise the first subtype wins. Ordinary numeric
array data is not recursively scanned. Allocations bypass subclass `__init__`
and use the existing holder move, dictionary, initialization and submission
sequence. Python-only subclass fields are therefore not initialized on results.

Gradient callbacks retain their creation type and restore it only while calling
Python, even when the original factory scope has already exited. Native
`jt.grad` now accepts a subclass as its single target. No independent Torch
Tensor or Module is installed yet; shared native class mutation and autograd
policy still need migration. Parameter-specific output-type selection also
belongs to that frontend installation work.

Verification used three focused cases in `test_pyjt_binding_protocol.py`:
subtype conversion and class/instance/slot-cycle release; arithmetic, reshape,
factory scope nesting and restoration; custom Function backward after leaving
the creation scope. CPU: 3 passed in 26.34 s including incremental core build.
CUDA: 3 passed in 9.08 s after incremental core build, with CUDA 12.2/sm89 and
both `has_cuda` and `flags.use_cuda` asserted before running. Independent CPU
and CUDA caches were used. The first run exposed the old exact-type `jt.grad`
check and failed; the final runs passed after its fix. Generated bindings were
inspected, and the standalone rendering helper was updated for the finalized
source layout. Full suites, wheel rebuild and NPU execution were not run.
