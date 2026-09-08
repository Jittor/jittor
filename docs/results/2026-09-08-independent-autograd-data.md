# Independent autograd and data boundaries

- Status: autograd ownership and default collation migrated; 7.12 open
- Baseline: `1b484894f`
- Date: 2026-09-08
- Owner: coord
- Review when: Function composition, autograd publication or collation changes

Independent installation now creates its own autograd namespace, functional
namespace and Function subclass. Native Function and autograd dictionaries
remain unchanged. The frontend Function explicitly bridges execute to its
subclass's forward; this previously worked only because native Module.execute
had been patched by the legacy installation. Context recording, saved tensors
and backward reuse the existing implementation and unified graph.

Default collation constructs frontend tensors. NumPy arrays/scalars retain
their dtype, Python integers/floats use int64/float64, and booleans use bool.
This avoids both native-Var results and automatic narrowing of wide integers.
Tensor inputs are stacked under the frontend allocation scope without
detaching their graph. A batch of stopped inputs explicitly produces a stopped
result, accounting for native stack's allocation path. Worker scheduling and
the existing thread-based worker implementation were not changed.

Verification: seven focused CPU cases passed in 3.78 s; two real CUDA
integration cases passed in 5.99 s. They cover native Function/autograd binding
identity, custom forward/save_for_backward/backward through autograd.grad,
collation gradients, Python and NumPy dtype behavior including integers above
2**45, and the preceding independent deployment/training/retry chain. Four
existing custom-Function cases also passed. The legacy autograd file reported
four retained Vars at session end; this is not a full liveness audit. No full
suite, wheel rebuild or NPU execution was performed.

Remaining independent API/model and storage/thread boundaries still belong to
7.12. This change does not establish complete DataLoader or autograd API fidelity.
