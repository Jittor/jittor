# Torch API implementation owners

Status: 7.03 family migration, 2026-09-08. Owner: compatibility maintainers.
Recheck when an installer adds an API or changes a native delegate.

Public API implementations live at module scope. Installation binds the same
objects and records the selected implementation's fidelity; it does not create
a new function for each activation. Native mathematical implementations remain
their delegates. A method can be imported from its physical owner and tested
directly against an activated frontend without rerunning its installer.

| Family | Implementation owner | Installation responsibility |
| --- | --- | --- |
| Tensor construction, concatenation, checks and legacy typed constructors | `installers/tensor/__init__.py` | Bind stable functions and classes; real independent Tensor types remain frontend-owned |
| Tensor views, updates, placement and arithmetic adaptation | `installers/tensor/method_api.py` | `tensor/methods.py` binds functions and snapshots native delegates |
| Tensor gradient access and backward | `installers/tensor/autograd_api.py` | Share the existing holder registry and native Var/Op graph |
| Tensor shape and reduction argument adaptation | `installers/tensor/shape_api.py` | Bind explicit reduction functions and snapshot the previous native methods |
| Neural-network extras and class implementations | `installers/nn/extra_api.py` | Publish templates through the existing installation-owned NN class adapter |
| Loss and pixel rearrangement layers | `installers/nn/loss_modules.py` | Reuse the chosen native functional implementations |
| Neural-network functionals and attention selection | `installers/nn/functional.py`, `nn/attention.py` | Bind stable function objects and initialize the installation's backend cache |
| Initializers | `installers/nn_init.py` | Publish the stable initializers and guarded namespace; retain native mathematics |
| Datasets, samplers, loaders and checkpoint fallback | `installers/data.py` | Create namespace modules and bind the module-owned implementations |

`get_install_context()` reads the active frontend's existing context. It never
creates an installation. Functions use the target namespace for frontend
policies and the native backend for shared mathematical operations. There is no
second module-global context or injected dictionary of closure variables.

Captured delegates live in read-only mappings under `InstallContext.state`:
`tensor_native_api`, `tensor_shape_api`, `nn_extra_native`, and
`nn_functional_native`. Each mapping explicitly names the delegates consumed by
its family. Capturing before rebinding prevents recursive calls back into the
adapter. The `sdpa_backend_cache` is mutable execution state owned by that same
installation, not by a public function closure.

NN templates are ordinary module-level classes. Independent frontends reuse
the existing `nn_class_adapter` and `nn_layer_adapters` cache, which create the
required subclasses of that installation's `Module`, apply its Tensor scope,
and preserve native parameter adoption. Those installation-specific type
factories are intentional; the mathematical methods remain on their physical
template owners. Legacy typed Tensor constructors are stable module-level
classes whose call path resolves the active frontend.

`fidelity_report()` returns the deterministic machine-readable records.
`fidelity_table(prefix=None)` generates the corresponding Markdown coverage
table. `register_api_bindings()` consumes explicit family name lists and records
the actual final callable, including native descriptors. Existing detailed
records survive when their implementation object is unchanged. Placeholder
transformations remain marked `unimplemented`; lifting a function does not
establish complete Torch semantics, device support, or performance parity.

The focused checks in `test_family_api_owners.py` cover installation-only
binding, import and pickle identity, allocation, dtype preservation, gradients,
views, initializers, data collation, and NN layer templates. This family batch
does not certify every Torch API. Remaining families such as scheduler,
optimizer and distributed APIs must be counted separately in the refactor board;
CUDA and utility installer migrations are validated by their respective owners.
