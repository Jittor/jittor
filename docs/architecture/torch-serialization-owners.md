# Torch serialization implementation owners

Status: 7.03 family migration, 2026-09-08. Owner: serialization maintainers.
Recheck when a format, dtype, pickle global or native loader is added.

The compatibility serialization family is a normal package. Its facade only
binds public `load`/`save` and re-exports the optional SafeTensor installer.

| Owner | Responsibility |
| --- | --- |
| `serialization/portable.py` | Public load/save, tensor snapshots, portable payloads and map_location |
| `serialization/security.py` | Restricted Unpickler and the explicit safe-global allowlist |
| `serialization/torch_archive.py` | Per-reader Torch zip storage cache, tensor/Parameter reconstruction and stride bounds |
| `serialization/safetensors.py` | SafeTensor codecs, reader/slice types and transactional optional bindings |

Implementations and reader classes are defined at module scope. A Torch archive
reader owns its storage cache and source-device map; no process-global archive
or storage table is introduced. Source tensor snapshots use the existing native
fetch operation and synchronization boundary without migrating the original
sharing group. Dtypes, scalar shapes, Parameter identity on reconstruction and
map_location handling remain explicit.

Torch zip tensors are read using their storage offset, shape and strides;
negative strides and out-of-storage descriptions fail before construction.
The existing value-materializing behavior is retained: this does not establish
round-trip identity of shared storages or preserve a noncontiguous physical
layout in portable snapshots. Unknown storage dtype names are rejected rather
than silently interpreted as float32.

`weights_only=True` remains the default. Format detection itself now uses the
restricted Unpickler, so a malicious ordinary `.pt` pickle cannot execute during
the legacy-format probe. The native legacy reader has no injectable Unpickler
interface and performs several direct pickle loads before reconstructing its
global storage tables. That path therefore requires explicit
`weights_only=False`; standard supported Torch zip checkpoints keep restricted
loading. Native URL extensions likewise require this explicit opt-in, and
native `.pkl` fallback cannot bypass restricted decoding. Native delegates are
read from `get_install_context().state["core_native_api"]`, never from a public
function that may already have been rebound.

SafeTensor bindings are applied through one `runtime_hook` and transactional
attribute writes. Its original non-Torch reader is held in the context's
read-only `safetensors_native_api` capture. NumPy requests continue to return
NumPy arrays, while Torch requests preserve supported wide integers, BF16 and
requested device. Float8 remains explicitly unsupported. A binding failure
restores both third-party functions and the captured context state. Encoding
an unsupported dtype fails before opening an output file for replacement.

The family registers conservative fidelity records. It does not claim complete
Torch format/keyword compatibility, a new download implementation or new
hardware support. Focused checks cover import/pickle identity, file and bytes
round trips, wide dtypes, source views, Parameter/device metadata, zip strides,
malicious pickle refusal, explicit native opt-in, SafeTensor NumPy delegation
and injected patch rollback.
