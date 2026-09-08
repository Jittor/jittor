# Distributed API implementation ownership

Status: 7.03 family migration, 2026-09-08. Owner: distributed compatibility maintainers.
Recheck when a process-group API, collective route or checkpoint adapter changes.

`compat/torch/installers/distributed.py` defines its public functions and class
implementations at module scope. Installation creates namespace objects and
binds those implementations; it does not manufacture new public function
closures. Historical Torch namespace paths retain the same underlying native
ProcessGroup, Work and Store classes.

`InstallContext.state["distributed_api"]` is a read-only mapping containing the
installed distributed namespace, WORLD object, group map and initialization
state. The initialization state's store and flag, and the group map itself,
remain mutable under their existing APIs. Functions read the active context
through `get_install_context()`. No second global context or process group is
introduced. GroupMember/group namespace types still carry the installation's
actual WORLD object; they are namespace publication, not a replacement group
implementation.

Group construction still delegates communicator creation to the native
ProcessGroup. Explicit group reductions still call that group's implementation;
the legacy WORLD routes keep their native collective spelling, reduction
selection, output updates and Work return behavior. Dynamic NCCL bootstrap
installs stable Var method implementations which read the selected ops and
current world size from their existing sole owner, `compile_extern`.

Checkpoint refusal functions are explicit module-level functions. They use the
same unsupported-operation policy, including its opt-in fallback, and retain
their `_jittor_unimplemented` marker. They no longer rely on a closure whose
name was rewritten to resemble an importable function. RPC, join, symmetric
memory and sharded-storage placeholders retain their behavior and are recorded
as `unimplemented`; they are not presented as completed communication support.

Fidelity records describe final bindings after FSDP composition, so a real
installed mesh owner is distinguished from the singleton placeholder. No
collective algorithm, NCCL/HCCL optimization, asynchronous execution model or
multi-machine behavior is introduced by this migration.

`test_distributed_api_owners.py` checks module/pickle identity, native group and
Work identity, simulated rank/group creation, reduction and gather/broadcast
parameter routing, store cleanup, backend validation and checkpoint refusal.
The simulated routes do not certify actual multi-rank CUDA or Ascend execution;
real hardware verification remains a separate integration batch.
