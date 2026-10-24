"""SOURCE-ONLY 452 resident backward with Native-style deferred micro gradients.

Resident extra U storage is never freed. Ordinary non-reentrant Tensor.backward
only; one native grad_optional, original shared reducer, no optimizer discovery
of participation. No CUDA/semantic approval follows from source tests.
"""
from contextlib import contextmanager
import threading
import weakref

_states = weakref.WeakValueDictionary()
_local = threading.local()


def selected(state):
    return (state is not None
            and getattr(state, 'frontend_type', None) is not None
            and getattr(state, 'true_fsdp_initialized', False)
            and getattr(state, 'true_fsdp_world_size', None) == 1)


def owners(state):
    if not selected(state):
        raise RuntimeError('resident FSDP owner not selected')
    if getattr(state, '_resident_poisoned', False):
        raise RuntimeError('resident FSDP task previously failed')
    if getattr(state, 'true_fsdp_flat', False):
        raise NotImplementedError('resident FSDP flat parameters')
    if type(getattr(state, 'requires_gradient_sync', True)) is not bool:
        raise TypeError('resident gradient sync policy must be bool')
    if getattr(state, 'true_fsdp_pending_full_grads', None) is not None:
        raise RuntimeError('resident owner cannot adopt legacy pending gradients')
    if getattr(state, 'reshard_after_backward', True) is not True:
        raise NotImplementedError('resident FSDP requires reshard_after_backward=True')
    from . import shard, grad_sync, common
    from ..torch.tensor_state import compatibility_owner
    torch = compatibility_owner(shard.jt)
    if torch.Var is not state.frontend_type:
        raise TypeError('resident FSDP frontend owner')
    policy = getattr(state, 'mp_policy', None)
    dtype = getattr(policy, 'param_dtype', None)
    for entry in state.true_fsdp_params:
        if dtype is not None and shard._storage._jittor_dtype_name(dtype) != shard._storage._jittor_dtype_name(entry.shard.dtype):
            raise NotImplementedError('resident FSDP mixed parameter dtype')
    return torch, shard, grad_sync, common


def unshard(module):
    state = module._fsdp_state
    torch, shard, _, _ = owners(state)
    _states[id(state)] = state
    if state.true_fsdp_unsharded:
        return module
    for entry in state.true_fsdp_params:
        c = entry.shard
        u = getattr(entry, 'resident_full_param', None)
        if u is None:
            with torch.no_grad():
                value = c.to_local().detach().clone()
            u = torch.nn.Parameter(value, requires_grad=entry.requires_grad)
            if u is c or shard.dtensor.is_dtensor(u):
                raise RuntimeError('full parameter must be a new plain Parameter')
            entry.resident_full_param = u
        else:
            if tuple(u.shape) != tuple(c.shape) or u.dtype != c.dtype or u.requires_grad != entry.requires_grad:
                raise RuntimeError('resident full parameter changed')
            with torch.no_grad():
                u.data.copy_(c.to_local().detach())
        entry.full_param = u
        shard._bind_entry(entry, u)
    state.true_fsdp_unsharded = True
    return module


def reshard(module):
    state = module._fsdp_state
    _, shard, _, _ = owners(state)
    for entry in state.true_fsdp_params:
        shard._bind_entry(entry, entry.shard)
    state.true_fsdp_unsharded = False
    return module


class Forward:
    def __init__(self):
        self.members = {}
        self.handles = []
        self.closed = False
        self.failed = False
        self.consumed = False

    def reached(self, gradient):
        task = current()
        if task is None or not self.closed or self.failed or self.consumed:
            raise NotImplementedError('resident output hook requires its ordinary backward task')
        for ref in self.members.values():
            state = ref()
            if state is None:
                raise RuntimeError('reached forward owner was destroyed')
            owners(state)
        task.reached[id(self)] = self
        task.queue_finish()
        return None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class InputEdge:
    """One real forward invocation's grouped differentiable-input edge."""
    def __init__(self, state, root, count):
        self.state = weakref.ref(state)
        self.root = root
        self.count = count
        self.arrived = False

    def reached(self, gradients):
        task = current()
        state = self.state()
        root = self.root
        if (task is None or task.published or self.arrived or state is None
                or not root.closed or root.failed or root.consumed
                or task.reached.get(id(root)) is not root
                or root.members.get(id(state), lambda: None)() is not state):
            raise RuntimeError('resident input edge has no unique reached task owner')
        if type(gradients) is not tuple or len(gradients) != self.count:
            raise RuntimeError('resident input edge gradient inventory changed')
        owners(state)
        self.arrived = True
        task.input_arrivals.append(self)
        return None  # GradHooker returns the original gradient tuple unchanged.


def wrap_inputs(frame, args, kwargs):
    state, root = frame
    torch, _, _, _ = owners(state)
    if not torch.is_grad_enabled():
        return args, kwargs
    pytree = torch.utils._pytree
    flat, spec = pytree.tree_flatten((args, kwargs))
    indices = [i for i, value in enumerate(flat)
               if isinstance(value, torch.Tensor) and value.requires_grad]
    if not indices:
        return args, kwargs  # Root task finalization handles this real branch.
    from jittor._core.hooks import grad_hooker
    edge = InputEdge(state, root, len(indices))
    values = grad_hooker(tuple(flat[i] for i in indices), edge.reached)
    if len(values) != len(indices):
        raise RuntimeError('resident grouped input edge changed forward inventory')
    for i, value in zip(indices, values):
        flat[i] = value
    return pytree.tree_unflatten(flat, spec)


def forward_enter(module):
    state = module._fsdp_state
    owners(state)
    if current() is not None:
        raise NotImplementedError('resident forward inside backward/reentrant task')
    stack = getattr(_local, 'forward', None)
    if stack is None:
        stack = _local.forward = []
    root = stack[0][1] if stack else Forward()
    root.members[id(state)] = weakref.ref(state)
    frame = (state, root)
    stack.append(frame)
    return frame


def forward_exit(frame, output, success):
    stack = getattr(_local, 'forward', ())
    if not stack or stack[-1] is not frame:
        raise RuntimeError('resident forward owner mismatch')
    state, root = frame
    try:
        if success:
            torch, _, _, common = owners(state)
            if torch.is_grad_enabled():
                seen = set()
                for tensor in common._tensor_values(output):
                    if tensor.requires_grad and id(tensor) not in seen:
                        seen.add(id(tensor))
                        root.handles.append(tensor.register_hook(root.reached))
        else:
            root.failed = True
    except BaseException:
        root.failed = True
        raise
    finally:
        stack.pop()
        if not stack:
            root.closed = not root.failed
            if root.failed:
                root.remove()


def current():
    return getattr(_local, 'task', None)


class Task:
    def __init__(self, states):
        self.states = tuple(states)
        self.targets = []
        self.target_entries = {}
        self.canonical_ids = set()
        self.reached = {}
        self.input_arrivals = []
        self.published = False
        self.completed = False
        self.finish_queued = False
        for state in states:
            for entry in state.true_fsdp_params:
                self.canonical_ids.add(id(entry.shard))
                u = getattr(entry, 'resident_full_param', None)
                if u is not None and entry.requires_grad:
                    if id(u) in self.target_entries:
                        raise RuntimeError('full Parameter owned by two FSDP entries')
                    self.targets.append(u)
                    self.target_entries[id(u)] = (state, entry)
        self.owned_ids = self.canonical_ids | set(self.target_entries)

    def queue_finish(self):
        if not self.finish_queued:
            from ..torch.autograd import Variable
            Variable._execution_engine.queue_callback(self.finish)
            self.finish_queued = True

    def validate_optimizers(self, optimizers):
        # Validate slot ownership only; never infer participation from optimizers.
        for opt in optimizers:
            for group in opt.param_groups:
                for p in group['params']:
                    if id(p) in self.target_entries:
                        raise NotImplementedError('optimizer must own canonical parameters, not full U')
                    state = getattr(p, '_jittor_fsdp2_state', None)
                    if state is not None and not selected(state):
                        raise NotImplementedError('mixed legacy/resident FSDP backward task')

    def publish(self, grad_by_id, initial_gradient, post_hooks):
        if self.published:
            raise RuntimeError('resident task publication repeated')
        reached_states = {}
        for root in self.reached.values():
            for key, ref in root.members.items():
                state = ref()
                if state is None:
                    raise RuntimeError('reached forward owner was destroyed')
                reached_states[key] = state
        for key, (state, entry) in self.target_entries.items():
            if key in grad_by_id:
                if id(state) not in reached_states:
                    raise RuntimeError('full gradient lacks actual forward/output ownership')
                u = entry.resident_full_param
                torch, _, _, _ = owners(state)
                with torch.no_grad():
                    old = u.grad
                    u.grad = initial_gradient(u, grad_by_id[key]) if old is None else old.add_(grad_by_id[key])
                post_hooks([u], {id(u): grad_by_id[key]})
        # All U post-accumulate mutations are now real. Do not use stale raw
        # grad_by_id if a public hook replaced or cleared U.grad.
        # Preserve real GradHooker traversal order; never sort names/records.
        # Repeated invocations share one aggregated U gradient, so each state
        # reduces once at its first actual arrival. All gradients are complete
        # here: the original single grad_optional has already returned.
        publication_states = {}
        for edge in self.input_arrivals:
            state = edge.state()
            if state is None or reached_states.get(id(state)) is not state:
                raise RuntimeError('resident input arrival owner disappeared')
            publication_states.setdefault(id(state), state)
        # Native root finalization also covers groups with no differentiable
        # input edge (or an edge not traversed by this actual gradient task).
        for key, state in reached_states.items():
            publication_states.setdefault(key, state)
        for state in publication_states.values():
            torch, _, grad_sync, common = owners(state)
            # Native FSDPParam accumulates a previous upcast gradient before
            # deciding whether this micro should reduce. Same-dtype no_sync
            # keeps the actual U.grad visible; it is not hidden in legacy
            # state.true_fsdp_pending_full_grads or copied through the reducer.
            with torch.no_grad():
                for entry in state.true_fsdp_params:
                    u = getattr(entry, 'resident_full_param', None)
                    accumulated = getattr(entry, 'resident_accumulated_grad', None)
                    if accumulated is not None and u is not None and u.grad is not None:
                        accumulated.add_(u.grad)
                        u.grad = None
            if not getattr(state, 'requires_gradient_sync', True):
                reshard(state.true_fsdp_module)
                reduce_dtype = getattr(getattr(state, 'mp_policy', None), 'reduce_dtype', None)
                with torch.no_grad():
                    for entry in state.true_fsdp_params:
                        u = getattr(entry, 'resident_full_param', None)
                        if u is not None and u.grad is not None and reduce_dtype is not None \
                                and u.grad.dtype != reduce_dtype:
                            g = u.grad
                            u.grad = None
                            entry.resident_accumulated_grad = g.to(dtype=reduce_dtype)
                continue  # No reducer, C.grad publication or C hook on this micro.
            full_grads = []
            for entry in state.true_fsdp_params:
                u = getattr(entry, 'resident_full_param', None)
                accumulated = getattr(entry, 'resident_accumulated_grad', None)
                # A parameter unused in the final micro can still own a real
                # gradient from an earlier no_sync micro. Do not drop it.
                g = accumulated if accumulated is not None else (u.grad if u is not None else None)
                full_grads.append(g)
                entry.resident_accumulated_grad = None
                if u is not None:
                    u.grad = None
                entry.full_public_grad = None
            reshard(state.true_fsdp_module)
            if not any(g is not None for g in full_grads):
                continue
            with common._frontend_scope(state), torch.no_grad():
                reduced = grad_sync._sync_sharded_grads_from_full_grads(state, full_grads)
                if len(reduced) != len(state.true_fsdp_params):
                    raise RuntimeError('original reducer changed inventory')
                for entry, g in zip(state.true_fsdp_params, reduced):
                    if g is None:
                        continue
                    c = entry.shard
                    previous = c.grad
                    # Same actual accumulated storage owner whether zero, one
                    # or multiple optimizers exist. Public setter mirrors slots.
                    c.grad = g if previous is None else previous.add_(g)
                    entry.last_grad = c.grad
                    post_hooks([c], {id(c): g})
        self.published = True

    def finish(self):
        if not self.published or self.completed:
            raise RuntimeError('resident completion without unique publication')
        for root in self.reached.values():
            for ref in root.members.values():
                state = ref()
                if state is None:
                    raise RuntimeError('resident completion owner disappeared')
                reshard(state.true_fsdp_module)
            root.consumed = True
            root.remove()
        self.completed = True

    def abort(self):
        # No rollback or successful completion is fabricated. Poison only
        # actual participating root members; retry needs an explicit new owner.
        for root in self.reached.values():
            root.failed = True
            root.remove()
            for ref in root.members.values():
                state = ref()
                if state is not None:
                    state._resident_poisoned = True


@contextmanager
def task_scope(inputs, create_graph, retain_graph, unknown_keywords):
    if current() is not None:
        raise NotImplementedError('resident reentrant backward')
    states = tuple(state for state in _states.values() if selected(state))
    if not states:
        yield None
        return
    if (inputs is not None or create_graph is not False
            or (retain_graph is not None and retain_graph is not False) or unknown_keywords):
        # Unrelated graphs keep their original mode; an actual resident output
        # hook rejects unsupported modes before any resident publication.
        yield None
        return
    task = Task(states)
    _local.task = task
    try:
        yield task
        if not task.completed:
            raise RuntimeError('resident backward returned without successful final callback')
    except BaseException as error:
        try:
            task.abort()
        except BaseException as cleanup_error:
            raise error from cleanup_error
        raise
    finally:
        _local.task = None
