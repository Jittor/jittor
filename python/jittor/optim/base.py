from jittor._core.dtypes import dtype_name as _jittor_dtype_name
from jittor._runtime.dispatch import dispatch_context
# ***************************************************************
# Copyright (c) 2023 Jittor. All Rights Reserved.
# Maintainers:
#     Guowei Yang <471184555@qq.com>
#     Guoye Yang <498731903@qq.com>
#     Wenyang Zhou <576825820@qq.com>
#     Meng-Hao Guo <guomenghao1997@gmail.com>
#     Dun Liang <randonlang@gmail.com>.
#
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
# ***************************************************************
import jittor as jt
import numpy as np
import operator
from copy import deepcopy
from functools import partial


_STANDARD_OPTIMIZER_ARITHMETIC = (
    operator.add, operator.mul, operator.truediv, operator.sub)


def _fp32_optimizer_binary(operation, left, right):
    if not isinstance(left, jt.Var):
        left = jt.array(left, dtype="float32").stop_grad()
    if not isinstance(right, jt.Var):
        right = jt.array(right, dtype="float32").stop_grad()
    return operation(left, right)


def _concrete_frontend_tensor(tensor):
    tensor_type = type(tensor)
    metadata = vars(tensor_type)
    return (metadata.get("_frontend_backend") is jt or (
        metadata.get("_parameter_backend") is jt
        and len(tensor_type.__bases__) == 1
        and vars(tensor_type.__bases__[0]).get("_frontend_backend") is jt))


def _optimizer_arithmetic(tensors, coefficients, *, enabled=True):
    """Use native FP32 arithmetic only for resolved frontend tensors."""
    if not enabled or not all(_concrete_frontend_tensor(t) for t in tensors):
        return _STANDARD_OPTIMIZER_ARITHMETIC
    if jt.flags.amp_reg != 0:
        return _STANDARD_OPTIMIZER_ARITHMETIC
    if not all(isinstance(c, (int, float)) for c in coefficients) or not all(
            _jittor_dtype_name(t.dtype) == "float32" for t in tensors):
        return _STANDARD_OPTIMIZER_ARITHMETIC
    param = tensors[0]
    placement = (param.placement_backend, param.device_id)
    if not all((t.placement_backend, t.device_id) == placement
               for t in tensors[1:]):
        return _STANDARD_OPTIMIZER_ARITHMETIC
    context = jt.core.dispatch_context([param])
    if context[0] not in ("cpu", "cuda") or context != jt.core.dispatch_context([]):
        return _STANDARD_OPTIMIZER_ARITHMETIC
    return tuple(partial(_fp32_optimizer_binary, operation) for operation in (
        jt.add, jt.multiply, jt.divide, jt.subtract))

def _grad_matches_param(p, g):
    return isinstance(g, jt.Var) and list(g.shape) == list(p.shape)

def _param_requires_grad(p):
    return bool(p.requires_grad)


def _native_ddp_readiness_plan(optimizer, targets, input_params):
    """Preflight the narrow NCCL path that can reduce ready backward buckets."""
    from jittor import distributed
    from jittor.nn.parallel.distributed_data_parallel import (
        _signature_check,
        _validate_reducer_state,
    )

    states = set(getattr(optimizer, "_jittor_ddp_states", ()))
    for parameter in targets:
        state = getattr(parameter, "_jittor_ddp_state", None)
        if state is not None:
            states.add(state)
    if len(states) != 1 or not targets:
        return None
    state = next(iter(states))
    if (state.world_size <= 1 or not state.sync_enabled
            or any(_param_requires_grad(parameter) for parameter in input_params)):
        return None
    backend_query = getattr(state.group, "_get_backend_name", None)
    backend = (backend_query() if callable(backend_query)
               else getattr(state.group, "_backend_kind", None))
    if backend != "nccl":
        return None

    parameter_index = {
        id(parameter): index
        for index, (_, parameter) in enumerate(state.parameters)
        if parameter.requires_grad
    }
    slot_by_id = {}
    duplicate = False
    for group in optimizer.param_groups:
        for slot_index, parameter in enumerate(group["params"]):
            if not _param_requires_grad(parameter):
                continue
            parameter_id = id(parameter)
            if parameter_id in slot_by_id:
                duplicate = True
            slot_by_id[parameter_id] = (group, slot_index)

    target_ids = [id(parameter) for parameter in targets]
    module_trainable_ids = set(parameter_index)
    eligible = (
        not duplicate
        and len(target_ids) == len(set(target_ids))
        and set(target_ids) == module_trainable_ids
        and all(id(parameter) in parameter_index for parameter in targets)
    )
    validation_error = None
    try:
        _validate_reducer_state(state)
    except RuntimeError as error:
        eligible = False
        validation_error = str(error)

    target_parameter_indices = tuple(sorted(
        parameter_index.get(id(parameter), -1) for parameter in targets
    ))
    payload = (
        "gradient_bucket_readiness",
        bool(eligible),
        validation_error,
        target_parameter_indices,
        tuple(tuple(bucket) for bucket in reversed(state.buckets)),
        bool(state.sync_enabled),
    )
    _signature_check(
        distributed, state.group, payload,
        "optimizer gradient bucket readiness plan", targets[0],
    )
    if not eligible:
        return None
    return {
        "state": state,
        "target_to_parameter": tuple(
            parameter_index[id(parameter)] for parameter in targets),
        "parameter_slots": {
            parameter_index[parameter_id]: slot
            for parameter_id, slot in slot_by_id.items()
            if parameter_id in parameter_index
        },
    }


def _synchronize_ddp_clip_coefficient(optimizer, coefficient):
    """Share one coefficient so identical synchronized gradients stay identical."""
    from jittor import distributed

    states = set(getattr(optimizer, "_jittor_ddp_states", ()))
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            state = getattr(parameter, "_jittor_ddp_state", None)
            if state is not None:
                states.add(state)
    if len(states) != 1:
        return coefficient
    state = next(iter(states))
    if state.world_size <= 1 or state.has_unsynced_grads:
        return coefficient
    backend_query = getattr(state.group, "_get_backend_name", None)
    backend = (backend_query() if callable(backend_query)
               else getattr(state.group, "_backend_kind", None))
    if backend != "nccl":
        return coefficient
    ranks = getattr(state.group, "ranks", None)
    src = int(ranks[0]) if ranks else 0
    shared = distributed.broadcast(
        coefficient, src=src, group=state.group)
    return coefficient if shared is None else shared


#: Group-dict keys that are not per-parameter state buffers.
_NON_STATE_KEYS = frozenset(("params", "grads"))


def _state_buffer(param):
    """A zeroed per-parameter state buffer, on the parameter's own device.

    ``jt.zeros(param.shape, param.dtype)`` allocates on the *ambient* device,
    so an optimizer built for parameters that are not on it put its momentum /
    moment buffers on the wrong card from the start. ``zeros_like`` follows the
    parameter, which is what every other ``*_like`` in jittor does.
    """
    return jt.zeros_like(param).stop_grad()


def _adaptive_state_buffer(param):
    """Keep adaptive optimizer moments in FP32 for half-precision params."""
    state = _state_buffer(param)
    if _jittor_dtype_name(param.dtype) in ("float16", "bfloat16"):
        with jt.flag_scope(amp_reg=0):
            state = state.float32().stop_grad()
    return state


def _realign_state_buffers(param_groups):
    """Move per-parameter optimizer state back onto its parameter's device.

    ``Module.to("cuda:1")`` is in place, so an optimizer built before the move
    keeps the very Parameter objects that moved -- but not its own buffers,
    which were allocated where the parameters used to be. The next step then
    handed a fused kernel a parameter on cuda:1 and a momentum buffer on
    cuda:0 and died in the middle of training with "Expected all tensor inputs
    on the same backend and device". torch gets away with it by creating its
    state lazily at the first ``step()``, i.e. after the move; jittor
    allocates at construction, so the alignment is checked here instead.

    Runs over the group dicts generically: any list of Vars parallel to
    ``params`` is state, whatever the algorithm calls it (``values``, ``m``,
    ``v``, ``d``, ``pre_grad``, ...).
    """
    for group in param_groups:
        params = group.get("params")
        if not params:
            continue
        for key, buffers in group.items():
            if key in _NON_STATE_KEYS or type(buffers) is not list:
                continue
            if len(buffers) != len(params):
                continue
            for param, buffer in zip(params, buffers):
                if not isinstance(buffer, jt.Var) or not isinstance(param, jt.Var):
                    continue
                # The same raw placement is the same device, whatever the
                # ambient flag says -- and it is what every step of an
                # unmoved model sees, so the resolution below (a dispatch
                # lookup for an unplaced Var) is kept for the pairs that differ.
                if (buffer.placement_backend == param.placement_backend
                        and buffer.device_id == param.device_id):
                    continue
                target = _effective_device(param)
                if _effective_device(buffer) == target:
                    continue
                # `update` keeps the buffer's object identity, which a caller
                # reading `optimizer.state[p][...]` and the algorithms' own
                # in-place kernels both rely on.
                buffer.update(buffer._copy_to_cpu() if target < 0
                              else buffer.to_device(target))


def _effective_device(var):
    """Where ``var`` actually is: ``-1`` for the host, else the device index.

    The raw fields cannot be compared directly. An explicitly placed Var says
    so in ``placement_backend`` (0 is the host, and its ``device_id`` is -1);
    an unplaced one follows the ambient ``use_cuda`` flag and its ``device_id``
    only means something while that flag is on. Comparing the raw pair instead
    read a host-resident `jt.array` (unplaced, ``device_id`` 0, under
    ``use_cuda=0``) as being on a different device from its own zero buffer
    (explicitly host-placed, ``device_id`` -1) and "moved" the buffer to
    cuda:0 -- inside a CPU-only scope.
    """
    backend = var.placement_backend
    if backend == 0:
        return -1
    if backend > 0:
        return int(var.device_id)
    # Unplaced: ask the dispatch table where this Var's ops would run, not the
    # `use_cuda` flag. The two answer the same question, but a backend-flag
    # read inside an operator domain is what
    # `tests/structure/runtime/test_backend_op_registry_contract` forbids:
    # device selection goes through the registered table, not around it.
    if dispatch_context(var).backend == "cpu":
        return -1
    index = int(var.device_id)
    return index if index >= 0 else int(jt.current_device())


def _update_preserve_dtype(target, value):
    if _jittor_dtype_name(value.dtype) != _jittor_dtype_name(target.dtype):
        value = value.to(target.dtype)
    target.update(value)

class Optimizer(object):
    """ Basic class of Optimizer.

    Example::

        optimizer = nn.SGD(model.parameters(), lr)
        optimizer.step(loss)
    """
    def __init__(self, params, lr, param_sync_iter=10000):
        self.param_groups = []
        self.lr = lr
        self.param_sync_iter = param_sync_iter

        if len(params) == 0:
            raise ValueError("Length of parameters should not be zero")
        if not isinstance(params[0], dict):
            params = [{'params': params}]
        for pg in params:
            if not isinstance(pg, dict):
                raise TypeError("optimizer parameter groups must be dictionaries")
            # A group is walked on every step and saved with the optimizer, so
            # it holds a list, whatever iterable it was given -- a module's
            # `parameters()` under the torch frontend produces as it is walked.
            if not isinstance(pg.get('params'), (list, tuple)) and 'params' in pg:
                pg['params'] = list(pg['params'])
            self.param_groups.append(pg)
        self.n_step = 0
        # __zero_grad is a value for fast determ the grad is zero or not
        # so we can omit 0+x
        self.__zero_grad = True
        self._grad_map = {}
        self.__input_params = []
        # Runtime DDP bindings stay with this optimizer, including when a user
        # edits a group to remove every managed parameter. Private fields are
        # excluded from defaults/state_dict in normal checkpoints.
        self._jittor_ddp_states = {}
        for pg in self.param_groups:
            for p in pg['params']:
                state = getattr(p, '_jittor_ddp_state', None)
                if state is not None:
                    self._jittor_ddp_states[state] = None

    def add_param_group(self, group):
        if 'params' in group and not isinstance(group['params'], (list, tuple)):
            group['params'] = list(group['params'])
        self.param_groups.append(group)
        for p in group.get('params', ()):
            state = getattr(p, '_jittor_ddp_state', None)
            if state is not None:
                self._jittor_ddp_states[state] = None

    def _advance_step_count(self, pg):
        """Number of optimizer steps this param group has taken, 1-based.

        Bias corrections must use this, not ``self.n_step``: ``n_step`` counts
        ``backward`` calls, and the gradient-accumulation loop documented on
        ``backward`` calls it once per micro-batch, so a correction keyed on it
        runs ahead by the accumulation factor. Kept inside the param group so
        it rides along in ``state_dict``/``load_state_dict`` and so a group
        added mid-training starts its own correction at step 1.
        """
        from jittor._runtime import step_capture
        # Baked into the update as a constant, so a replay would repeat it.
        step_capture.refuse("the optimizer bakes its step count into the graph")
        n = int(pg.get("n_step", 0)) + 1
        pg["n_step"] = n
        return n

    def set_input_into_param_group(self, inputs):
        """ This function adds inputs to the optimizer as variables that need tuning.
            This is to enforce the calculation of gradients from the output to the input,
            ensuring that the backward hook is called correctly.

        Args:
            inputs: List of the input
        """
        self.__input_params = []
        if isinstance(inputs, jt.Var):
            self.__input_params.append(inputs)
        elif isinstance(inputs, (list, tuple)):
            for v in inputs:
                if isinstance(v, jt.Var):
                    self.__input_params.append(v)
        else:
            raise NotImplementedError

    def clip_grad_norm(self, max_norm:float, norm_type:int=2):
        r"""Clips gradient norm of this optimizer.
        The norm is computed over all gradients together.

        Args:
            max_norm (float or int): max norm of the gradients
            norm_type (int): 1-norm or 2-norm

        Example::

            a = jt.ones(2)
            opt = jt.optim.SGD([a], 0.1)

            loss = a*a
            opt.zero_grad()
            opt.backward(loss)

            print(opt.param_groups[0]['grads'][0].norm()) # output: 2.83
            opt.clip_grad_norm(0.01, 2)
            print(opt.param_groups[0]['grads'][0].norm()) # output: 0.01

            opt.step()

        """
        if self.__zero_grad: return
        # Keep explicit FP32 norm inputs from being recast by an enclosing AMP scope.
        with jt.flag_scope(amp_reg=0):
            grads = []
            for pg in self.param_groups:
                for p, g in zip(pg["params"], pg["grads"]):
                    if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                    dtype = _jittor_dtype_name(g.dtype)
                    # Half-precision sum-of-squares can overflow or underflow.
                    norm_grad = (g.float32() if dtype in ("float16", "bfloat16")
                                 else g)
                    grads.append(norm_grad.flatten())
            if len(grads) == 0: return
            total_norm = jt.norm(jt.concat(grads), norm_type)
            clip_coef = jt.minimum(max_norm / (total_norm + 1e-6), 1.0)
            clip_coef = _synchronize_ddp_clip_coefficient(self, clip_coef)
            for pg in self.param_groups:
                for p, g in zip(pg["params"], pg["grads"]):
                    if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                    dtype = _jittor_dtype_name(g.dtype)
                    grad = (g.float32() if dtype in ("float16", "bfloat16")
                            else g)
                    g.update((grad * clip_coef).cast(dtype))

    @property
    def defaults(self):
        exclude = set(("defaults", "pre_step", "step"))
        return { k:v for k, v in self.__dict__.items()
            if k[0] != '_' and k not in exclude and not callable(v) }

    def state_dict(self):
        state = {"defaults": self.defaults}
        return state

    def load_state_dict(self, state):

        def dfs(x):
            if isinstance(x, list):
                return [dfs(value) for value in x]
            if isinstance(x, tuple):
                return tuple(dfs(value) for value in x)
            if isinstance(x, dict):
                return {key: dfs(value) for key, value in x.items()}
            if isinstance(x, np.ndarray):
                return jt.array(x).stop_grad()
            if isinstance(x, jt.Var):
                return x.clone().stop_grad()
            return x

        exclude = set(("param_groups", "params"))
        for k, v in state["defaults"].items():
            if k not in exclude:
                setattr(self, k, dfs(v))
        param_groups = state["defaults"].get('param_groups', None)
        if param_groups is not None:
            for i in range(len(param_groups)):
                for k, v in param_groups[i].items():
                    if k != "params":
                        self.param_groups[i][k] = dfs(v)



    def zero_grad(self):
        ''' Reset the accumulated gradients of every param group to zero.

        The buffers in ``pg["grads"]`` are cleared, not just marked: they are
        public (``opt_grad``, ``clip_grad_norm`` and user code all read them),
        and ``post_step`` calls this after every step, so leaving the consumed
        gradients in place made every later reader silently work on stale data.

        The write is skipped when the gradients are already known to be zero,
        and in the ordinary training loop the zeros are overwritten by the next
        ``backward`` before anything can observe them, so jittor's lazy graph
        drops them without ever running the fill.

        The zero itself is built once per gradient and reused. `zeros_like` is
        a broadcast of a scalar and carries no storage, so what a fresh one per
        step per gradient costs is not memory but host time: building the Var
        and rebinding the holder ran 96 times a step on an 8-layer transformer
        and measured 1.03 ms of an 8.49 ms training step. Reuse is safe because
        nothing writes a gradient buffer in place -- every path that changes a
        gradient (`update`, `assign`, `+=`) rebinds the holder to a new Var and
        leaves the one cached here alone.
        '''
        if not self.__zero_grad:
            # id(grad holder) -> a zeros Var of its shape. Per instance, not
            # per class: a class attribute would be shared by every optimizer
            # in the process and would never be released.
            cache = self.__dict__.setdefault("_zero_grad_cache", {})
            for pg in self.param_groups:
                for g in pg.get("grads", ()):
                    key = id(g)
                    zero = cache.get(key)
                    if zero is None or zero.shape != g.shape or zero.dtype != g.dtype:
                        zero = cache[key] = jt.zeros_like(g).stop_grad()
                    g.update(zero)
        for pg in self.param_groups:
            for p in pg["params"]:
                state = getattr(p, "_jittor_ddp_state", None)
                if state is not None:
                    self._jittor_ddp_states[state] = None
        for state in self._jittor_ddp_states:
            state.has_unsynced_grads = False
        self.__zero_grad = True

    def backward(self, loss, retain_graph=False):
        '''
        optimize.backward(loss) is used for accumulate multiple step,
        it can be used as following:

        Origin source code ::

        n_iter = 10000
        batch_size = 100
        ...
        for i in range(n_iter):
            ...
            loss = calc_loss()
            optimizer.step(loss)

        Accumulation version ::

        n_iter = 10000
        batch_size = 100
        accumulation_steps = 10
        n_iter *= accumulation_steps
        batch_size //= accumulation_steps
        ...
        for i in range(n_iter):
            ...
            loss = calc_loss()
            # if loss is a mean across batch, we need to divide accumulation_steps
            optimizer.backward(loss / accumulation_steps)
            if (i+1) % accumulation_steps == 0:
                optimizer.step()


        '''
        # clean prev grads
        params = []
        params_has_grad = []
        for pg in self.param_groups:
            for p in pg['params']:
                params.append(p)
                if _param_requires_grad(p):
                    params_has_grad.append(p)
        for p in self.__input_params:
            if _param_requires_grad(p):
                params_has_grad.append(p)

        # sync prev params
        jt.sync(params_has_grad)

        readiness_plan = _native_ddp_readiness_plan(
            self, params_has_grad, self.__input_params)
        if readiness_plan is not None:
            from jittor import distributed
            from jittor.distributed.bucket import bucket_scope, comm_wait
            from jittor.nn.parallel.distributed_data_parallel import (
                _GradientBucketReadiness,
            )
            from jittor.backends.cuda.kernels.misc.ddp_bucket import (
                pack_gradients, unpack_gradients,
            )

            for pg in self.param_groups:
                if "grads" not in pg:
                    pg["grads"] = [
                        jt.zeros_like(p).stop_grad().stop_fuse()
                        for p in pg["params"]
                    ]

            state = readiness_plan["state"]
            parameter_slots = readiness_plan["parameter_slots"]
            reduced_pairs = []
            # Publish optimizer holders only after every bucket has joined;
            # pending communication owns the exact accumulated inputs meanwhile.
            pending_values = {}
            comm_pending = False
            ready_gradients = []

            def record_gradient_ready(target_index, gradient):
                ready_gradients.append((target_index, gradient))

            def on_bucket_ready(parameter_indices, _gradients):
                nonlocal comm_pending
                members = [
                    parameter_slots[index][0]["grads"][
                        parameter_slots[index][1]
                    ]
                    for index in parameter_indices
                ]
                values = [pending_values[index] for index in parameter_indices]
                jt.sync(values, weak_sync=False)
                value = (pack_gradients(values)
                         if len(values) > 1 else values[0])
                with bucket_scope(defer_join=True):
                    reduced = distributed.all_reduce(
                        value, op="sum", group=state.group)
                    comm_pending = True
                    jt.sync([reduced], weak_sync=False)
                reduced_pairs.append((members, reduced, value))

            readiness = _GradientBucketReadiness(
                buckets=tuple(reversed(state.buckets)),
                target_to_parameter=readiness_plan["target_to_parameter"],
                on_bucket_ready=on_bucket_ready,
            )

            def on_gradient_ready(target_index, gradient):
                parameter_index = readiness_plan["target_to_parameter"][
                    target_index]
                group, slot = parameter_slots[parameter_index]
                value = gradient.stop_grad()
                if not self.__zero_grad:
                    value = value + group["grads"][slot]
                pending_values[parameter_index] = value
                readiness(target_index, gradient)

            try:
                # Finish graph construction before evaluating any bucket. The
                # autograd builder owns intermediate-gradient roots until it
                # returns; executing inside its callback keeps those buffers
                # alive across the entire backward pass. Replay the original
                # readiness order after those temporary roots are released.
                # Disable automatic submission through the return conversion
                # and replay: otherwise it can execute all pending gradients
                # before our explicit per-bucket sync, losing device overlap.
                with jt.flag_scope(auto_flush_ops=0):
                    jt.core.grad_with_callback(
                        loss, params_has_grad, retain_graph,
                        record_gradient_ready)
                    for target_index, gradient in ready_gradients:
                        on_gradient_ready(target_index, gradient)
                    readiness.finish()
            finally:
                ready_gradients.clear()
                if comm_pending:
                    comm_wait()

            for members, reduced, _value in reduced_pairs:
                if len(members) == 1:
                    members[0].update(reduced / state.world_size)
                    continue
                for gradient, averaged in zip(
                        members,
                        unpack_gradients(reduced, members, state.world_size)):
                    gradient.update(averaged)

            entries = []
            for group in self.param_groups:
                for parameter, gradient in zip(
                        group["params"], group["grads"]):
                    if _param_requires_grad(parameter):
                        entries.append((parameter, gradient))
            ddp_parameter_ids = {id(parameter) for parameter, _ in entries}
            dep = []
            state.has_unsynced_grads = False
        else:
            # Legacy path remains authoritative for no_sync, MPI, mixed
            # optimizer/DDP groups, input gradients and non-NCCL backends.
            grads = jt.grad(loss, params_has_grad, retain_graph)

            pid = 0
            entries = []
            for pg in self.param_groups:
                if "grads" not in pg:
                    pg["grads"] = [
                        jt.zeros_like(p).stop_grad().stop_fuse()
                        for p in pg["params"]
                    ]
                pg_grads = pg["grads"]
                for i, p in enumerate(pg["params"]):
                    if _param_requires_grad(p):
                        g = grads[pid].stop_grad()
                        pid += 1
                        if (getattr(p, "_jittor_ddp_state", None) is not None
                                and not _grad_matches_param(p, g)):
                            g = jt.zeros_like(p).stop_grad()
                        if not self.__zero_grad:
                            g = g + pg_grads[i]
                        pg_grads[i].update(g)
                        entries.append((p, pg_grads[i]))

            from jittor.nn.parallel.distributed_data_parallel import (
                _sync_optimizer_gradients,
            )
            ddp_parameter_ids, dep = _sync_optimizer_gradients(
                entries, optimizer=self)

        self.__zero_grad = False
        self.n_step += 1

        # jt.in_mpi means a distributed process is active for both MPI and
        # the MPI-free NCCL launcher. Only the actual MPI backend uses the
        # legacy Var.mpi_all_reduce path.
        from jittor import distributed as _distributed
        native_nccl = (
            _distributed.is_initialized()
            and _distributed.get_backend() == "nccl"
        )
        # The MPI-free launcher also sets ``jt.in_mpi`` for a one-rank job.
        # There is no peer to synchronize in that case, and its process group
        # may not be initialized (for example, a raw single-card benchmark).
        # Keep the legacy Var.mpi_* path limited to actual multi-rank jobs.
        if (jt.in_mpi and not native_nccl
                and _distributed.get_world_size() > 1):
            def add_dep(value):
                nonlocal dep
                value._add_dependency(dep)
                dep = [value]

            for parameter, gradient in entries:
                if id(parameter) in ddp_parameter_ids:
                    continue
                gradient.assign(gradient.mpi_all_reduce("mean"))
                add_dep(gradient._input(0))
            for gradient in grads[pid:]:
                gradient.assign(gradient.mpi_all_reduce("mean"))
                add_dep(gradient._input(0))
            if self.n_step % self.param_sync_iter == 0:
                for parameter in params:
                    if id(parameter) in ddp_parameter_ids:
                        continue
                    parameter.assign(parameter.mpi_broadcast())
                    add_dep(parameter)

    def pre_step(self, loss, retain_graph=False):
        """ something should be done before step, such as calc gradients, mpi sync, and so on.

        Example::

            class MyOptimizer(Optimizer):
                def step(self, loss):
                    self.pre_step(loss)
                    ...
                    self.post_step()
        """
        if loss is not None:
            self.backward(loss, retain_graph)
        _realign_state_buffers(self.param_groups)
        from jittor.nn.parallel.distributed_data_parallel import (
            _assert_ddp_step_ready,
        )
        _assert_ddp_step_ready(self)
        jt.flags.node_order = 1

    def post_step(self):
        """ something should be done before step, such as zero grad, and so on.

        Example::

            class MyOptimizer(Optimizer):
                def step(self, loss):
                    self.pre_step(loss)
                    ...
                    self.post_step()
        """
        jt.flags.node_order = 0
        self.zero_grad()


    def step(self, loss=None, retain_graph=False):
        self.pre_step(loss, retain_graph)
        for pg in self.param_groups:
            lr = pg.get("lr", self.lr)
            for p, g in zip(pg["params"], pg["grads"]):
                if not _param_requires_grad(p) or not _grad_matches_param(p, g): continue
                _update_preserve_dtype(p, p - g * lr)
        self.post_step()

    def _build_grad_map(self):
        _grad_map = {}
        for pg in self.param_groups:
            for p, g in zip(pg["params"], pg["grads"]):
                if not _grad_matches_param(p, g):
                    continue
                _grad_map[id(p)] = g
        self._grad_map = _grad_map

    def find_grad(self, v:jt.Var) -> jt.Var:
        if id(v) not in self._grad_map:
            self._build_grad_map()
            if id(v) not in self._grad_map:
                raise RuntimeError("This variable is not managed by this optimizer")
        return self._grad_map[id(v)]

def opt_grad(v:jt.Var, opt:Optimizer):
    ''' Get grad of certain variable in optimizer, Example::


    model = Model()
    optimizer = SGD(model.parameters())
    ...
    optimizer.backward(loss)

    for p in model.parameters():
        grad = p.opt_grad(optimizer)
    '''
    return opt.find_grad(v)


jt.Var.opt_grad = opt_grad
