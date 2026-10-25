"""Pinned DeepSpeed ZeRO-1/2 optimizer fast paths for Jittor."""
from functools import wraps
from weakref import WeakKeyDictionary


def make_zero_optimizer_methods(module, *, native, cuda_allowed):
    """Return guarded replacements for the pinned ZeRO-1/2 optimizer."""
    torch = module.torch
    optimizer_type = module.DeepSpeedZeroOptimizer
    original_norm = optimizer_type.get_grad_norm_direct
    original_overflow = optimizer_type.has_overflow_partitioned_grads_serial
    original_has_overflow = optimizer_type.has_overflow
    original_update = optimizer_type._update_model_bit16_weights
    original_allreduce_bucket = optimizer_type.allreduce_bucket
    original_allreduce_and_copy = optimizer_type.allreduce_and_copy
    original_partition_epilogue = (
        optimizer_type.independent_gradient_partition_epilogue)
    overflow_descriptor = vars(optimizer_type).get("_has_inf_or_nan")
    expected_overflow_check = (
        overflow_descriptor.__func__
        if isinstance(overflow_descriptor, staticmethod) else None)
    view_owners = WeakKeyDictionary()
    norm_results = WeakKeyDictionary()

    def is_native_cuda(value):
        # A placed native Var exposes its backend without allocating a public
        # torch.device object. Unplaced values fail this fast-path guard and
        # retain DeepSpeed's original implementation.
        return (isinstance(value, native.Var)
                and int(value.placement_backend) == 1)

    def same_native_placement(left, right):
        return (int(left.placement_backend) == int(right.placement_backend)
                and int(left.device_id) == int(right.device_id))

    @wraps(original_norm)
    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        eligible = (type(self) is optimizer_type
                    and isinstance(norm_type, (int, float)) and norm_type == 2
                    and self.model_parallel_rank == 0
                    and isinstance(gradients, (list, tuple))
                    and isinstance(params, (list, tuple)))
        cached = None
        if (eligible and type(self).__hash__ is not None
                and hasattr(self, "__weakref__")):
            cached = norm_results.get(self)
        if cached is not None:
            dense = [gradient for gradient in gradients
                     if not getattr(gradient, "_zero_padding", False)]
            signature = (
                tuple(id(gradient) for gradient in dense),
                tuple(id(parameter) for parameter in params),
            )
            if signature in cached:
                total = cached.pop(signature)
                if not cached:
                    norm_results.pop(self, None)
                module.mask_nan_or_inf_with_val_inplace(
                    total, device=self.device)
                return total
            norm_results.pop(self, None)
        if eligible:
            dense = [gradient for gradient in gradients
                     if not getattr(gradient, "_zero_padding", False)]
            eligible = (len(dense) == len(params) and len(dense) > 0
                        and all(
                            not (hasattr(parameter, module.PIPE_REPLICATED)
                                 and parameter.ds_pipe_replicated)
                            and not module.is_model_parallel_parameter(parameter)
                            for parameter in params))
        if eligible:
            device = dense[0].device
            eligible = (device.type == "cpu"
                        or (device.type == "cuda" and bool(cuda_allowed)))
            eligible = eligible and all(
                torch.is_tensor(gradient) and gradient.device == device
                and gradient.dtype == dense[0].dtype
                and gradient.dtype in (
                    torch.float16, torch.bfloat16, torch.float32)
                and not gradient.is_sparse and gradient.is_contiguous()
                for gradient in dense)
        if not eligible:
            return original_norm(self, gradients, params, norm_type=norm_type)
        flat = torch.cat([
            gradient.data.to(module.get_norm_dtype()).detach().reshape(-1)
            for gradient in dense
        ])
        total = torch.linalg.vector_norm(flat, ord=2).pow(2).float()
        group_reduce = getattr(self.dp_process_group, "all_reduce", None)
        direct_reduce = (
            bool(cuda_allowed)
            and getattr(self, "model_parallel_world_size", None) == 1
            and callable(group_reduce)
            and all(is_native_cuda(gradient) for gradient in dense)
        )
        if direct_reduce:
            # The native process group returns the reduced value. DeepSpeed's
            # public in-place facade would copy that value back into total
            # after a global graph flush, even though this scalar has no aliases.
            total = group_reduce(total, "sum")
            if not is_native_cuda(total):
                raise RuntimeError(
                    "DeepSpeed norm all-reduce contract changed")
        else:
            module.dist.all_reduce(
                total, op=module.dist.ReduceOp.SUM,
                group=self.dp_process_group)
            self._model_parallel_all_reduce(
                tensor=total, op=module.dist.ReduceOp.SUM)
        total = total.pow(0.5)
        module.mask_nan_or_inf_with_val_inplace(total, device=self.device)
        return total

    @wraps(original_overflow)
    def has_overflow_partitioned_grads_serial(self):
        groups = getattr(self, "averaged_gradients", None)
        bit16_groups = getattr(self, "bit16_groups", None)
        eligible = (type(self) is optimizer_type
                    and expected_overflow_check is not None
                    and self._has_inf_or_nan is expected_overflow_check
                    and isinstance(bit16_groups, (list, tuple))
                    and isinstance(groups, (dict, list, tuple)))
        if eligible:
            count = len(bit16_groups)
            eligible = all(
                (index in groups if isinstance(groups, dict)
                 else index < len(groups))
                and isinstance(groups[index], (list, tuple))
                for index in range(count))
        if eligible:
            device_name = module.get_accelerator().current_device_name()
            device_type = torch.device(device_name).type
            eligible = (device_type == "cpu"
                        or (device_type == "cuda" and bool(cuda_allowed)))
        if eligible:
            eligible = all(
                gradient is None
                or (torch.is_tensor(gradient)
                    and gradient.device.type == device_type
                    and gradient.dtype in (
                        torch.float16, torch.bfloat16, torch.float32)
                    and not gradient.is_sparse and gradient.is_contiguous())
                for index in range(count) for gradient in groups[index])
        if not eligible:
            return original_overflow(self)
        invalid = torch.zeros(
            [1], dtype=torch.float, device=device_name)
        for index in range(count):
            for gradient in groups[index]:
                if gradient is not None:
                    invalid = invalid + self._has_inf_or_nan(gradient)
        return invalid.bool()

    @wraps(original_has_overflow)
    def has_overflow(self, partition_gradients=True):
        can_cache = (type(self).__hash__ is not None
                     and hasattr(self, "__weakref__"))
        if can_cache:
            norm_results.pop(self, None)
        group_reduce = getattr(self.dp_process_group, "all_reduce", None)
        groups = getattr(self, "averaged_gradients", None)
        params = getattr(self, "params_in_partition", None)
        eligible = (
            type(self) is optimizer_type and bool(cuda_allowed) and can_cache
            and partition_gradients is True and not self.cpu_offload
            and getattr(self, "model_parallel_world_size", None) == 1
            and callable(group_reduce) and isinstance(groups, dict)
            and isinstance(params, (list, tuple)) and len(params) > 0
            and len(groups) == len(params))
        grouped = []
        if eligible:
            for index, parameters in enumerate(params):
                gradients = groups.get(index)
                eligible = (
                    isinstance(parameters, (list, tuple))
                    and isinstance(gradients, (list, tuple))
                    and len(gradients) == len(parameters)
                    and len(gradients) > 0)
                if not eligible:
                    break
                dense = [gradient for gradient in gradients
                         if gradient is not None
                         and not getattr(gradient, "_zero_padding", False)]
                eligible = (
                    len(dense) == len(gradients)
                    and all(
                        not (hasattr(parameter, module.PIPE_REPLICATED)
                             and parameter.ds_pipe_replicated)
                        and not module.is_model_parallel_parameter(parameter)
                        for parameter in parameters)
                    and all(
                        is_native_cuda(gradient)
                        and gradient.dtype in (
                            torch.float16, torch.bfloat16, torch.float32)
                        and not gradient.is_sparse
                        and gradient.is_contiguous()
                        for gradient in dense))
                if not eligible:
                    break
                grouped.append((dense, parameters))
        if eligible:
            invalid = None
            squared_norms = []
            signatures = []
            for gradients, parameters in grouped:
                flat = torch.cat([
                    gradient.data.to(
                        module.get_norm_dtype()).detach().reshape(-1)
                    for gradient in gradients
                ])
                local_invalid = torch.isfinite(
                    flat).all().logical_not().float()
                invalid = (
                    local_invalid if invalid is None
                    else invalid + local_invalid)
                squared_norms.append(
                    torch.linalg.vector_norm(
                        flat, ord=2).pow(2).float())
                signatures.append((
                    tuple(id(gradient) for gradient in gradients),
                    tuple(id(parameter) for parameter in parameters),
                ))
            reduced = group_reduce(
                torch.stack([invalid] + squared_norms), "sum")
            if not is_native_cuda(reduced):
                raise RuntimeError(
                    "DeepSpeed scalar all-reduce contract changed")
            overflow = bool(reduced[0].item())
            if not overflow:
                norm_results[self] = {
                    signature: reduced[index + 1].pow(0.5)
                    for index, signature in enumerate(signatures)
                }
            return overflow

        direct_reduce = (
            type(self) is optimizer_type and bool(cuda_allowed)
            and not self.cpu_offload
            and getattr(self, "model_parallel_world_size", None) == 1
            and callable(group_reduce))
        if not direct_reduce:
            return original_has_overflow(
                self, partition_gradients=partition_gradients)
        overflow = self.has_overflow_partitioned_grads_serial()
        overflow = overflow.byte().to(
            module.get_accelerator().current_device_name())
        if not is_native_cuda(overflow):
            return original_has_overflow(
                self, partition_gradients=partition_gradients)
        overflow = group_reduce(overflow, "max")
        if not is_native_cuda(overflow):
            raise RuntimeError(
                "DeepSpeed overflow all-reduce contract changed")
        return bool(overflow[0].item())

    @wraps(original_update)
    def _update_model_bit16_weights(self, group_index):
        eligible = (type(self) is optimizer_type and not self.cpu_offload
                    and not self.zenflow and not self.has_moe_layers
                    and type(self).__hash__ is not None
                    and hasattr(self, "__weakref__")
                    and isinstance(group_index, int)
                    and 0 <= group_index < len(self.bit16_groups_flat))
        if not eligible:
            return original_update(self, group_index)
        metadata = self.round_robin_bit16_meta[group_index]
        indices = self.round_robin_bit16_indices[group_index]
        index_signature = (
            tuple(indices.items()) if isinstance(indices, dict)
            else tuple(enumerate(indices)))
        signature = (
            id(self.bit16_groups_flat[group_index]),
            tuple(id(value)
                  for value in self.round_robin_bit16_groups[group_index]),
            tuple(id(value) for value in self.bit16_groups[group_index]),
            tuple(self.round_robin_bit16_offsets[group_index]),
            index_signature,
            tuple((tuple(meta.shape), int(meta.numel())) for meta in metadata),
        )
        owned = view_owners.get(self)
        if owned is None or owned.get(group_index) != signature:
            result = original_update(self, group_index)
            if owned is None:
                owned = {}
                view_owners[self] = owned
            owned[group_index] = signature
            return result
        return None

    @wraps(original_partition_epilogue)
    def independent_gradient_partition_epilogue(self):
        eligible = (
            type(self) is optimizer_type and bool(cuda_allowed)
            and self.partition_gradients and not self.cpu_offload
            and not self.overlap_comm and not self.zenflow
            and not self.has_moe_layers
            and isinstance(self.all_grad_tensors, dict)
            and isinstance(self.bit16_groups, (list, tuple))
            and isinstance(self.params_already_reduced, (list, tuple))
            and all(
                is_native_cuda(parameter)
                and parameter.dtype in (
                    torch.float16, torch.bfloat16, torch.float32)
                for group in self.bit16_groups for parameter in group))
        if not eligible:
            return original_partition_epilogue(self)

        self.report_ipg_memory_usage(
            "In ipg_epilogue before reduce_ipg_grads", 0)
        self.reduce_ipg_grads()
        self.report_ipg_memory_usage(
            "In ipg_epilogue after reduce_ipg_grads", 0)
        for index in range(len(self.params_already_reduced)):
            self.params_already_reduced[index] = False

        for index, _ in enumerate(self.bit16_groups):
            accumulated = self.all_grad_tensors.get(index)
            if accumulated is None:
                self.all_grad_tensors[index] = self.get_all_grad_tensors(
                    self.params_in_partition[index],
                    dtype=self.gradient_accumulation_dtype)
            else:
                incoming = self.get_all_grad_tensors(
                    self.params_in_partition[index],
                    dtype=self.gradient_accumulation_dtype)
                can_rebind = (
                    isinstance(accumulated, (list, tuple))
                    and isinstance(incoming, (list, tuple))
                    and len(accumulated) == len(incoming)
                    and all(
                        is_native_cuda(current)
                        and is_native_cuda(new)
                        and same_native_placement(new, current)
                        and new.dtype == current.dtype
                        and tuple(new.shape) == tuple(current.shape)
                        and current._storage_is_contiguous()
                        and new._storage_is_contiguous()
                        for current, new in zip(accumulated, incoming)))
                if can_rebind:
                    # The list owns these temporary accumulation values.
                    # Rebinding avoids one storage-preserving global flush
                    # per local gradient while keeping the same sum graph.
                    self.all_grad_tensors[index] = [
                        current + new
                        for current, new in zip(accumulated, incoming)]
                else:
                    for current, new in zip(accumulated, incoming):
                        current.add_(new)
            if self.is_gradient_accumulation_boundary():
                self.averaged_gradients[index] = self.get_flat_partition(
                    self.params_in_partition[index],
                    self.first_offset[index],
                    self.partition_size[index],
                    dtype=self.gradient_accumulation_dtype,
                    device=module.get_accelerator().current_device_name(),
                    param_group_idx=index, return_tensor_list=True)
                self.all_grad_tensors[index] = None

        self._release_ipg_buffers()
        self._clear_param_grad_only()
        self._epilogue_ran_this_backward = True
        module.see_memory_usage("End ipg_epilogue")

    @wraps(original_allreduce_bucket)
    def allreduce_bucket(self, bucket, communication_data_type,
                         rank=None, log=None, divide=True,
                         process_group=None):
        process_group = (
            self.dp_process_group if process_group is None
            else process_group)
        group_reduce = getattr(process_group, "all_reduce", None)
        eligible = (
            type(self) is optimizer_type and bool(cuda_allowed)
            and not self.overlap_comm and rank is None
            and not module.pg_correctness_test
            and self.sequence_parallel_size == 1
            and callable(group_reduce)
            and module.dist.get_world_size(group=process_group) > 1
            and communication_data_type in (
                torch.float16, torch.bfloat16, torch.float32)
            and isinstance(bucket, (list, tuple)) and len(bucket) > 0
            and all(
                is_native_cuda(value)
                and value.dtype in (
                    torch.float16, torch.bfloat16, torch.float32)
                and value._storage_is_contiguous()
                for value in bucket))
        if not eligible:
            return original_allreduce_bucket(
                self, bucket, communication_data_type,
                rank=rank, log=log, divide=divide,
                process_group=process_group)
        tensor = self.flatten(bucket)
        if (not is_native_cuda(tensor)
                or not tensor._storage_is_contiguous()):
            return original_allreduce_bucket(
                self, bucket, communication_data_type,
                rank=rank, log=log, divide=divide,
                process_group=process_group)
        reduced = tensor
        if communication_data_type != tensor.dtype:
            reduced = tensor.to(communication_data_type)
        if divide and self.gradient_average:
            reduced = reduced / (
                module.dist.get_world_size(group=process_group)
                / float(self.sequence_parallel_size))
        reduced = group_reduce(reduced, "sum")
        if (not is_native_cuda(reduced)
                or reduced.dtype != communication_data_type
                or not reduced._storage_is_contiguous()):
            raise RuntimeError(
                "DeepSpeed process-group all-reduce contract changed")
        reduced.sync()
        if communication_data_type != tensor.dtype:
            reduced = reduced.to(tensor.dtype)
        return reduced

    @wraps(original_allreduce_and_copy)
    def allreduce_and_copy(self, small_bucket, communication_data_type,
                           rank=None, log=None, divide=True,
                           process_group=None):
        eligible = (
            type(self) is optimizer_type and bool(cuda_allowed)
            and not self.overlap_comm
            and isinstance(small_bucket, (list, tuple))
            and len(small_bucket) > 0
            and all(
                is_native_cuda(value)
                and value.dtype in (
                    torch.float16, torch.bfloat16, torch.float32)
                and value._storage_is_contiguous()
                and hasattr(value, "_copy_into")
                for value in small_bucket))
        if not eligible:
            return original_allreduce_and_copy(
                self, small_bucket, communication_data_type,
                rank=rank, log=log, divide=divide,
                process_group=process_group)
        process_group = (
            self.dp_process_group if process_group is None
            else process_group)
        stream = module.get_accelerator().current_stream()
        with module.get_accelerator().stream(stream):
            allreduced = self.allreduce_bucket(
                small_bucket, communication_data_type,
                rank=rank, log=log, divide=divide,
                process_group=process_group)
            if rank is None or rank == module.dist.get_rank(
                    group=self.dp_process_group):
                synced_values = list(self.unflatten(
                    allreduced, small_bucket))
                if len(synced_values) != len(small_bucket):
                    raise RuntimeError(
                        "DeepSpeed allreduce unflatten length changed")
                # Drain every pending reader once for the complete bucket.
                # Each private storage copy below then preserves the same
                # destination allocation without another global graph flush.
                native.sync_all()
                with native.no_grad():
                    for destination, source in zip(
                            small_bucket, synced_values):
                        if (not isinstance(source, native.Var)
                                or tuple(destination.shape) != tuple(source.shape)
                                or destination.dtype != source.dtype
                                or not source._storage_is_contiguous()
                                or not destination._mem_ptr_now
                                or not source._mem_ptr_now):
                            raise RuntimeError(
                                "DeepSpeed batched storage copy contract changed")
                        destination._copy_into(source, False)

    return {
        "independent_gradient_partition_epilogue": (
            original_partition_epilogue,
            independent_gradient_partition_epilogue),
        "allreduce_bucket": (
            original_allreduce_bucket, allreduce_bucket),
        "allreduce_and_copy": (
            original_allreduce_and_copy, allreduce_and_copy),
        "get_grad_norm_direct": (original_norm, get_grad_norm_direct),
        "has_overflow_partitioned_grads_serial": (
            original_overflow, has_overflow_partitioned_grads_serial),
        "has_overflow": (
            original_has_overflow, has_overflow),
        "_update_model_bit16_weights": (
            original_update, _update_model_bit16_weights),
    }
