"""Unit contracts for pinned DeepSpeed optimizer helpers."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest


def _module(torch):
    class ReduceOp:
        SUM = object()

    dist = SimpleNamespace(
        ReduceOp=ReduceOp,
        all_reduce=lambda value, op=None, group=None: value,
    )

    class Optimizer:
        calls = 0
        allreduce_calls = 0
        overflow_calls = 0

        @staticmethod
        def _has_inf_or_nan(value):
            return torch.isfinite(value.detach()).all().logical_not()

        epilogue_calls = 0

        def independent_gradient_partition_epilogue(self):
            type(self).epilogue_calls += 1
            return "original-partition-epilogue"

        def allreduce_bucket(self, bucket, communication_data_type,
                             rank=None, log=None, divide=True,
                             process_group=None):
            type(self).allreduce_calls += 1
            return "original-allreduce-bucket"

        def allreduce_and_copy(self, small_bucket, communication_data_type,
                               rank=None, log=None, divide=True,
                               process_group=None):
            type(self).allreduce_calls += 1
            return "original-allreduce-and-copy"

        def get_grad_norm_direct(self, gradients, params, norm_type=2):
            total = sum(float((gradient.detach().float() ** 2).sum().item())
                        for gradient in gradients)
            return torch.tensor(total ** 0.5, dtype=torch.float32)

        def has_overflow_partitioned_grads_serial(self):
            invalid = torch.zeros([1], dtype=torch.float)
            for group in self.averaged_gradients:
                for gradient in self.averaged_gradients[group]:
                    if gradient is not None:
                        invalid = invalid + self._has_inf_or_nan(gradient)
            return invalid.bool()

        def has_overflow(self, partition_gradients=True):
            type(self).overflow_calls += 1
            return bool(
                self.has_overflow_partitioned_grads_serial()[0].item())

        def _update_model_bit16_weights(self, group_index):
            type(self).calls += 1
            flat = self.bit16_groups_flat[group_index]
            for parameter, meta, offset in zip(
                    self.round_robin_bit16_groups[group_index],
                    self.round_robin_bit16_meta[group_index],
                    self.round_robin_bit16_offsets[group_index]):
                parameter.data = flat.narrow(
                    0, offset, meta.numel()).view(meta.shape).data
            for original_index, parameter in enumerate(
                    self.bit16_groups[group_index]):
                new_index = self.round_robin_bit16_indices[
                    group_index][original_index]
                parameter.data = self.round_robin_bit16_groups[
                    group_index][new_index].data

        def _model_parallel_all_reduce(self, tensor=None, op=None):
            return tensor

    return SimpleNamespace(
        torch=torch, DeepSpeedZeroOptimizer=Optimizer,
        PIPE_REPLICATED="ds_pipe_replicated",
        is_model_parallel_parameter=lambda parameter: False,
        get_norm_dtype=lambda: torch.float32,
        dist=dist,
        get_accelerator=lambda: SimpleNamespace(
            current_device_name=lambda: "cpu"),
        mask_nan_or_inf_with_val_inplace=lambda value, device=None: value,
    )


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16", "float32"])
@pytest.mark.parametrize("values, expected", [
    ([1.0, -2.0, 3.0], False),
    ([1.0, float("inf")], True),
    ([1.0, float("nan")], True),
])
def test_finite_check_matches_expected(dtype_name, values, expected):
    import torch
    from jittor_adapters.deepspeed._finite import make_has_inf_or_nan

    tensor = torch.tensor(values, dtype=getattr(torch, dtype_name))
    wrapper = make_has_inf_or_nan(
        SimpleNamespace(torch=torch),
        lambda value: (_ for _ in ()).throw(
            AssertionError("eligible finite check fell back")),
        cuda_allowed=True)
    assert bool(wrapper(tensor).item()) is expected


def test_zero_norm_and_overflow_match_original_cpu():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    module = _module(torch)
    methods = make_zero_optimizer_methods(module, native=torch, cuda_allowed=False)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.model_parallel_rank = 0
    optimizer.dp_process_group = object()
    optimizer.device = "cpu"
    gradients = [
        torch.tensor([3.0, 4.0], dtype=torch.float32),
        torch.tensor([12.0], dtype=torch.float32),
    ]
    params = [SimpleNamespace(), SimpleNamespace()]
    original, replacement = methods["get_grad_norm_direct"]
    assert replacement(optimizer, gradients, params).item() == pytest.approx(
        original(optimizer, gradients, params).item())

    optimizer.bit16_groups = [[object()], [object()]]
    optimizer.averaged_gradients = {
        0: [torch.tensor([1.0, 2.0])],
        1: [torch.tensor([float("inf")]), None],
    }
    original, replacement = methods[
        "has_overflow_partitioned_grads_serial"]
    assert bool(replacement(optimizer).item()) is bool(original(optimizer).item())

    original, replacement = methods["has_overflow"]
    module.DeepSpeedZeroOptimizer.overflow_calls = 0
    assert replacement(optimizer) is original(optimizer)
    assert module.DeepSpeedZeroOptimizer.overflow_calls == 2


def test_overflow_uses_out_of_place_group_result_on_verified_cuda():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    class FakeItem:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeVar:
        placement_backend = 1
        device_id = 0

        def __init__(self, values):
            self.values = list(values)

        def byte(self):
            return self

        def to(self, device):
            return self

        def __getitem__(self, index):
            return FakeItem(self.values[index])

    class Group:
        def __init__(self):
            self.inputs = []

        def all_reduce(self, value, operation):
            self.inputs.append((list(value.values), operation))
            return FakeVar([1])

    class Native:
        Var = FakeVar

    module = _module(torch)
    methods = make_zero_optimizer_methods(
        module, native=Native, cuda_allowed=True)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.cpu_offload = False
    optimizer.model_parallel_world_size = 1
    optimizer.dp_process_group = Group()
    optimizer.has_overflow_partitioned_grads_serial = lambda: FakeVar([0])
    module.DeepSpeedZeroOptimizer.overflow_calls = 0
    _, replacement = methods["has_overflow"]
    assert replacement(optimizer) is True
    assert optimizer.dp_process_group.inputs == [([0], "max")]
    assert module.DeepSpeedZeroOptimizer.overflow_calls == 0



def test_overflow_and_norm_share_one_cuda_collective():
    import math
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    class FakeVar:
        placement_backend = 1
        device_id = 0
        is_sparse = False

        def __init__(self, values, dtype="float32"):
            self.values = list(values)
            self.dtype = dtype

        @property
        def data(self):
            return self

        def to(self, dtype):
            return FakeVar(self.values, dtype=dtype)

        def detach(self):
            return self

        def reshape(self, *_shape):
            return self

        def is_contiguous(self):
            return True

        def all(self):
            return FakeVar([all(self.values)], dtype="bool")

        def logical_not(self):
            return FakeVar([not bool(self.values[0])], dtype="bool")

        def float(self):
            return FakeVar([float(value) for value in self.values])

        def pow(self, exponent):
            return FakeVar(
                [float(value) ** exponent for value in self.values])

        def item(self):
            return self.values[0]

        def __add__(self, other):
            return FakeVar([
                left + right
                for left, right in zip(self.values, other.values)
            ])

        def __getitem__(self, index):
            return FakeVar([self.values[index]], dtype=self.dtype)

    class FakeTorch:
        float16 = "float16"
        bfloat16 = "bfloat16"
        float32 = "float32"

        class linalg:
            @staticmethod
            def vector_norm(value, ord=2):
                assert ord == 2
                return FakeVar([
                    sum(float(item) ** 2 for item in value.values) ** 0.5
                ])

        @staticmethod
        def cat(values):
            return FakeVar([
                item for value in values for item in value.values
            ])

        @staticmethod
        def stack(values):
            return FakeVar([value.item() for value in values])

        @staticmethod
        def isfinite(value):
            return FakeVar([
                math.isfinite(float(item)) for item in value.values
            ], dtype="bool")

    class Native:
        Var = FakeVar

    class Group:
        def __init__(self):
            self.calls = []

        def all_reduce(self, value, operation):
            self.calls.append((list(value.values), operation))
            return FakeVar([item * 2 for item in value.values])

    torch = FakeTorch()
    module = _module(torch)
    methods = make_zero_optimizer_methods(
        module, native=Native, cuda_allowed=True)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.cpu_offload = False
    optimizer.model_parallel_rank = 0
    optimizer.model_parallel_world_size = 1
    optimizer.dp_process_group = Group()
    optimizer.device = "cuda"
    gradients = [FakeVar([3.0, 4.0])]
    params = [SimpleNamespace()]
    optimizer.averaged_gradients = {0: gradients}
    optimizer.params_in_partition = [params]

    _, overflow = methods["has_overflow"]
    _, norm = methods["get_grad_norm_direct"]
    assert overflow(optimizer) is False
    assert norm(optimizer, gradients, params).item() == pytest.approx(50.0 ** 0.5)
    assert optimizer.dp_process_group.calls == [([0.0, 25.0], "sum")]

    optimizer.averaged_gradients = {
        0: [FakeVar([1.0, float("inf")])]
    }
    assert overflow(optimizer) is True
    assert optimizer.dp_process_group.calls[-1][0][0] == 1.0

def test_model_views_are_reused_and_rebound_when_layout_changes():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    module = _module(torch)
    methods = make_zero_optimizer_methods(module, native=torch, cuda_allowed=False)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.cpu_offload = False
    optimizer.zenflow = False
    optimizer.has_moe_layers = False
    flat = torch.arange(5, dtype=torch.float32)
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(3))
    optimizer.bit16_groups_flat = [flat]
    optimizer.round_robin_bit16_groups = [[first, second]]
    optimizer.bit16_groups = [[first, second]]
    optimizer.round_robin_bit16_offsets = [[0, 2]]
    optimizer.round_robin_bit16_indices = [{0: 0, 1: 1}]
    optimizer.round_robin_bit16_meta = [[
        SimpleNamespace(shape=(2,), numel=lambda: 2),
        SimpleNamespace(shape=(3,), numel=lambda: 3),
    ]]
    original, replacement = methods["_update_model_bit16_weights"]
    module.DeepSpeedZeroOptimizer.calls = 0
    replacement(optimizer, 0)
    pointers = (first.data_ptr(), second.data_ptr())
    replacement(optimizer, 0)
    assert module.DeepSpeedZeroOptimizer.calls == 1
    assert pointers == (first.data_ptr(), second.data_ptr())
    with torch.no_grad():
        flat.add_(10)
    assert first.tolist() == pytest.approx([10.0, 11.0])
    assert second.tolist() == pytest.approx([12.0, 13.0, 14.0])

    replacement_flat = torch.arange(5, dtype=torch.float32) + 20
    optimizer.bit16_groups_flat[0] = replacement_flat
    replacement(optimizer, 0)
    assert module.DeepSpeedZeroOptimizer.calls == 2
    assert first.data_ptr() == replacement_flat.data_ptr()
    assert first.tolist() == pytest.approx([20.0, 21.0])


def test_batch_copy_falls_back_without_verified_cuda():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    module = _module(torch)
    methods = make_zero_optimizer_methods(
        module, native=torch, cuda_allowed=False)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.overlap_comm = False
    module.DeepSpeedZeroOptimizer.allreduce_calls = 0
    _, replacement = methods["allreduce_and_copy"]
    result = replacement(
        optimizer, [torch.tensor([1.0])], torch.float32)
    assert result == "original-allreduce-and-copy"
    assert module.DeepSpeedZeroOptimizer.allreduce_calls == 1


def test_batch_copy_uses_one_flush_for_verified_cuda_bucket():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    class FakeVar:
        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)
            self.dtype = torch.float32
            self.device = SimpleNamespace(type="cuda")
            self.placement_backend = 1
            self.device_id = 0
            self._mem_ptr_now = True
            self.copies = []

        def _storage_is_contiguous(self):
            return True

        def _copy_into(self, source, sync_source=True):
            self.values = list(source.values)
            self.copies.append((source, sync_source))

    class Native:
        Var = FakeVar
        sync_calls = 0

        @classmethod
        def sync_all(cls):
            cls.sync_calls += 1

        @staticmethod
        def no_grad():
            return nullcontext()

    module = _module(torch)
    module.dist.get_rank = lambda group=None: 0
    module.get_accelerator = lambda: SimpleNamespace(
        current_stream=lambda: object(),
        stream=lambda stream: nullcontext())
    methods = make_zero_optimizer_methods(
        module, native=Native, cuda_allowed=True)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.overlap_comm = False
    optimizer.dp_process_group = object()
    destinations = [FakeVar([0.0, 0.0]), FakeVar([0.0])]
    sources = [FakeVar([3.0, 4.0]), FakeVar([5.0])]
    optimizer.allreduce_bucket = (
        lambda *args, **kwargs: FakeVar([3.0, 4.0, 5.0]))
    optimizer.unflatten = lambda allreduced, bucket: sources
    module.DeepSpeedZeroOptimizer.allreduce_calls = 0
    Native.sync_calls = 0
    _, replacement = methods["allreduce_and_copy"]
    assert replacement(optimizer, destinations, torch.float32) is None
    assert [value.values for value in destinations] == [[3.0, 4.0], [5.0]]
    assert all(value.copies[0][1] is False for value in destinations)
    assert Native.sync_calls == 1
    assert module.DeepSpeedZeroOptimizer.allreduce_calls == 0


def test_bucket_reduce_falls_back_without_verified_cuda():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    module = _module(torch)
    module.pg_correctness_test = False
    module.dist.get_world_size = lambda group=None: 2
    methods = make_zero_optimizer_methods(
        module, native=torch, cuda_allowed=False)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.dp_process_group = SimpleNamespace(
        all_reduce=lambda value, operation: value)
    optimizer.overlap_comm = False
    optimizer.sequence_parallel_size = 1
    optimizer.gradient_average = True
    module.DeepSpeedZeroOptimizer.allreduce_calls = 0
    _, replacement = methods["allreduce_bucket"]
    result = replacement(
        optimizer, [torch.tensor([1.0])], torch.float32)
    assert result == "original-allreduce-bucket"
    assert module.DeepSpeedZeroOptimizer.allreduce_calls == 1


def test_bucket_reduce_returns_out_of_place_group_result_on_verified_cuda():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    class FakeVar:
        def __init__(self, values, dtype=torch.float32):
            self.values = list(values)
            self.shape = (len(self.values),)
            self.dtype = dtype
            self.device = SimpleNamespace(type="cuda")
            self.placement_backend = 1
            self.device_id = 0
            self.sync_calls = 0

        def _storage_is_contiguous(self):
            return True

        def to(self, dtype):
            return FakeVar(self.values, dtype)

        def __truediv__(self, divisor):
            return FakeVar(
                [value / divisor for value in self.values], self.dtype)

        def sync(self):
            self.sync_calls += 1
            return self

    class Group:
        def __init__(self):
            self.inputs = []

        def all_reduce(self, value, operation):
            self.inputs.append((list(value.values), operation))
            return FakeVar([item * 2 for item in value.values], value.dtype)

    class Native:
        Var = FakeVar

    module = _module(torch)
    module.pg_correctness_test = False
    module.dist.get_world_size = lambda group=None: 2
    methods = make_zero_optimizer_methods(
        module, native=Native, cuda_allowed=True)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.dp_process_group = Group()
    optimizer.overlap_comm = False
    optimizer.sequence_parallel_size = 1
    optimizer.gradient_average = True
    optimizer.flatten = lambda bucket: FakeVar(
        [item for value in bucket for item in value.values])
    module.DeepSpeedZeroOptimizer.allreduce_calls = 0
    _, replacement = methods["allreduce_bucket"]
    result = replacement(
        optimizer, [FakeVar([2.0, 4.0]), FakeVar([6.0])],
        torch.float32)
    assert result.values == pytest.approx([2.0, 4.0, 6.0])
    assert result.sync_calls == 1
    assert optimizer.dp_process_group.inputs == [
        ([1.0, 2.0, 3.0], "sum")]
    assert module.DeepSpeedZeroOptimizer.allreduce_calls == 0


def test_partition_epilogue_rebinds_cuda_accumulation_values():
    import torch
    from jittor_adapters.deepspeed._zero import make_zero_optimizer_methods

    class FakeVar:
        inplace_calls = 0

        def __init__(self, values):
            self.values = list(values)
            self.shape = (len(self.values),)
            self.dtype = torch.float32
            self.device = SimpleNamespace(type="cuda")
            self.placement_backend = 1
            self.device_id = 0

        def _storage_is_contiguous(self):
            return True

        def __add__(self, other):
            return FakeVar([left + right for left, right in zip(
                self.values, other.values)])

        def add_(self, other):
            type(self).inplace_calls += 1
            self.values = [left + right for left, right in zip(
                self.values, other.values)]
            return self

    class Native:
        Var = FakeVar

    module = _module(torch)
    module.get_accelerator = lambda: SimpleNamespace(
        current_device_name=lambda: "cuda")
    module.see_memory_usage = lambda message: None
    methods = make_zero_optimizer_methods(
        module, native=Native, cuda_allowed=True)
    optimizer = module.DeepSpeedZeroOptimizer()
    optimizer.partition_gradients = True
    optimizer.cpu_offload = False
    optimizer.overlap_comm = False
    optimizer.zenflow = False
    optimizer.has_moe_layers = False
    optimizer.bit16_groups = [[FakeVar([0.0])]]
    optimizer.params_already_reduced = [True]
    optimizer.all_grad_tensors = {0: [FakeVar([1.0, 2.0])]}
    optimizer.params_in_partition = [[object()]]
    optimizer.gradient_accumulation_dtype = torch.float32
    optimizer.report_ipg_memory_usage = lambda message, index: None
    optimizer.reduce_ipg_grads = lambda: None
    optimizer.get_all_grad_tensors = lambda params, dtype: [
        FakeVar([3.0, 4.0])]
    optimizer.is_gradient_accumulation_boundary = lambda: False
    optimizer._release_ipg_buffers = lambda: None
    optimizer._clear_param_grad_only = lambda: None
    optimizer._epilogue_ran_this_backward = False
    original = optimizer.all_grad_tensors[0][0]
    FakeVar.inplace_calls = 0
    module.DeepSpeedZeroOptimizer.epilogue_calls = 0
    _, replacement = methods["independent_gradient_partition_epilogue"]
    assert replacement(optimizer) is None
    assert optimizer.all_grad_tensors[0][0] is not original
    assert optimizer.all_grad_tensors[0][0].values == [4.0, 6.0]
    assert optimizer.params_already_reduced == [False]
    assert optimizer._epilogue_ran_this_backward
    assert FakeVar.inplace_calls == 0
    assert module.DeepSpeedZeroOptimizer.epilogue_calls == 0
