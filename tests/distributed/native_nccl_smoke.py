"""Real-device smoke test for jtrun, native collectives, DDP and Dataset."""

import ctypes
import os

import jittor as jt
from jittor import distributed as dist
from jittor import nn
from jittor.dataset import Dataset
from jittor.nn.parallel import DistributedDataParallel


def _scalar(value):
    return float(value.numpy().reshape(-1)[0])


def _close(actual, expected, tolerance=1e-4):
    return abs(actual - expected) <= tolerance


def _nccl_version_code():
    get_version = ctypes.CDLL(None).ncclGetVersion
    get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
    get_version.restype = ctypes.c_int
    version = ctypes.c_int()
    if get_version(ctypes.byref(version)) != 0:
        raise RuntimeError("ncclGetVersion failed")
    return version.value


class _TraceDataset(Dataset):
    def __init__(self):
        super().__init__(batch_size=6, shuffle=False, drop_last=True,
                         num_workers=0)
        self.set_attrs(total_len=12)

    def __getitem__(self, index):
        return index


class _Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.register_buffer(
            "sync_marker", jt.array([0.0], dtype="float32").stop_grad())

    def execute(self, value):
        return self.linear(value)


def main():
    dist.init_process_group(backend="nccl")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        nccl_version_code = _nccl_version_code()
        checks = [
            world_size >= 2,
            dist.get_local_rank() == rank,
            dist.get_backend() == "nccl",
            len(os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")) == 1,
        ]

        reduced = dist.all_reduce(
            jt.array([rank + 1.0]).cuda(0).stop_grad(), op="sum")
        checks.append(_close(_scalar(reduced), world_size * (world_size + 1) / 2.0))

        bf16_tested = nccl_version_code >= 21000
        if bf16_tested:
            bf16_input = jt.array(
                [(rank + 1) * float(index) for index in range(1, 33)]
            ).float32().cuda(0).bfloat16().stop_grad()
            bf16_reduced = dist.all_reduce(bf16_input, op="sum")
            bf16_values = bf16_reduced.float32().numpy().reshape(-1).tolist()
            bf16_factor = world_size * (world_size + 1) / 2.0
            checks.append(
                all(abs(actual - bf16_factor * index) <= max(
                        0.1, abs(bf16_factor * index) * 0.02)
                    for index, actual in enumerate(bf16_values, 1)))
            if rank == 0:
                print("NATIVE_NCCL_BF16 version={} values={}".format(
                    nccl_version_code, bf16_values[:4]), flush=True)

        broadcast = dist.broadcast(
            jt.array([rank + 10.0]).cuda(0).stop_grad(), src=0)
        checks.append(_close(_scalar(broadcast), 10.0))

        gathered = dist.all_gather(
            jt.array([rank + 1.0]).cuda(0).stop_grad())
        checks.append(
            gathered.numpy().reshape(-1).tolist() ==
            [float(index) for index in range(1, world_size + 1)])

        dataset = _TraceDataset()
        local_indices = jt.array(dataset._get_index_list()).int32().cuda(0)
        global_indices = dist.all_gather(local_indices)
        checks.append(
            sorted(set(global_indices.numpy().reshape(-1).tolist())) == list(range(12)))

        sync_bn = nn.BatchNorm(1, momentum=1.0, sync=True)
        sync_bn.cuda(0)
        bn_input = jt.array([[rank * 10.0], [rank * 10.0 + 2.0]])
        bn_output = sync_bn(bn_input.float32().cuda(0).stop_grad())
        bn_running_means = dist.all_gather(
            sync_bn.running_mean.reshape((-1,)).stop_grad())
        expected_bn_mean = 5.0 * (world_size - 1) + 1.0
        checks.append(all(
            _close(value, expected_bn_mean)
            for value in bn_running_means.numpy().reshape(-1).tolist()))
        checks.append(bn_output.shape == (2, 1))

        jt.set_global_seed(811 + rank)
        raw_model = _Regressor().cuda(0)
        raw_model.sync_marker.update(
            jt.array([rank + 10.0], dtype="float32").cuda(0).stop_grad())
        raw_parameter = raw_model.linear.weight
        unsynchronized = dist.all_gather(
            raw_parameter.reshape((-1,)).stop_grad())
        initial_values = unsynchronized.numpy().reshape(-1).tolist()
        checks.append(initial_values[0] != initial_values[1])

        model = DistributedDataParallel(raw_model)
        parameter = model.module.linear.weight
        initial = _scalar(parameter)
        synced_initial = dist.all_gather(parameter.reshape((-1,)).stop_grad())
        checks.append(
            synced_initial.numpy().reshape(-1).tolist() ==
            [initial] * world_size)
        synced_buffers = dist.all_gather(
            model.module.sync_marker.reshape((-1,)).stop_grad())
        checks.append(
            synced_buffers.numpy().reshape(-1).tolist() == [10.0] * world_size)

        optimizer = jt.optim.SGD(model.parameters(), lr=0.05)
        input_value = jt.array([[rank + 1.0]]).float32().cuda(0).stop_grad()
        loss = (model(input_value) ** 2).mean()
        all_reduce = dist.all_reduce
        reduced_shapes = []

        def count_all_reduce(tensor, op="mean", group=None, async_op=False):
            reduced_shapes.append(tuple(int(dim) for dim in tensor.shape))
            return all_reduce(tensor, op=op, group=group, async_op=async_op)

        dist.all_reduce = count_all_reduce
        try:
            optimizer.backward(loss)
        finally:
            dist.all_reduce = all_reduce
        gradient = optimizer.param_groups[0]["grads"][0]
        mean_square_rank = sum(
            float(index * index) for index in range(1, world_size + 1)
        ) / world_size
        expected_gradient = 2.0 * mean_square_rank * initial
        checks.append(_close(_scalar(gradient), expected_gradient))
        checks.append(reduced_shapes.count(tuple(raw_parameter.shape)) == 1)
        optimizer.step()
        after_sync = _scalar(parameter)
        checks.append(_close(after_sync, initial - 0.05 * expected_gradient))

        local_step_failed = False
        with model.no_sync():
            optimizer.backward((model(input_value) ** 2).mean())
        try:
            optimizer.step()
        except RuntimeError as error:
            local_step_failed = "still local" in str(error)
        checks.append(local_step_failed)

        optimizer.backward((model(input_value) ** 2).mean())
        accumulated = optimizer.param_groups[0]["grads"][0]
        checks.append(_close(_scalar(accumulated), 2.0 * expected_gradient / initial * after_sync))
        optimizer.step()
        final_weights = dist.all_gather(parameter.reshape((-1,)).stop_grad())
        final_values = final_weights.numpy().reshape(-1).tolist()
        checks.append(all(_close(value, final_values[0]) for value in final_values))

        passed = dist.all_reduce(
            jt.array([1.0 if all(checks) else 0.0]).cuda(0).stop_grad(),
            op="sum")
        if _scalar(passed) != float(world_size):
            raise AssertionError("native NCCL smoke failed: {}".format(checks))
        if rank == 0:
            print("NATIVE_NCCL_SMOKE_OK checks={} nccl_version_code={} "
                  "bf16_tested={}".format(
                      len(checks), nccl_version_code, bf16_tested), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
