"""Public global-to-group rank translation matches native process-group maps."""

from _helpers.child_process import run_child_script


_CPU_ENV = {
    "JITTOR_TORCH_SHIM": "1", "JITTOR_TEST_DEVICES": "cpu",
    "CUDA_VISIBLE_DEVICES": "", "use_cuda": "0", "use_nccl": "0",
    "use_mpi": "0", "JITTOR_TORCH_DISTRIBUTED_AUTO_INIT": "1",
}


def test_registered_rank_translation_and_public_namespace_identity():
    # No tensor communication is needed for a rank-map query. A local Store
    # supplies a three-rank metadata world; real multiprocess CPU transport is
    # independently covered by test_distributed_host_group.py.
    result = run_child_script(
        r'''
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

assert hasattr(dist, "get_group_rank"), "public get_group_rank is missing"
assert dist.get_group_rank is c10d.get_group_rank
store = dist.TCPStore()
dist.init_process_group(backend="gloo", rank=0, world_size=3, store=store)
try:
    world = dist.group.WORLD
    group = dist.new_group([0, 2], backend="gloo")
    # Native WORLD translation is identity, including values outside its
    # size; subgroup membership checks must not be applied to this shortcut.
    for value in (-1, 0, 2, 99):
        assert dist.get_group_rank(world, value) == value
    assert dist.get_group_rank(group, 0) == 0
    assert dist.get_group_rank(group, 2) == 1
    assert c10d.get_group_rank(group, 2) == 1

    def fails(group, value, fragment):
        try:
            dist.get_group_rank(group, value)
        except ValueError as error:
            assert fragment in str(error), str(error)
        else:
            raise AssertionError("invalid rank/group was accepted")

    for value in (-1, 1, 3, "2"):
        fails(group, value, "not part of group")
    for unregistered in (None, object(), dist.GroupMember.NON_GROUP_MEMBER):
        fails(unregistered, 0, "not registered")
    dist.destroy_process_group(group)
    fails(group, 0, "not registered")
    assert dist.get_group_rank(world, 2) == 2
finally:
    dist.destroy_process_group()
''', env=_CPU_ENV, text=True, merge_stderr=True)
    assert result.returncode == 0, result.stdout
