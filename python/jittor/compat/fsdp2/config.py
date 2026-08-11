"""FSDP2 policy, state-dict, and sharding configuration types."""

import enum

import jittor as jt


class StateDictType(enum.Enum):
    FULL_STATE_DICT = "full"
    LOCAL_STATE_DICT = "local"
    SHARDED_STATE_DICT = "sharded"


class ShardingStrategy(enum.Enum):
    FULL_SHARD = "full_shard"
    SHARD_GRAD_OP = "shard_grad_op"
    NO_SHARD = "no_shard"
    HYBRID_SHARD = "hybrid_shard"
    _HYBRID_SHARD_ZERO2 = "hybrid_shard_zero2"


class BackwardPrefetch(enum.Enum):
    BACKWARD_PRE = "backward_pre"
    BACKWARD_POST = "backward_post"


class CPUOffload:
    def __init__(self, offload_params=False):
        self.offload_params = bool(offload_params)


class _Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StateDictConfig(_Config):
    pass


class OptimStateDictConfig(_Config):
    pass


class FullStateDictConfig(StateDictConfig):
    def __init__(self, offload_to_cpu=False, rank0_only=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu), rank0_only=bool(rank0_only))


class LocalStateDictConfig(StateDictConfig):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class ShardedStateDictConfig(LocalStateDictConfig):
    pass


class FullOptimStateDictConfig(OptimStateDictConfig):
    def __init__(self, offload_to_cpu=False, rank0_only=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu), rank0_only=bool(rank0_only))


class LocalOptimStateDictConfig(OptimStateDictConfig):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class ShardedOptimStateDictConfig(LocalOptimStateDictConfig):
    pass


class StateDictSettings:
    def __init__(self, state_dict_type=StateDictType.FULL_STATE_DICT,
                 state_dict_config=None, optim_state_dict_config=None):
        self.state_dict_type = state_dict_type
        self.state_dict_config = state_dict_config
        self.optim_state_dict_config = optim_state_dict_config


class OptimStateKeyType(enum.Enum):
    PARAM_NAME = "param_name"
    PARAM_ID = "param_id"


class FlatParameter:
    def __new__(cls, data=None, requires_grad=True, *args, **kwargs):
        maker = getattr(jt, "_torch_make_parameter", None)
        if data is not None and callable(maker):
            return maker(data, requires_grad=requires_grad)
        return object.__new__(cls)

    def __init__(self, data=None, requires_grad=True, *args, **kwargs):
        self.data = data
        self.requires_grad = requires_grad


class MixedPrecisionPolicy:
    def __init__(self, param_dtype=None, reduce_dtype=None, output_dtype=None,
                 cast_forward_inputs=True, **kwargs):
        self.param_dtype = param_dtype
        self.reduce_dtype = reduce_dtype
        self.output_dtype = output_dtype
        self.cast_forward_inputs = cast_forward_inputs
        for k, v in kwargs.items():
            setattr(self, k, v)


class MixedPrecision(MixedPrecisionPolicy):
    pass


class OffloadPolicy:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class CPUOffloadPolicy(OffloadPolicy):
    def __init__(self, pin_memory=True, **kwargs):
        super().__init__(pin_memory=pin_memory, **kwargs)


class NoOffloadPolicy(OffloadPolicy):
    pass


class DataParallelMeshDims:
    def __init__(self, shard=None, replicate=None):
        self.shard = shard
        self.replicate = replicate
        self.shard_names = tuple(() if shard is None else
                                 (shard if isinstance(shard, (tuple, list)) else (shard,)))
        self.replicate_names = tuple(() if replicate is None else
                                     (replicate if isinstance(replicate, (tuple, list))
                                      else (replicate,)))


class UnshardHandle:
    def __init__(self, module=None):
        self.module = module

    def wait(self):
        return None


_EXPORTS = (
    "StateDictType",
    "ShardingStrategy",
    "BackwardPrefetch",
    "CPUOffload",
    "_Config",
    "StateDictConfig",
    "OptimStateDictConfig",
    "FullStateDictConfig",
    "LocalStateDictConfig",
    "ShardedStateDictConfig",
    "FullOptimStateDictConfig",
    "LocalOptimStateDictConfig",
    "ShardedOptimStateDictConfig",
    "StateDictSettings",
    "OptimStateKeyType",
    "FlatParameter",
    "MixedPrecisionPolicy",
    "MixedPrecision",
    "OffloadPolicy",
    "CPUOffloadPolicy",
    "NoOffloadPolicy",
    "DataParallelMeshDims",
    "UnshardHandle",
)
