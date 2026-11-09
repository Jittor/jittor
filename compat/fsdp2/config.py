"""FSDP2 policy, state-dict, and sharding configuration types."""

import enum


# torch's FSDP public enums carry integer values (1, 2, ...). Accelerate and
# other callers convert plugin arguments with ``ShardingStrategy(int(v))`` and
# ``BackwardPrefetch(int(v))``, and compare the members by identity, so the
# values must be the integer member numbers. IntEnum keeps those conversions
# working and additionally allows ``int(member)``.
#
# The ``_missing_`` hooks retain the lowercase strings this shim published
# before the values were corrected, so state-dict snapshots pickled with the
# old enum (see the legacy-pickle contract in the FSDP2 structure tests)
# keep loading.
class StateDictType(enum.IntEnum):
    FULL_STATE_DICT = 1
    LOCAL_STATE_DICT = 2
    SHARDED_STATE_DICT = 3

    @classmethod
    def _missing_(cls, value):
        legacy = {
            "full": cls.FULL_STATE_DICT,
            "local": cls.LOCAL_STATE_DICT,
            "sharded": cls.SHARDED_STATE_DICT,
        }
        return legacy.get(value.lower()) if isinstance(value, str) else None


class ShardingStrategy(enum.IntEnum):
    FULL_SHARD = 1
    SHARD_GRAD_OP = 2
    NO_SHARD = 3
    HYBRID_SHARD = 4
    _HYBRID_SHARD_ZERO2 = 5

    @classmethod
    def _missing_(cls, value):
        legacy = {
            "full_shard": cls.FULL_SHARD,
            "shard_grad_op": cls.SHARD_GRAD_OP,
            "no_shard": cls.NO_SHARD,
            "hybrid_shard": cls.HYBRID_SHARD,
            "hybrid_shard_zero2": cls._HYBRID_SHARD_ZERO2,
        }
        return legacy.get(value.lower()) if isinstance(value, str) else None


class BackwardPrefetch(enum.IntEnum):
    BACKWARD_PRE = 1
    BACKWARD_POST = 2

    @classmethod
    def _missing_(cls, value):
        legacy = {"backward_pre": cls.BACKWARD_PRE, "backward_post": cls.BACKWARD_POST}
        return legacy.get(value.lower()) if isinstance(value, str) else None


class CPUOffload:
    def __init__(self, offload_params=False):
        self.offload_params = bool(offload_params)


class _Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class StateDictConfig(_Config):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class OptimStateDictConfig(_Config):
    def __init__(self, offload_to_cpu=True):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class FullStateDictConfig(StateDictConfig):
    def __init__(self, offload_to_cpu=False, rank0_only=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))
        self.rank0_only = bool(rank0_only)


class LocalStateDictConfig(StateDictConfig):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class ShardedStateDictConfig(LocalStateDictConfig):
    def __init__(self, offload_to_cpu=False, _use_dtensor=False):
        super().__init__(offload_to_cpu=offload_to_cpu)
        self._use_dtensor = bool(_use_dtensor)


class FullOptimStateDictConfig(OptimStateDictConfig):
    def __init__(self, offload_to_cpu=True, rank0_only=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))
        self.rank0_only = bool(rank0_only)


class LocalOptimStateDictConfig(OptimStateDictConfig):
    def __init__(self, offload_to_cpu=False):
        super().__init__(offload_to_cpu=bool(offload_to_cpu))


class ShardedOptimStateDictConfig(LocalOptimStateDictConfig):
    def __init__(self, offload_to_cpu=True, _use_dtensor=False):
        super().__init__(offload_to_cpu=offload_to_cpu)
        self._use_dtensor = bool(_use_dtensor)


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
        if data is not None:
            from ..torch.nested import _torch_make_parameter
            return _torch_make_parameter(data, requires_grad=requires_grad)
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
