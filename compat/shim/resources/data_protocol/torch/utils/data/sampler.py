"""Same-object aliases of the generic compatibility sampler owners."""
from jittor.compat.torch.installers.data import (
    _Sampler as Sampler, _SequentialSampler as SequentialSampler,
    _RandomSampler as RandomSampler, _SubsetRandomSampler as SubsetRandomSampler,
    _BatchSampler as BatchSampler, _DistributedSampler as DistributedSampler,
)
