"""Same-object aliases of the generic compatibility Dataset owners."""
from jittor.compat.torch.installers.data import (
    _TorchDataset as Dataset, _IterableDataset as IterableDataset,
    _TensorDataset as TensorDataset, _ConcatDataset as ConcatDataset,
    _Subset as Subset,
)
