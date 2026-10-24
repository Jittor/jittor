"""Map fetching reuses its one existing compatibility implementation."""
from jittor.compat.torch.installers.data import _MapDatasetFetcher


class _IterableDatasetFetcher:
    def __init__(self, dataset, auto_collation, collate_fn, drop_last):
        raise NotImplementedError('IterableDataset fetching is not implemented')
