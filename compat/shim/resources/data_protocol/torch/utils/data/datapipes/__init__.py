"""Real base/serialization protocol, not the full Native operator catalogue.

Only the base protocol and graph sharding module are bundled. Other built-in
functional transforms, dataframe pipes, and DataPipe execution/profiling are
not declared supported by this compatibility subset.
"""
from .datapipe import DataChunk, DFIterDataPipe, IterDataPipe, MapDataPipe
