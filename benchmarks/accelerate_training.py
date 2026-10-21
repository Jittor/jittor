"""Compatibility entry point for the Accelerate ASV benchmark.

The implementation and its shared helpers live under :mod:`benchmarks.asv`.
"""
from .asv.accelerate_training import AccelerateTrainingBenchmarks

__all__ = ["AccelerateTrainingBenchmarks"]
