"""PyTorch's caching-allocator statistics and summary table, for Jittor's pools.

``memory_stats`` has every key ``torch.cuda.memory_stats`` has, from counters
the SFRL pools keep on every allocation, free and segment change
(``sfrl_device_memory_stats`` in ``src/mem/allocator/sfrl_allocator.cc``).
``memory_summary`` lays them out exactly as ``torch.cuda.memory_summary`` does.

Where the allocators differ, the numbers say what Jittor's does: small
segments are 1 MiB, a block belongs to its segment's pool, and with no
``max_split_size`` there are no oversize allocations or segments.
"""

import collections

import jittor_core as _core

__all__ = ["memory_stats", "memory_summary", "reset_accumulated_memory_stats"]

# The order of SfrlStat, SfrlPool and SfrlCounter.
_STATS = ("allocation", "segment", "active", "inactive_split", "allocated_bytes",
          "reserved_bytes", "active_bytes", "inactive_split_bytes", "requested_bytes")
_POOLS = ("all", "small_pool", "large_pool")
_PARTS = ("current", "peak", "allocated", "freed")
_COUNTERS = ("num_alloc_retries", "num_ooms", "num_device_alloc", "num_device_free")


def _device(device):
    return int(_core.current_device()) if device is None else int(device)


def memory_stats(device=None):
    """The allocator statistics of ``device`` (default: the current one; -1:
    the host pools), keyed and sorted as ``torch.cuda.memory_stats`` keys them."""
    flat = list(_core.device_memory_stats(_device(device)))
    n = len(_STATS) * len(_POOLS) * len(_PARTS) + len(_COUNTERS)
    if not flat:
        flat = [0] * n
    if len(flat) != n:
        raise RuntimeError("device_memory_stats returned %d values, expected %d" % (len(flat), n))
    values = iter(flat)
    stats = {}
    for stat in _STATS:
        for pool in _POOLS:
            for part in _PARTS:
                stats["%s.%s.%s" % (stat, pool, part)] = next(values)
    for counter in _COUNTERS:
        stats[counter] = next(values)
    for stat in ("oversize_allocations", "oversize_segments"):
        for part in _PARTS:
            stats[stat + "." + part] = 0
    stats["max_split_size"] = -1
    stats["num_sync_all_streams"] = 0
    return collections.OrderedDict(sorted(stats.items()))


def reset_accumulated_memory_stats(device=None):
    """Restart the allocated/freed totals and the counters of ``device``."""
    _core.reset_device_memory_accumulated(_device(device))


def _format_size(size, scale):
    prefixes = ("B  ", "KiB", "MiB", "GiB", "TiB", "PiB")
    prefix = prefixes[0]
    for next_prefix in prefixes[1:]:
        if scale < 768 * 1024:
            break
        prefix = next_prefix
        size //= 1024
        scale /= 1024
    return "%6d %s" % (size, prefix)


def _format_count(count, scale):
    prefixes = (" ", "K", "M")
    prefix = prefixes[0]
    for next_prefix in prefixes[1:]:
        if scale < 750 * 1000:
            break
        prefix = next_prefix
        count //= 1000
        scale /= 1000
    return "%7d %s " % (count, prefix)


_ROWS = (("allocated_bytes", "Allocated memory", _format_size),
         ("active_bytes", "Active memory", _format_size),
         ("requested_bytes", "Requested memory", _format_size),
         ("reserved_bytes", "GPU reserved memory", _format_size),
         ("inactive_split_bytes", "Non-releasable memory", _format_size),
         ("allocation", "Allocations", _format_count),
         ("active", "Active allocs", _format_count),
         ("segment", "GPU reserved segments", _format_count),
         ("inactive_split", "Non-releasable allocs", _format_count))
_OVERSIZE_ROWS = (("oversize_allocations", "Oversize allocations"),
                  ("oversize_segments", "Oversize GPU segments"))


def memory_summary(device=None, abbreviated=False, stats=None, framework="Jittor"):
    """The table ``torch.cuda.memory_summary`` prints, from :func:`memory_stats`.

    A sub-pool row is scaled by its "all" row, as PyTorch scales it, so the
    rows of one metric share a unit."""
    device = _device(device)
    stats = memory_stats(device) if stats is None else stats
    title = "%s CUDA memory summary, device ID %d" % (framework, device)
    lines = ["=" * 75,
             " %s%s " % (" " * 17, title.ljust(56)),
             "-" * 75,
             "  %s CUDA OOMs: %-12d | %s cudaMalloc retries: %-8d  " % (
                 " " * 9, stats["num_ooms"], " " * 6, stats["num_alloc_retries"]),
             "=" * 75,
             "        Metric         | Cur Usage  | Peak Usage | Tot Alloc  | Tot Freed  "]
    for key, name, formatter in _ROWS:
        lines.append("-" * 75)
        rows = [("all", name)]
        if not abbreviated:
            rows += [("large_pool", "      from large pool"), ("small_pool", "      from small pool")]
        scale = None
        for pool, label in rows:
            values = [stats["%s.%s.%s" % (key, pool, part)] for part in _PARTS]
            if scale is None:
                scale = values
            lines.append(" %-21s | %s " % (label, " | ".join(
                formatter(v, s) for v, s in zip(values, scale))))
    for key, name in _OVERSIZE_ROWS:
        lines.append("-" * 75)
        values = [stats["%s.%s" % (key, part)] for part in _PARTS]
        lines.append(" %-21s | %s " % (name, " | ".join(_format_count(v, v) for v in values)))
    lines.append("=" * 75)
    return "|" + "|\n|".join(lines) + "|\n"
