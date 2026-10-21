"""Read-only status of native collective communicators."""


def get_library(name, *, load=False):
    """The backend library registry, looked up when asked rather than imported.

    A module-level import would put this module on jittor's import-time cycle
    (`tools/lint/check_import_layering.py`) for a registry that a status query
    only needs once it is called.
    """
    from .._runtime.backend_libraries import get_library as _get_library
    return _get_library(name, load=load)


def get_hccl_world_info():
    """Return the initialized HCCL WORLD rank and size, including size one.

    This reads the loaded native communicator; environment variables and the
    presence of compiled collective operators do not establish initialization.
    It never loads a library, compiles an extension or initializes communication.
    If the module is absent or WORLD is not initialized, rank and world_size
    are None. Errors from a loaded module propagate to the caller.
    """
    module = get_library("hccl", load=False)
    if module is None or not module.hccl_is_initialized():
        return {"initialized": False, "rank": None, "world_size": None}
    rank = int(module.hccl_process_group_rank(0))
    world_size = int(module.hccl_process_group_size(0))
    if world_size < 1 or not 0 <= rank < world_size:
        raise RuntimeError(
            "initialized HCCL WORLD has invalid rank/size: {}/{}".format(rank, world_size)
        )
    return {"initialized": True, "rank": rank, "world_size": world_size}
