"""Pure argument validation shared by Torch device parsing and migration."""

def kind_of(value):
    return type(value).__name__


def _check_type(name):
    """The device type ``name`` designates, or a torch-shaped refusal."""
    if name not in _DEVICE_TYPES:
        raise RuntimeError(
            "Expected one of %s device type at start of device string: %s"
            % (", ".join(sorted(_DEVICE_TYPES)), name))
    return name


def _is_index(value):
    """An integer that is an index, not a bool.

    ``numbers.Integral`` rather than ``int``: a numpy integer is not an
    ``int``, and ``torch.device(np.int64(1))`` works in torch.
    """
    import numbers
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _check_index(index):
    """torch rejects a negative device index rather than storing it."""
    if index is None:
        return None
    if not _is_index(index):
        raise TypeError("torch.device(): device index must be an int, not %s"
                        % (kind_of(index),))
    if index < 0:
        raise RuntimeError("Device index must not be negative")
    return int(index)


#: The device types ``torch.device`` accepts, plus jittor's own ``npu``
#: spelling for Ascend. torch rejects anything else with "Expected one of
#: cpu, cuda, ..."; accepting an unknown name is not harmless here, because
#: ``_device_is_cuda``/``_device_is_cpu`` both answer False for it and the
#: tensor then silently stays on the ambient device -- a typo like
#: ``.to("cuda1")`` became "leave it where it is" with no error.
_DEVICE_TYPES = frozenset((
    "cpu", "cuda", "npu", "ipu", "xpu", "mkldnn", "opengl", "opencl", "ideep",
    "hip", "ve", "fpga", "maia", "xla", "lazy", "vulkan", "mps", "meta", "hpu",
    "mtia", "privateuseone",
))
