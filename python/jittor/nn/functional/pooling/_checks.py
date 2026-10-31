"""Input checks shared by the pooling entry points."""

_LAYOUTS = {3: "(N, C, L)", 4: "(N, C, H, W)", 5: "(N, C, D, H, W)"}


def expect_rank(x, rank, name):
    """Reject an input of the wrong rank before its shape is unpacked.

    Unpacking ``N, C, H, W = x.shape`` first reported a 1-D input as
    "not enough values to unpack (expected 4, got 1)", from inside jittor.
    """
    ndim = len(x.shape)
    if ndim != rank:
        # Imported here, on the failing path only: at import time this module
        # would join jittor's import cycle.
        import jittor as jt
        raise jt.JittorShapeError(
            "%s expected a %d-D input %s, but got a %d-D input of shape %s."
            % (name, rank, _LAYOUTS[rank], ndim, list(x.shape)))
