"""A ``device="meta"`` factory argument is observable on the result.

Transformers' meta-state-dict path asks for metadata-only tensors through the
factory argument as well as through ``with torch.device("meta")``. Jittor has
no meta backend, so the result is really allocated on the host and reports
``meta`` through the same mark the context uses; it must not silently claim a
CPU identity.
"""

import torch


def test_factory_meta_argument_reports_meta():
    value = torch.empty((2, 3), device="meta")
    assert value.device.type == "meta"
    assert value.is_meta
    assert tuple(value.shape) == (2, 3)


def test_meta_factory_result_does_not_leak_into_later_factories():
    assert torch.zeros(4, device="meta").device.type == "meta"
    assert torch.zeros(4).device.type == "cpu"
