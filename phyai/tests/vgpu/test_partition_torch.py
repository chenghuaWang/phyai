"""Torch backend deliberately rejects multi-shard splitting.

The torch backend supports only single-shard ``create_single``: ATen
exposes no public split / remainder API, and two back-to-back
``GreenContext.create`` calls produce streams that nearly serialise
rather than running disjoint. We assert ``BackendCapabilityError``
rather than letting callers obtain non-disjoint shards.
"""

from __future__ import annotations

import pytest
import torch

import phyai.vgpu as V
from phyai.vgpu.exceptions import BackendCapabilityError


def test_torch_backend_split_by_count_raises():
    V.init(device="cuda:0", backend="torch")
    with pytest.raises(BackendCapabilityError) as ei:
        V.split_device("cuda:0", num_groups=2, min_count=16)
    msg = str(ei.value)
    assert "torch backend" in msg
    assert "flashinfer" in msg


def test_torch_backend_split_by_sm_counts_raises():
    V.init(device="cuda:0", backend="torch")
    with pytest.raises(BackendCapabilityError):
        V.split_device_by_sm_count("cuda:0", sm_counts=[16, 16])


def test_torch_backend_create_single_works():
    """The torch backend should still let users create a single vGPU."""
    V.init(device="cuda:0", backend="torch")
    a = V.vGPU(name="solo", sm_count=64, backend="torch")
    try:
        assert a.shard.sm_count == 64
        assert a.shard.backend == "torch"
        assert isinstance(a.stream, torch.cuda.Stream)
        with a.activate():
            x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
            y = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
            _ = x @ y
        torch.cuda.synchronize()
    finally:
        a.close()
