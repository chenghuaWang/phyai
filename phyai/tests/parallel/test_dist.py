"""Unit tests for phyai.parallel.dist.init_dist.

No real process group is created: ``init_process_group`` is stubbed so the
ordering and argument contract can be asserted on CPU, which is where CI
runs.
"""

from __future__ import annotations

import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist

import phyai.parallel.dist as D


@pytest.fixture(autouse=True)
def _restore_environ():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def trace(monkeypatch):
    """Record the ordering of set_device / init_process_group calls."""
    calls: list[tuple[str, tuple, dict]] = []

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        D.dist,
        "init_process_group",
        lambda *a, **kw: calls.append(("init_process_group", a, kw)),
    )
    monkeypatch.setattr(
        torch.cuda, "set_device", lambda *a, **kw: calls.append(("set_device", a, kw))
    )
    return calls


def test_single_rank_cpu_creates_nothing(trace):
    assert D.init_dist(world_size=1, device_type="cpu") is False
    assert trace == []


def test_single_rank_cuda_only_pins_the_device(trace):
    os.environ["LOCAL_RANK"] = "2"
    assert D.init_dist(world_size=1, device_type="cuda") is False
    assert [name for name, _a, _kw in trace] == ["set_device"]
    assert trace[0][1][0] == torch.device("cuda", 2)


def test_device_is_pinned_before_the_group_is_built(trace):
    """NCCL binds to the current device while building its communicator."""
    os.environ["LOCAL_RANK"] = "3"
    assert D.init_dist(world_size=4, device_type="cuda") is True
    assert [name for name, _a, _kw in trace] == ["set_device", "init_process_group"]
    assert trace[0][1][0] == torch.device("cuda", 3)


def test_timeout_is_forwarded(trace):
    D.init_dist(world_size=2, device_type="cpu", timeout=timedelta(seconds=42))
    _name, _args, kwargs = trace[-1]
    assert kwargs["timeout"] == timedelta(seconds=42)


def test_no_timeout_leaves_the_torch_default(trace):
    D.init_dist(world_size=2, device_type="cpu")
    _name, _args, kwargs = trace[-1]
    assert "timeout" not in kwargs


def test_rank_comes_from_the_launcher(trace):
    os.environ["RANK"] = "5"
    os.environ["LOCAL_RANK"] = "1"
    D.init_dist(world_size=8, device_type="cpu")
    _name, args, kwargs = trace[-1]
    assert args == ("gloo",)
    assert kwargs["rank"] == 5
    assert kwargs["world_size"] == 8


def test_rendezvous_defaults_are_filled_in(trace):
    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"):
        os.environ.pop(key, None)
    D.init_dist(world_size=2, device_type="cpu")
    assert os.environ["MASTER_ADDR"] == "127.0.0.1"
    assert os.environ["WORLD_SIZE"] == "2"


def test_existing_group_is_reused_not_owned(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 4)
    assert D.init_dist(world_size=4, device_type="cuda") is False


def test_mismatched_existing_group_raises(monkeypatch):
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    with pytest.raises(ValueError, match="does not match"):
        D.init_dist(world_size=4, device_type="cuda")
