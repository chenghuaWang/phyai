"""Unit tests for phyai.utils.env_setup — the write side of the environment."""

from __future__ import annotations

import os
import resource

import pytest
from phyai.utils.env_setup import (
    TUNED_ENV_VARS,
    init_env,
    set_ulimit,
    init_process_debug,
)


@pytest.fixture(autouse=True)
def _restore_environ():
    """Snapshot/restore os.environ — every test here mutates it."""
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _clear_tuned():
    for var in TUNED_ENV_VARS:
        os.environ.pop(var.name, None)
    os.environ.pop("PHYAI_SKIP_ENV_SETUP", None)


def _cuda_vars() -> tuple[str, ...]:
    return tuple(v.name for v in TUNED_ENV_VARS if v.applies_when(1, "cuda"))


# --------------------------------------------------------------------------- #
# init_env                                                                    #
# --------------------------------------------------------------------------- #


def test_init_env_writes_applicable_vars():
    _clear_tuned()
    applied = init_env(world_size=1, device_type="cuda")
    assert set(applied) == set(_cuda_vars())
    for name, value in applied.items():
        assert os.environ[name] == value


def test_init_env_never_overwrites_a_preset_value():
    _clear_tuned()
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    applied = init_env(world_size=1, device_type="cuda")
    assert "CUDA_DEVICE_MAX_CONNECTIONS" not in applied
    assert os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] == "1"


def test_init_env_cpu_target_writes_nothing():
    _clear_tuned()
    assert init_env(world_size=1, device_type="cpu") == {}


def test_init_env_is_idempotent():
    _clear_tuned()
    first = init_env(world_size=1, device_type="cuda")
    assert first
    assert init_env(world_size=1, device_type="cuda") == {}


def test_init_env_skipped_by_env_var():
    _clear_tuned()
    os.environ["PHYAI_SKIP_ENV_SETUP"] = "1"
    assert init_env(world_size=1, device_type="cuda") == {}
    for name in _cuda_vars():
        assert name not in os.environ


def test_recommended_only_vars_are_never_written():
    """The NCCL / allocator entries stay documentation until measured."""
    _clear_tuned()
    applied = init_env(world_size=8, device_type="cuda")
    for name in (
        "NCCL_CUMEM_ENABLE",
        "NCCL_NVLS_ENABLE",
        "NCCL_GRAPH_MIXING_SUPPORT",
        "PYTORCH_CUDA_ALLOC_CONF",
    ):
        assert name not in applied
        assert name not in os.environ


def test_every_tuned_var_documents_why():
    for var in TUNED_ENV_VARS:
        assert var.why.strip(), f"{var.name} has no rationale"


# --------------------------------------------------------------------------- #
# set_ulimit / init_process_debug                                             #
# --------------------------------------------------------------------------- #


def test_set_ulimit_never_lowers_a_limit():
    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    set_ulimit(target_soft_limit=1)
    assert resource.getrlimit(resource.RLIMIT_NOFILE) == before


def test_set_ulimit_warns_instead_of_raising_when_capped():
    """A hard limit below the target is the operator's call, not an error."""
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = (1 << 40) if hard == resource.RLIM_INFINITY else hard + 1
    set_ulimit(target_soft_limit=target)  # must not raise
    soft_after, hard_after = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert hard_after == hard
    if hard != resource.RLIM_INFINITY:
        assert soft_after <= hard


def test_init_process_debug_sets_requested_title(monkeypatch):
    titles = []
    monkeypatch.setattr("setproctitle.setproctitle", titles.append)

    init_process_debug()
    init_process_debug(title="phyai::test_DP1_TP2")

    assert titles == ["phyai::test_DP1_TP2"]
