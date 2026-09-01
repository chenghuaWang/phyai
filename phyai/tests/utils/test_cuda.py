"""Unit tests for phyai.utils.cuda device-resolution and probe helpers."""

from __future__ import annotations

import os

import pytest
import torch

from phyai.utils.cuda import (
    available_memory_bytes,
    format_gib,
    init_cublas,
    init_cuda,
    memory_summary,
    resolve_device,
)


@pytest.fixture(autouse=True)
def _restore_globals():
    dtype = torch.get_default_dtype()
    saved_env = dict(os.environ)
    yield
    torch.set_default_dtype(dtype)
    os.environ.clear()
    os.environ.update(saved_env)


# --------------------------------------------------------------------------- #
# resolve_device                                                              #
# --------------------------------------------------------------------------- #


def test_explicit_index_wins_over_launcher_env():
    os.environ["LOCAL_RANK"] = "5"
    assert resolve_device("cuda:2") == torch.device("cuda", 2)


def test_bare_cuda_folds_in_local_rank():
    """The whole point: a torchrun rank must not silently bind GPU 0."""
    os.environ["LOCAL_RANK"] = "3"
    assert resolve_device("cuda") == torch.device("cuda", 3)


def test_bare_cuda_without_launcher_is_device_zero():
    os.environ.pop("LOCAL_RANK", None)
    assert resolve_device("cuda") == torch.device("cuda", 0)


def test_non_cuda_device_passes_through():
    os.environ["LOCAL_RANK"] = "3"
    assert resolve_device("cpu") == torch.device("cpu")


def test_accepts_a_torch_device_object():
    assert resolve_device(torch.device("cuda", 1)) == torch.device("cuda", 1)


# --------------------------------------------------------------------------- #
# init_cuda / init_cublas                                                     #
# --------------------------------------------------------------------------- #


def test_init_cuda_sets_default_dtype():
    torch.set_default_dtype(torch.float32)
    assert init_cuda("cpu", torch.float64) is None
    assert torch.get_default_dtype() is torch.float64


def test_init_cublas_does_not_raise():
    # Handle warm-up must be silent whether or not a context already exists.
    assert init_cublas() is None


# --------------------------------------------------------------------------- #
# memory probing                                                              #
# --------------------------------------------------------------------------- #


def test_memory_summary_of_an_invalid_device_degrades_to_zero():
    assert memory_summary("cuda:999") == (0, 0)


def test_available_memory_bytes_matches_summary():
    free, _total = memory_summary()
    assert available_memory_bytes() == free


def test_format_gib():
    assert format_gib(0) == "0.00"
    assert format_gib(1 << 30) == "1.00"
    assert format_gib(3 * (1 << 29)) == "1.50"
