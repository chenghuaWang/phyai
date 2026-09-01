"""Unit tests for phyai.utils.torch_setup — process-wide torch state.

Every test restores what it changed: these are process globals, and a leak
would silently reconfigure the rest of the suite.
"""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from phyai.utils.torch_setup import (
    disable_grad,
    init_seed,
    init_threads,
    local_rank,
)


@pytest.fixture(autouse=True)
def _restore_torch_globals():
    threads = torch.get_num_threads()
    grad = torch.is_grad_enabled()
    saved_env = dict(os.environ)
    yield
    torch.set_num_threads(threads)
    torch.set_grad_enabled(grad)
    os.environ.clear()
    os.environ.update(saved_env)


# --------------------------------------------------------------------------- #
# init_threads                                                                #
# --------------------------------------------------------------------------- #


def test_cpu_target_is_left_alone():
    """On a CPU target the intra-op pool IS the compute; do not shrink it."""
    before = torch.get_num_threads()
    assert init_threads(device_type="cpu") == before
    assert torch.get_num_threads() == before


def test_accelerator_target_defaults_to_one_thread():
    assert init_threads(device_type="cuda") == 1
    assert torch.get_num_threads() == 1


def test_explicit_count_overrides_the_auto_choice():
    assert init_threads(device_type="cuda", num_threads=3) == 3
    assert init_threads(device_type="cpu", num_threads=2) == 2


def test_explicit_count_must_be_positive():
    with pytest.raises(ValueError, match="num_threads"):
        init_threads(device_type="cuda", num_threads=0)


# --------------------------------------------------------------------------- #
# init_seed                                                                   #
# --------------------------------------------------------------------------- #


def test_none_seed_is_a_no_op():
    torch.manual_seed(1234)
    expected = torch.rand(4)
    torch.manual_seed(1234)
    init_seed(None)
    assert torch.equal(torch.rand(4), expected)


def test_seed_makes_all_three_global_rngs_reproducible():
    init_seed(7)
    first = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    init_seed(7)
    second = (random.random(), float(np.random.rand()), float(torch.rand(1)))
    assert first == second


def test_seed_does_not_touch_generator_local_rng():
    """Model code owning a RandomState must be immune to process seeding.

    The cosmos3 samplers rely on this: their per-request
    ``np.random.RandomState(seed)`` noise has to be bit-identical whatever
    else the process did.
    """
    expected = np.random.RandomState(99).rand(4)
    init_seed(12345)
    assert np.array_equal(np.random.RandomState(99).rand(4), expected)


# --------------------------------------------------------------------------- #
# disable_grad                                                                #
# --------------------------------------------------------------------------- #


def test_disable_grad_turns_autograd_off():
    torch.set_grad_enabled(True)
    disable_grad()
    assert not torch.is_grad_enabled()


def test_disable_grad_is_idempotent():
    torch.set_grad_enabled(False)
    disable_grad()
    assert not torch.is_grad_enabled()


# --------------------------------------------------------------------------- #
# local_rank                                                                  #
# --------------------------------------------------------------------------- #


def test_local_rank_reads_launcher_env():
    os.environ["LOCAL_RANK"] = "3"
    assert local_rank() == 3


def test_local_rank_defaults_to_zero_without_launcher():
    os.environ.pop("LOCAL_RANK", None)
    assert local_rank() == 0


def test_local_rank_falls_back_on_garbage():
    os.environ["LOCAL_RANK"] = "not-an-int"
    assert local_rank() == 0
