"""End-to-end smoke tests for the vGPU class.

Validates: ``activate()`` enters the vGPU's stream + mempool scope,
GEMM run inside the scope produces numerically correct results, and
``close()`` is idempotent.
"""

from __future__ import annotations

import pytest
import torch

import phyai.vgpu as V


def _flashinfer_available() -> bool:
    try:
        import flashinfer.green_ctx  # noqa: F401
    except ImportError:
        return False
    return True


def test_vgpu_activate_runs_gemm_correctly():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    try:
        dev = torch.device("cuda:0")
        torch.manual_seed(0)
        x = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        y = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
        z_ref = (x @ y).clone()
        torch.cuda.synchronize()

        with a.activate():
            z = x @ y
        torch.cuda.synchronize()

        rel_err = ((z - z_ref).abs().mean() / z_ref.abs().mean()).item()
        assert rel_err < 1e-2, rel_err
    finally:
        a.close()


def test_vgpu_activate_uses_its_stream():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    try:
        # ``torch.cuda.current_stream()`` returns a fresh Python wrapper
        # each call, so compare by underlying handle.
        before_handle = torch.cuda.current_stream().cuda_stream
        with a.activate():
            assert torch.cuda.current_stream().cuda_stream == a.stream.cuda_stream
        # After exit, we should be back on whatever stream we were on.
        assert torch.cuda.current_stream().cuda_stream == before_handle
    finally:
        a.close()


def test_vgpu_close_is_idempotent():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    a.close()
    a.close()  # second close must not raise


def test_vgpu_activate_after_close_raises():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    a.close()
    with pytest.raises(RuntimeError, match="closed"):
        with a.activate():
            pass


def test_vgpu_own_mem_pool_default_true():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64)
    try:
        assert a.mem_pool is not None
    finally:
        a.close()


def test_vgpu_no_mem_pool_when_disabled():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a = V.vGPU(name="a", sm_count=64, own_mem_pool=False)
    try:
        assert a.mem_pool is None
    finally:
        a.close()


def test_create_vgpus_returns_correct_number():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    a, b = V.create_vgpus(
        device="cuda:0",
        sm_counts=[64, 64],
        names=["a", "b"],
    )
    try:
        assert a.name == "a"
        assert b.name == "b"
        assert a.shard.sm_count == 64
        assert b.shard.sm_count == 64
    finally:
        a.close()
        b.close()


def test_create_vgpus_include_remainder_appends_extra_vgpu():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    vgpus = V.create_vgpus(
        device="cuda:0",
        sm_counts=[16, 16],
        include_remainder_vgpu=True,
    )
    try:
        # Two requested + one remainder.
        assert len(vgpus) == 3
        assert vgpus[-1].shard.is_remainder is True
    finally:
        for v in vgpus:
            v.close()


def test_create_vgpus_names_length_mismatch_raises():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    with pytest.raises(ValueError, match="names length"):
        V.create_vgpus(
            device="cuda:0",
            sm_counts=[16, 16],
            names=["only-one"],
        )


def test_vgpu_from_shard():
    if not _flashinfer_available():
        pytest.skip("flashinfer required for default backend")
    V.init(device="cuda:0")
    shards = V.split_device("cuda:0", num_groups=2, min_count=16)
    a = V.vGPU.from_shard(shards[0], name="a")
    try:
        assert a.name == "a"
        assert a.shard is shards[0]
        assert a.stream is shards[0].stream
    finally:
        a.close()
