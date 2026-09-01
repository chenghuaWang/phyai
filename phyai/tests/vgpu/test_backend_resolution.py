"""Backend resolution priority tests (explicit > env > auto)."""

from __future__ import annotations

import warnings

import pytest

from phyai.vgpu import backend as backend_mod
from phyai.vgpu.exceptions import VGPUNotApplicableError


@pytest.fixture(autouse=True)
def _clean_backend_state(monkeypatch):
    """Restore registry state and current pointer between tests."""
    saved_backends = dict(backend_mod._BACKENDS)
    saved_current = backend_mod._CURRENT
    monkeypatch.delenv("PHYAI_VGPU_BACKEND", raising=False)
    yield
    backend_mod._BACKENDS.clear()
    backend_mod._BACKENDS.update(saved_backends)
    backend_mod._CURRENT = saved_current


class _FakeFlashInfer:
    name = "flashinfer"


class _FakeTorch:
    name = "torch"


def _install_fakes(*, flashinfer_available: bool = True) -> None:
    """Replace registry and toggle the auto-detection probe."""
    backend_mod._BACKENDS.clear()
    backend_mod._BACKENDS["flashinfer"] = _FakeFlashInfer
    backend_mod._BACKENDS["torch"] = _FakeTorch
    backend_mod._flashinfer_available = lambda: flashinfer_available  # type: ignore[assignment]


def test_explicit_overrides_env(monkeypatch):
    _install_fakes()
    monkeypatch.setenv("PHYAI_VGPU_BACKEND", "torch")
    b = backend_mod.resolve("flashinfer")
    assert b.name == "flashinfer"


def test_env_overrides_auto(monkeypatch):
    _install_fakes()
    monkeypatch.setenv("PHYAI_VGPU_BACKEND", "torch")
    b = backend_mod.resolve(None)
    assert b.name == "torch"


def test_auto_picks_flashinfer_when_available():
    _install_fakes(flashinfer_available=True)
    b = backend_mod.resolve(None)
    assert b.name == "flashinfer"


def test_auto_falls_back_to_torch_with_warning():
    _install_fakes(flashinfer_available=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = backend_mod.resolve(None)
    assert b.name == "torch"
    fallback_warns = [x for x in w if "falling back" in str(x.message)]
    assert len(fallback_warns) == 1


def test_explicit_torch_does_not_warn():
    """Explicit choice should never trigger the fallback warning."""
    _install_fakes(flashinfer_available=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        b = backend_mod.resolve("torch")
    assert b.name == "torch"
    fallback_warns = [x for x in w if "falling back" in str(x.message)]
    assert not fallback_warns


def test_unknown_backend_name_raises():
    _install_fakes()
    with pytest.raises(VGPUNotApplicableError):
        backend_mod.resolve("nonexistent")


def test_get_backend_without_init_raises():
    _install_fakes()
    backend_mod._CURRENT = None
    with pytest.raises(RuntimeError, match="no active backend"):
        backend_mod.get_backend()
