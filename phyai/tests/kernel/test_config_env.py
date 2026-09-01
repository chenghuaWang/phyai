"""Env-var overlay for the kernel policy config.

``PHYAI_KERNEL_CONFIG`` / ``PHYAI_KERNEL_PROFILE`` /
``PHYAI_KERNEL_AUTOTUNE_CACHE`` are wired through
:meth:`phyai.engine_config.EngineConfig.from_env`, but nothing exercised
them. They are the only way to point a deployment at a policy file without
editing code, so a silent break here is expensive and invisible.
"""

from __future__ import annotations

import pytest

from phyai.engine_config import EngineConfig


POLICY_YAML = """\
schema: phyai.kernel/v1
profile: autotune
"""


@pytest.fixture(autouse=True)
def _clear_kernel_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start from a clean slate; the host may export these already."""

    for name in (
        "PHYAI_KERNEL_CONFIG",
        "PHYAI_KERNEL_PROFILE",
        "PHYAI_KERNEL_AUTOTUNE_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_no_kernel_env_leaves_config_at_defaults() -> None:
    kernel = EngineConfig.from_env().kernel
    assert kernel.config_path is None
    # ``auto()`` deliberately leaves the profile unset so an external YAML
    # can select ``autotune``; the effective default stays ``static``.
    assert kernel.profile is None
    assert kernel.policy().profile == "static"
    assert kernel.autotune_cache is None


def test_kernel_config_path_env_is_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_YAML, encoding="utf-8")
    monkeypatch.setenv("PHYAI_KERNEL_CONFIG", str(path))

    kernel = EngineConfig.from_env().kernel
    assert kernel.config_path == str(path)
    # The env var must reach the loader, not just the dataclass field.
    assert kernel.policy().profile == "autotune"
    assert kernel.policy().source == str(path)


def test_kernel_profile_env_overrides_yaml(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``PHYAI_KERNEL_PROFILE`` is the outermost override.

    The YAML says ``autotune``; the env var must still win, otherwise there
    is no way to force deterministic selection on a machine whose policy
    file asks for measurement.
    """

    path = tmp_path / "policy.yaml"
    path.write_text(POLICY_YAML, encoding="utf-8")
    monkeypatch.setenv("PHYAI_KERNEL_CONFIG", str(path))
    monkeypatch.setenv("PHYAI_KERNEL_PROFILE", "static")

    kernel = EngineConfig.from_env().kernel
    assert kernel.profile == "static"
    assert kernel.policy().profile == "static"


def test_kernel_autotune_cache_env_is_applied(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cache = tmp_path / "autotune.json"
    monkeypatch.setenv("PHYAI_KERNEL_AUTOTUNE_CACHE", str(cache))

    assert EngineConfig.from_env().kernel.autotune_cache == str(cache)


def test_invalid_kernel_profile_env_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PHYAI_KERNEL_PROFILE", "turbo")

    with pytest.raises(ValueError, match="profile"):
        EngineConfig.from_env()


def test_kernel_env_overlays_an_explicit_base(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """``from_env(base)`` must overlay rather than discard the base."""

    cache = tmp_path / "autotune.json"
    monkeypatch.setenv("PHYAI_KERNEL_AUTOTUNE_CACHE", str(cache))

    base = EngineConfig.auto()
    merged = EngineConfig.from_env(base)
    assert merged.kernel.autotune_cache == str(cache)
    # Untouched sibling fields survive the overlay.
    assert merged.kernel.config_path == base.kernel.config_path
    assert merged.kernel.profile == base.kernel.profile
    assert merged.device.target == base.device.target
