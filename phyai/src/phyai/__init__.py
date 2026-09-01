"""PhyAI. Main library."""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import TYPE_CHECKING

try:
    __version__ = _pkg_version("phyai")
except PackageNotFoundError:  # raw source tree, not installed
    __version__ = "0.0.0+unknown"


# note(chenghua): Re-export name -> defining submodule. Resolved lazily on first access.
_LAZY: dict[str, str] = {
    # phyai.engine
    "Engine": "phyai.engine",
    "EngineArgs": "phyai.engine",
    "Entry": "phyai.engine",
    "EntryArgs": "phyai.engine",
    # phyai.engine_config
    "BackendConfig": "phyai.engine_config",
    "DeviceConfig": "phyai.engine_config",
    "EngineConfig": "phyai.engine_config",
    "KernelConfig": "phyai.engine_config",
    "ParallelConfig": "phyai.engine_config",
    "RuntimeConfig": "phyai.engine_config",
    "get_engine_config": "phyai.engine_config",
    "init_engine_config": "phyai.engine_config",
    "set_engine_config": "phyai.engine_config",
}

# Type-checkers and IDEs don't run __getattr__; declare the names statically
# so ``from phyai import Engine`` resolves under mypy / pyright / autocomplete.
if TYPE_CHECKING:
    from phyai.engine import Engine, EngineArgs, Entry, EntryArgs
    from phyai.engine_config import (
        BackendConfig,
        DeviceConfig,
        EngineConfig,
        KernelConfig,
        ParallelConfig,
        RuntimeConfig,
        get_engine_config,
        init_engine_config,
        set_engine_config,
    )


def __getattr__(name: str) -> object:
    """Lazily import a re-exported engine symbol (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    obj = getattr(importlib.import_module(module), name)
    globals()[name] = obj  # cache: subsequent access skips __getattr__
    return obj


def __dir__() -> list[str]:
    return sorted({*globals(), *_LAZY})


__all__ = [
    "__version__",
    # engine
    "Engine",
    "EngineArgs",
    "Entry",
    "EntryArgs",
    # engine config
    "EngineConfig",
    "KernelConfig",
    "BackendConfig",
    "DeviceConfig",
    "ParallelConfig",
    "RuntimeConfig",
    "get_engine_config",
    "set_engine_config",
    "init_engine_config",
]
