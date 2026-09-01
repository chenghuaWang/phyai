"""Shared fixtures for phyai.layers.linear tests.

We avoid spinning up a real process group for unit tests — a mock
``Mesh`` registered under the usual ``"model"`` name is enough for
``resolve_mesh`` to find. Layer tests that exercise collectives at
ws>1 live under a separate multiprocess harness (see
``tests/parallel/multiprocess.py``).

The ``FakeKernel`` / ``make_probe`` factories live here so they're
shared without inter-test-file relative imports (which don't work
under pytest ``--import-mode=importlib`` unless ``phyai/tests`` is on
``pythonpath``).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from phyai.parallel.mesh import Mesh
from phyai.parallel.state import _meshes, register_mesh


def _fake_mesh(
    *,
    name: str = "model",
    sizes: dict[str, int] | None = None,
    ranks: dict[str, int] | None = None,
) -> Mesh:
    sizes = sizes or {}
    ranks = ranks or {}

    def size_of(axis: str) -> int:
        return sizes.get(axis, 1)

    def rank_of(axis: str) -> int:
        return ranks.get(axis, 0)

    tm = MagicMock()
    tm.mesh_dim_names = tuple(sizes.keys()) if sizes else ()
    _names = tm.mesh_dim_names

    def _size(axis):
        if isinstance(axis, str):
            return size_of(axis)
        return size_of(_names[axis])

    tm.size.side_effect = _size
    tm.get_local_rank.side_effect = rank_of
    tm.get_group.side_effect = lambda axis: MagicMock(name=f"pg-{axis}")
    mesh = Mesh(tm, name=name)
    register_mesh(mesh)
    return mesh


@pytest.fixture
def fake_mesh():
    saved = dict(_meshes)
    try:
        yield _fake_mesh
    finally:
        _meshes.clear()
        _meshes.update(saved)
