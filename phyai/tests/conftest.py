"""Test-suite kernel-resolver defaults.

The CUDA requirement lives in the workspace-root ``conftest.py`` — the
suite aborts collection on a machine without CUDA. Layer construction
here uses the engine default ``device.target = "cuda"``; CPU tensors
appear only in device-less logic tests (index arithmetic, policy
parsing, weight-loading I/O).

This conftest isolates the process-level kernel resolver per test. See
:func:`_kernel_resolver_isolation` for why that matters.
"""

from __future__ import annotations

import pytest

from phyai.kernel.bootstrap import kernel_selector_scope


@pytest.fixture(autouse=True)
def _kernel_selector_isolation():
    """Give every test a pristine, uninstalled kernel selector.

    The selector is a process global carrying a policy, a device profile and a
    selection cache. A test that installs one with a custom catalog or a
    forcing policy would otherwise change what every later test selects — and
    pytest collects ``tests/kernel/`` before ``tests/layers/``, so the leak
    direction that matters is exactly the one that happens.

    The scope reads the global directly rather than through
    ``get_kernel_selector``, which builds a default on demand — constructing one
    just to look at it would restore that default instead of "nothing
    installed", and defeat the isolation.
    """

    with kernel_selector_scope():
        yield
