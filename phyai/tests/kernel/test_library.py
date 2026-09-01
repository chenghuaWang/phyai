"""Library availability probing.

The behaviour under test is the difference between "installed" and
"importable". A package whose native extension fails to load — flashinfer on
a host with a mismatched CUDA or a missing driver library — is *installed*, so
a module-spec check reports it as present. Every kernel gated on it then
claims to be eligible, preparation throws, the selector silently falls through
to the next candidate, and the trace records a viable kernel that never was.
One raised exception per cache miss, and a lie in the diagnostics.
"""

from __future__ import annotations

import pytest

from phyai.kernel import library


@pytest.fixture(autouse=True)
def _clear_probe_cache():
    library.reset_library_probes()
    yield
    library.reset_library_probes()


@pytest.fixture
def fake_import(monkeypatch: pytest.MonkeyPatch):
    """Replace ``import_module`` with a scripted one, recording each call."""

    calls: list[str] = []
    real = library.importlib.import_module

    def install(behaviour):
        def patched(target: str):
            calls.append(target)
            outcome = behaviour(target)
            if isinstance(outcome, BaseException):
                raise outcome
            return real(target) if outcome is None else outcome

        monkeypatch.setattr(library.importlib, "import_module", patched)
        return calls

    return install


def test_importable_module_is_available() -> None:
    assert library.library_available("json")


def test_missing_module_is_unavailable() -> None:
    assert not library.library_available("phyai_no_such_module_xyz")


def test_installed_but_unimportable_module_is_unavailable(fake_import) -> None:
    """The case a module-spec check gets wrong.

    A broken native extension raises from the dynamic loader — an ``OSError``,
    not an ``ImportError`` — which is why the probe catches broadly.
    """

    fake_import(
        lambda target: (
            OSError("libcuda.so.1: cannot open shared object file")
            if target == "flashinfer"
            else None
        )
    )
    assert not library.library_available("flashinfer")


def test_a_module_raising_systemexit_is_also_unavailable(fake_import) -> None:
    """``BaseException``, not ``Exception``: a broken extension can exit."""

    fake_import(lambda target: SystemExit(1) if target == "brittle" else None)
    assert not library.library_available("brittle")


def test_success_is_memoized(fake_import) -> None:
    """At most one import attempt per library per process."""

    calls = fake_import(lambda _target: None)
    assert library.library_available("json")
    assert library.library_available("json")
    assert calls == ["json"]


def test_failure_is_memoized_too(fake_import) -> None:
    """Otherwise every selector cache miss pays another failed import."""

    calls = fake_import(lambda _target: ImportError("nope"))
    assert not library.library_available("absent")
    assert not library.library_available("absent")
    assert calls == ["absent"]


def test_library_facts_builds_lib_prefixed_paths() -> None:
    """These feed the ``lib.*`` facts that capability predicates read."""

    values = library.library_facts(frozenset({"json", "phyai_no_such_module_xyz"}))
    assert values == {"lib.json": True, "lib.phyai_no_such_module_xyz": False}


def test_flashinfer_probe_matches_reality_on_this_host() -> None:
    """Cross-check the probe against an actual import."""

    try:
        import flashinfer  # noqa: F401
    except BaseException:
        expected = False
    else:
        expected = True

    assert library.library_available("flashinfer") is expected
