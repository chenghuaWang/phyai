"""Hot-path cost of selection.

Selection sits inside every forward pass, so the cost of *reaching* a cached
answer matters as much as the cache itself. These are guard tests, not
benchmarks: they assert an order of magnitude, generously, so they fail on a
regression of the kind that already happened once rather than on a slow
machine.

The regression they guard against: ``Catalog.version`` and ``Policy.version``
are part of the selector's cache key, and both used to be recomputed on every
read — rendering every capability in the catalog and hashing the result. That
measured as 77% of the cost of an already-cached selection, making the
bookkeeping around a kernel more expensive than some kernels.
"""

from __future__ import annotations

import inspect
import time

import pytest

from phyai.kernel.facts import device, dtype
from phyai.kernel.opspec import Impl, OpSpec
from phyai.kernel.policy import Policy, policy_from_mapping
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


def elapsed_us(fn, n: int = 2000) -> float:
    fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - start) / n * 1e6


@pytest.fixture
def selector():
    return Selector(build_catalog(), device="nvidia:SM90")


def rmsnorm_query() -> KernelQuery:
    return KernelQuery.build(
        "rmsnorm",
        role="norm",
        dtype={"input": "bf16", "weight": "bf16", "output": "bf16"},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )


# --------------------------------------------------------------------------- #
# The fingerprints are computed once
# --------------------------------------------------------------------------- #


def test_catalog_version_is_memoized() -> None:
    """Reading it renders every capability, so it must not happen per call."""

    catalog = build_catalog()
    first = catalog.version
    # Cheap enough that 5000 reads are nothing. Before memoization each read
    # rendered 36 predicates and hashed the result.
    assert elapsed_us(lambda: catalog.version, n=5000) < 1.0
    assert catalog.version == first


def test_policy_version_is_computed_at_construction() -> None:
    policy = Policy()
    assert policy.version
    assert elapsed_us(lambda: policy.version, n=5000) < 1.0


def test_catalog_version_still_changes_when_a_capability_changes() -> None:
    """Memoization must not cost invalidation — a stale key would pin a
    selection to a kernel whose contract has since been tightened."""

    def build(sm: str) -> Catalog:
        catalog = Catalog()
        catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
        catalog.register(
            Impl(
                kernel_id="fast.toy",
                op="toy",
                when=all_of(device.arch.at_least(sm), dtype.input.is_set()),
                prepare=lambda facts, params: None,
            )
        )
        return catalog

    assert build("sm90").version != build("sm100").version


def test_catalog_version_is_invalidated_by_registration() -> None:
    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
    catalog.register(
        Impl(
            kernel_id="a.toy",
            op="toy",
            when=dtype.input.is_set(),
            prepare=lambda facts, params: None,
        )
    )
    before = catalog.version
    catalog.register(
        Impl(
            kernel_id="b.toy",
            op="toy",
            when=dtype.input == "bf16",
            prepare=lambda facts, params: None,
        )
    )
    assert catalog.version != before


def test_policy_version_changes_with_the_rules(selector) -> None:
    plain = Policy()
    steered = policy_from_mapping(
        {
            "rules": [
                {"id": "r", "match": {"op": "gemm"}, "prefer": ["torch.gemm.bf16"]}
            ]
        },
        selector.catalog,
    )
    assert plain.version != steered.version


# --------------------------------------------------------------------------- #
# Reaching the cache
# --------------------------------------------------------------------------- #


def test_a_warm_selection_is_not_dominated_by_key_construction(selector) -> None:
    """A cached selection must cost far less than building the query twice.

    The bound-call-site path is the one layers use and is roughly two orders of
    magnitude cheaper again; this bound covers the raw ``Selector.select``
    entry point, which still has to build a query to look up its own cache.
    """

    query = rmsnorm_query()
    selector.select(query)  # warm
    assert elapsed_us(lambda: selector.select(query)) < 200.0


def test_repeated_selection_returns_the_identical_object(selector) -> None:
    query = rmsnorm_query()
    assert selector.select(query) is selector.select(query)


# --------------------------------------------------------------------------- #
# CallSite — and the completeness of its memo key
# --------------------------------------------------------------------------- #
#
# This group is the correctness basis for the memo, not a nicety. Keying on too
# little returns a stale selection, which is the one failure mode in this
# system that is silent rather than loud.


@pytest.mark.parametrize("op", [spec.name for spec in build_catalog().ops()])
def test_memo_key_covers_every_fact_any_capability_reads(op: str) -> None:
    """The key is derived from the catalog, and this proves the derivation.

    If an implementation reads a fact the key omits, two calls differing only
    in that fact collide and the second gets the first's answer.
    """

    from phyai.kernel.call import CallSite

    catalog = build_catalog()
    site = CallSite(op)
    covered = set(site.key_paths())

    required: set[str] = set()
    for impl in catalog.impls(op):
        required |= impl.when.facts_used()
    # Library availability is a process constant, and the device is keyed
    # explicitly rather than by path.
    required = {path for path in required if not path.startswith("lib.")}

    missing = required - covered
    assert not missing, (
        f"{op}: memo key omits {sorted(missing)}, so two calls differing only "
        f"in those facts would collide"
    )


def test_memo_key_covers_the_facts_the_policy_reads(selector) -> None:
    """Policy is the other half of what decides a selection."""

    from phyai.kernel.call import CallSite
    from phyai.kernel.bootstrap import set_kernel_selector

    steered = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "by-model",
                    "match": {"op": "gemm", "model.family": "qwen"},
                    "prefer": ["torch.gemm.bf16"],
                }
            ]
        },
        selector.catalog,
    )
    set_kernel_selector(Selector(selector.catalog, steered, device="nvidia:SM90"))
    site = CallSite("gemm")
    assert "model.family" in site.key_paths()


def test_a_dtype_change_is_not_served_from_the_memo() -> None:
    """The end-to-end property the key exists for."""

    from phyai.kernel.call import CallSite

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    bf16 = site.select(device="cuda", dtype={"input": "bf16", "weight": "bf16"})
    fp32 = site.select(device="cuda", dtype={"input": "fp32", "weight": "fp32"})
    assert bf16.kernel_id != fp32.kernel_id


def test_a_device_change_is_not_served_from_the_memo() -> None:
    from phyai.kernel.call import CallSite

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    dtypes = {"input": "bf16", "weight": "bf16"}
    on_cuda = site.select(device="cuda", dtype=dtypes)
    on_cpu = site.select(device="cpu", dtype=dtypes)
    assert on_cuda.kernel_id != on_cpu.kernel_id


def test_capture_mode_is_not_served_from_the_eager_memo() -> None:
    """``graph_capture()`` flips the ambient mode; the key must notice."""

    from phyai.kernel.call import CallSite

    site = CallSite("attention_gdn", role="gdn", attrs={"layout": "paged"})
    dtypes = {
        "input": "bf16",
        "key": "bf16",
        "value": "bf16",
        "a": "bf16",
        "b": "bf16",
        "a_log": "fp32",
        "dt_bias": "bf16",
    }
    eager = site.select(device="nvidia:SM90", dtype=dtypes, mode="eager")
    captured = site.select(device="nvidia:SM90", dtype=dtypes, mode="capture")
    # FlashInfer's GDN backend is not capture-safe, so the captured call must
    # land on FLA instead.
    assert eager.kernel_id != captured.kernel_id


def test_the_memo_is_rebuilt_when_the_policy_changes(selector) -> None:
    """Otherwise a policy swap would be served stale answers forever."""

    from phyai.kernel.call import CallSite
    from phyai.kernel.bootstrap import set_kernel_selector

    set_kernel_selector(Selector(selector.catalog, Policy(), device="nvidia:SM90"))
    site = CallSite("gemm", role="mlp.down")
    call = dict(
        device="nvidia:SM100",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "bf16"},
        dims={"M": 512, "N": 4096, "K": 4096},
    )
    assert site.select(**call).kernel_id == "flashinfer.gemm.bf16"

    steered = policy_from_mapping(
        {
            "rules": [
                {"id": "r", "match": {"op": "gemm"}, "prefer": ["torch.gemm.bf16"]}
            ]
        },
        selector.catalog,
    )
    set_kernel_selector(Selector(selector.catalog, steered, device="nvidia:SM100"))
    assert site.select(**call).kernel_id == "torch.gemm.bf16"


def test_the_memo_is_rebuilt_when_the_model_context_changes(selector) -> None:
    """Same catalog, policy, and device — only the engine's model differs.

    A model-scoped policy rule flips the winner, so a stale memo keyed
    before the model context changed would keep serving the old kernel.
    """

    from phyai.kernel.call import CallSite
    from phyai.kernel.types import ModelContext
    from phyai.kernel.bootstrap import set_kernel_selector

    steered = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "by-model",
                    "match": {"op": "gemm", "model.family": "qwen"},
                    "prefer": ["torch.gemm.bf16"],
                }
            ]
        },
        selector.catalog,
    )
    call = dict(
        device="nvidia:SM90",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "bf16"},
        dims={"M": 512, "N": 4096, "K": 4096},
    )

    set_kernel_selector(Selector(selector.catalog, steered, device="nvidia:SM90"))
    site = CallSite("gemm", role="mlp.down")
    assert site.select(**call).kernel_id == "flashinfer.gemm.bf16"

    set_kernel_selector(
        Selector(
            selector.catalog,
            steered,
            model=ModelContext(family="qwen"),
            device="nvidia:SM90",
        )
    )
    assert site.select(**call).kernel_id == "torch.gemm.bf16"


def test_a_warm_call_site_is_an_order_of_magnitude_cheaper(selector) -> None:
    """The point of the whole exercise.

    Generous bound: measured at ~2 us against ~50 us for the bare entry point
    and ~340 us before the fingerprints were memoized.
    """

    from phyai.kernel.call import CallSite

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    dtypes = {"input": "bf16", "weight": "bf16"}
    assert elapsed_us(lambda: site.select(device="cuda", dtype=dtypes), n=5000) < 20.0


# --------------------------------------------------------------------------- #
# Pinning — taking the selector off the hot path entirely
# --------------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def unpinned():
    """No test leaks a pin into the next one."""

    yield
    from phyai.kernel.call import reset_verify_frozen, unfreeze_kernel_choices

    unfreeze_kernel_choices()
    reset_verify_frozen()


def warm_site(**overrides):
    from phyai.kernel.call import CallSite

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    site.select(device="cuda", dtype={"input": "bf16", "weight": "bf16"}, **overrides)
    return site


def test_freezing_removes_the_key_from_the_hot_path() -> None:
    """Measured ~1.0 us against ~2.3 us keyed and ~340 us originally.

    Not quite one attribute load: the ambient mode still has to be read, because
    a site is frozen *per mode* and returning another mode's answer would be the
    wrong kernel. That read plus one dict lookup is the whole remaining cost --
    no query, no fact mapping, no policy.
    """

    site = warm_site()
    site.freeze()
    dtypes = {"input": "bf16", "weight": "bf16"}
    assert elapsed_us(lambda: site.select(device="cuda", dtype=dtypes), n=20000) < 1.8


def test_a_frozen_site_returns_what_it_was_frozen_to() -> None:
    site = warm_site()
    expected = site.freeze()["eager"]
    # Facts that would normally change the answer are ignored once frozen --
    # that is the whole point, and why freezing has to refuse polymorphic sites.
    assert site.select(device="cpu", dtype={"input": "fp32"}) is expected


def test_freezing_refuses_a_polymorphic_site() -> None:
    """The safety property. Picking one of two answers would be silently wrong."""

    from phyai.kernel.call import FrozenChoiceError

    site = warm_site()
    site.select(device="cpu", dtype={"input": "bf16", "weight": "bf16"})
    with pytest.raises(FrozenChoiceError, match="polymorphic"):
        site.freeze()
    assert site.frozen_choices is None


def test_freezing_refuses_a_site_that_never_ran() -> None:
    from phyai.kernel.call import CallSite, FrozenChoiceError

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    with pytest.raises(FrozenChoiceError, match="has not run yet"):
        site.freeze()


def test_freeze_kernel_choices_counts_rather_than_raising() -> None:
    """One polymorphic site is normal and must not stop the rest going free."""

    from phyai.kernel.call import CallSite, freeze_kernel_choices

    mono = warm_site()
    poly = warm_site()
    poly.select(device="cpu", dtype={"input": "bf16", "weight": "bf16"})
    cold = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})

    tally = freeze_kernel_choices()
    assert tally["frozen"] >= 1 and tally["polymorphic"] >= 1 and tally["cold"] >= 1
    assert mono.frozen_choices is not None
    assert poly.frozen_choices is None
    assert cold.frozen_choices is None


def test_unfreeze_returns_to_keying() -> None:
    from phyai.kernel.call import unfreeze_kernel_choices

    site = warm_site()
    site.freeze()
    unfreeze_kernel_choices()
    assert site.frozen_choices is None
    on_cpu = site.select(device="cpu", dtype={"input": "bf16", "weight": "bf16"})
    assert on_cpu.kernel_id == "torch.rmsnorm"


def test_verify_frozen_catches_a_wrongly_frozen_site(monkeypatch) -> None:
    """The escape hatch: pay the key again, get told when the pin is a lie."""

    from phyai.kernel.call import FrozenChoiceError, reset_verify_frozen

    site = warm_site()
    site.freeze()
    monkeypatch.setenv("PHYAI_KERNEL_VERIFY_FROZEN", "1")
    reset_verify_frozen()
    dtypes = {"input": "bf16", "weight": "bf16"}
    # Same call as before: still fine.
    assert site.select(device="cuda", dtype=dtypes) is site.frozen_choices["eager"]
    with pytest.raises(FrozenChoiceError, match="not\\s+monomorphic"):
        site.select(device="cpu", dtype=dtypes)


def test_verify_frozen_is_read_once_per_process(monkeypatch) -> None:
    """It sits in the path pinning exists to empty, so it cannot be an env read."""

    from phyai.kernel.call import reset_verify_frozen, verify_frozen

    reset_verify_frozen()
    assert verify_frozen() is False
    monkeypatch.setenv("PHYAI_KERNEL_VERIFY_FROZEN", "1")
    assert verify_frozen() is False  # cached
    reset_verify_frozen()
    assert verify_frozen() is True


def test_call_sites_are_registered_weakly() -> None:
    """A discarded layer must not keep its bindings, or the registry leaks."""

    import gc

    from phyai.kernel.call import CallSite, get_call_sites

    site = CallSite("rmsnorm", role="throwaway")
    assert any(s is site for s in get_call_sites())
    # A count would be fragile: other tests' sites become collectable at
    # arbitrary points, so ``gc.collect()`` below can move it either way.
    del site
    gc.collect()
    assert not any(s.role == "throwaway" for s in get_call_sites())


# --------------------------------------------------------------------------- #
# The engine-level switch
# --------------------------------------------------------------------------- #


def test_freezing_is_off_by_default() -> None:
    """Freezing trusts warmup to have exercised every dtype the run will use.

    When that assumption is wrong the result is a silently wrong kernel, not an
    error, so the speed has to be asked for.
    """

    from phyai.engine_config import EngineConfig

    assert EngineConfig().runtime.freeze_kernel_choices is False


def test_the_engine_freezes_only_when_asked(monkeypatch) -> None:
    """Wiring test: the call happens after ``entry.setup()``, so warmup and
    graph capture have already exercised the call sites."""

    import phyai.engine as engine_module

    calls: list[bool] = []
    monkeypatch.setattr(
        engine_module, "freeze_kernel_choices", lambda: calls.append(True)
    )
    source = inspect.getsource(engine_module.Engine.__init__)
    assert "self.entry.setup(" in source
    assert "freeze_kernel_choices()" in source
    # The guard, not just the call.
    assert "self.config.runtime.freeze_kernel_choices" in source
    setup_at = source.index("self.entry.setup(")
    freeze_at = source.index("freeze_kernel_choices()")
    assert setup_at < freeze_at, "freezing must come after warmup"


def test_two_modes_naming_one_kernel_can_be_frozen() -> None:
    """The regression a real pi0.5 run found.

    Warmup runs eager and graph capture runs captured, so on a real model
    *every* call site has two memo entries. An earlier ``freeze()`` compared
    ``Selection`` object identity, and the selector caches per mode — so two
    entries naming the *same* kernel looked polymorphic and 447 of 486 sites
    were refused. Freezing per mode is what makes the feature do anything.
    """

    from phyai.kernel.call import CallSite

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    dtypes = {"input": "bf16", "weight": "bf16"}
    eager = site.select(device="nvidia:SM90", dtype=dtypes, mode="eager")
    captured = site.select(device="nvidia:SM90", dtype=dtypes, mode="capture")
    assert eager.kernel_id == captured.kernel_id
    assert eager is not captured  # distinct Selection objects, same choice

    frozen = site.freeze()
    assert set(frozen) == {"eager", "capture"}
    assert frozen["eager"] is eager and frozen["capture"] is captured


def test_a_mode_warmup_never_saw_is_not_frozen() -> None:
    """Falling through to keying is the safe answer — guessing from another
    mode's entry could hand a captured region a capture-unsafe kernel."""

    from phyai.kernel.call import CallSite

    site = CallSite("rmsnorm", role="norm", dims={"hidden": 4096})
    dtypes = {"input": "bf16", "weight": "bf16"}
    site.select(device="nvidia:SM90", dtype=dtypes, mode="eager")
    site.freeze()
    assert set(site.frozen_choices) == {"eager"}
    # Not frozen for capture, so this keys normally and gets the right answer.
    captured = site.select(device="nvidia:SM90", dtype=dtypes, mode="capture")
    assert captured.kernel_id


# --------------------------------------------------------------------------- #
# Selector lifecycle
# --------------------------------------------------------------------------- #


def test_the_scope_restores_nothing_installed() -> None:
    """The property that made a plain read-then-set wrong.

    ``get_kernel_selector`` builds a default as a side effect of *looking*, so a
    save/restore written on top of it restores that freshly built default rather
    than "nothing installed" — and the next caller inherits a catalog and a
    selection cache it never asked for.
    """

    from phyai.kernel import bootstrap

    bootstrap.reset_kernel_selector()
    with bootstrap.kernel_selector_scope():
        # Looking through the public entry point installs one...
        assert bootstrap.get_kernel_selector() is not None
    # ...and the scope still restores the original absence.
    assert bootstrap._selector is None


def test_the_scope_restores_a_previously_installed_selector(selector) -> None:
    from phyai.kernel import bootstrap

    bootstrap.set_kernel_selector(selector)
    replacement = Selector(build_catalog(), device="cpu")
    with bootstrap.kernel_selector_scope(replacement) as installed:
        assert installed is replacement
        assert bootstrap.get_kernel_selector() is replacement
    assert bootstrap.get_kernel_selector() is selector
    bootstrap.reset_kernel_selector()


def test_the_scope_restores_on_an_exception() -> None:
    """Impossible to half-use, unlike the read-then-set-twice it replaces."""

    from phyai.kernel import bootstrap

    bootstrap.reset_kernel_selector()
    original = Selector(build_catalog(), device="nvidia:SM90")
    bootstrap.set_kernel_selector(original)
    with pytest.raises(RuntimeError, match="boom"):
        with bootstrap.kernel_selector_scope(None):
            raise RuntimeError("boom")
    assert bootstrap.get_kernel_selector() is original
    bootstrap.reset_kernel_selector()
