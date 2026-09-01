"""Selector semantics.

These pin the behaviours the previous resolver established deliberately, and
which a rewrite could plausibly lose:

* an omitted device is filled from the engine profile, but an *explicit* one
  — including an explicit CPU — is never replaced;
* a losing candidate is never prepared;
* ``fallback: error`` does not quietly substitute a reference implementation;
* the selection cache is keyed on both the catalog and the policy fingerprint;
* capture mode excludes implementations that are not capture-safe.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import Facts, device as device_ns, dtype as dtype_ns, lib
from phyai.kernel.opspec import (
    Impl,
    OpSpec,
    Priority,
    any_float,
    fixed,
    matches_activation,
)
from phyai.kernel.policy import Policy, policy_from_mapping
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import NoKernelError, Selector
from phyai.kernel.types import KernelQuery


TOY = OpSpec(
    name="toy",
    dims=("M",),
    dtypes=("input",),
    attributes=(),
    signature="(x) -> Tensor",
)


def toy_catalog(*, prepared: list[str] | None = None) -> Catalog:
    """A two-row catalog that records which rows were actually prepared."""

    log = prepared if prepared is not None else []

    def make(name: str):
        def prepare(facts, params):
            log.append(name)
            return lambda value: (name, value)

        return prepare

    catalog = Catalog()
    catalog.register_op(TOY)
    catalog.register(
        Impl(
            kernel_id="fast.toy",
            op="toy",
            priority=Priority.OPTIMIZED + 2,
            when=dtype_ns.input == "bf16",
            prepare=make("fast"),
        )
    )
    catalog.register(
        Impl(
            kernel_id="ref.toy",
            op="toy",
            priority=Priority.REFERENCE,
            reference=True,
            when=dtype_ns.input.is_set(),
            prepare=make("ref"),
        )
    )
    return catalog


def toy_query(**kwargs):
    base = dict(dtype={"input": "bf16"}, shape={"M": 8})
    base.update(kwargs)
    return KernelQuery.build("toy", **base)


# --------------------------------------------------------------------------- #
# Device normalization
# --------------------------------------------------------------------------- #


def test_omitted_device_is_filled_from_the_engine_profile() -> None:
    selector = Selector(build_catalog(), device="nvidia:SM100")
    trace = selector.explain(
        KernelQuery.build(
            "rmsnorm",
            dtype={"input": "bf16", "weight": "bf16"},
            shape={"tokens": 8, "hidden": 4096},
            attrs={"variant": "rms"},
        )
    )
    assert trace.facts["device.vendor"] == "nvidia"
    assert trace.facts["device.arch"] == "sm100"


def test_explicit_cpu_device_survives_normalization() -> None:
    """A caller asking for a CPU reference must get one, on any host."""

    selector = Selector(build_catalog(), device="nvidia:SM100")
    trace = selector.explain(
        KernelQuery.build(
            "rmsnorm",
            device="cpu",
            dtype={"input": "fp32", "weight": "fp32"},
            shape={"tokens": 8, "hidden": 4096},
            attrs={"variant": "rms"},
        )
    )
    assert trace.facts["device.vendor"] == "cpu"
    assert trace.selected == "torch.rmsnorm"


def test_cpu_has_no_architecture_rather_than_a_zero() -> None:
    """The distinction that made "unknown device" readable as "too old".

    A CPU has no architecture in this sense, and saying so with ``None`` is what
    lets a capability fail with "unknown" instead of "too old". The helper this
    replaces returned ``0``.
    """

    selector = Selector(build_catalog(), device="cpu")
    trace = selector.explain(
        KernelQuery.build(
            "gemm",
            dtype={"input": "bf16", "output": "bf16"},
            quant={"format": "bf16"},
            shape={"M": 8, "N": 4096, "K": 4096},
        )
    )
    assert trace.facts["device.arch"] is None
    assert "device.sm" not in trace.facts


def test_the_architecture_fact_is_the_canonical_name() -> None:
    """One fact, lower-cased, whatever spelling the caller used.

    Structure -- the generation, the ordering -- is derived by the operators that
    need it, so it does not appear here as extra facts. There used to be three:
    ``device.arch``, ``device.sm`` and ``device.sm_major``.
    """

    for spelling, canonical in (
        ("SM90", "sm90"),
        ("SM100", "sm100"),
        ("sm120", "sm120"),
    ):
        selector = Selector(build_catalog(), device=f"nvidia:{spelling}")
        trace = selector.explain(
            KernelQuery.build(
                "rmsnorm",
                dtype={"input": "bf16", "weight": "bf16"},
                shape={"tokens": 8, "hidden": 4096},
                attrs={"variant": "rms"},
            )
        )
        assert trace.facts["device.arch"] == canonical
        assert "device.sm" not in trace.facts
        assert "device.sm_major" not in trace.facts


def test_amd_architecture_digits_are_not_read_as_an_sm_number() -> None:
    """``gfx942`` carries digits too. Reading them as a compute capability would
    make every NVIDIA-gated kernel look eligible on AMD."""

    selector = Selector(build_catalog(), device="amd:gfx942")
    trace = selector.explain(
        KernelQuery.build(
            "rmsnorm",
            dtype={"input": "bf16", "weight": "bf16"},
            shape={"tokens": 8, "hidden": 4096},
            attrs={"variant": "rms"},
        )
    )
    assert trace.facts["device.arch"] == "gfx942"
    # Structurally ineligible for an sm-gated row, and the reason says why --
    # better than the "device.sm is unknown" a NVIDIA-only numeric fact produced
    # for a device that is perfectly well known.
    failure = device_ns.arch.at_least("sm80").eval(
        Facts(values={"device.arch": "gfx942"})
    )
    assert failure is not None
    assert "not a 'sm' architecture" in failure.detail


# --------------------------------------------------------------------------- #
# Laziness
# --------------------------------------------------------------------------- #


def test_only_the_winning_candidate_is_prepared() -> None:
    prepared: list[str] = []
    selector = Selector(toy_catalog(prepared=prepared), device="nvidia:SM90")
    selection = selector.select(toy_query())
    assert selection.kernel_id == "fast.toy"
    assert prepared == ["fast"]


def test_a_strict_override_prepares_nothing_else() -> None:
    prepared: list[str] = []
    catalog = toy_catalog(prepared=prepared)
    policy = policy_from_mapping(
        {"overrides": [{"id": "o", "match": {"op": "toy"}, "use": "ref.toy"}]},
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "ref.toy"
    assert prepared == ["ref"]


def test_a_failing_candidate_falls_through_and_is_recorded() -> None:
    catalog = Catalog()
    catalog.register_op(TOY)
    catalog.register(
        Impl(
            kernel_id="broken.toy",
            op="toy",
            priority=Priority.OPTIMIZED + 2,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: (_ for _ in ()).throw(
                RuntimeError("no kernel image")
            ),
        )
    )
    catalog.register(
        Impl(
            kernel_id="ref.toy",
            op="toy",
            priority=Priority.REFERENCE,
            reference=True,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: lambda value: value,
        )
    )
    selector = Selector(catalog, device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "ref.toy"

    trace = selector.explain(toy_query())
    assert trace.candidates[0].kernel_id == "broken.toy"


def test_a_strict_override_that_cannot_prepare_is_an_error() -> None:
    catalog = Catalog()
    catalog.register_op(TOY)
    catalog.register(
        Impl(
            kernel_id="broken.toy",
            op="toy",
            priority=Priority.OPTIMIZED + 2,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: (_ for _ in ()).throw(
                RuntimeError("no kernel image")
            ),
        )
    )
    policy = policy_from_mapping(
        {"overrides": [{"id": "o", "match": {"op": "toy"}, "use": "broken.toy"}]},
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    with pytest.raises(NoKernelError, match="failed to prepare"):
        selector.select(toy_query())


def test_a_strict_override_naming_an_ineligible_kernel_is_an_error() -> None:
    catalog = toy_catalog()
    policy = policy_from_mapping(
        {"overrides": [{"id": "o", "match": {"op": "toy"}, "use": "fast.toy"}]},
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    with pytest.raises(NoKernelError, match="cannot handle this call"):
        selector.select(toy_query(dtype={"input": "fp32"}))


# --------------------------------------------------------------------------- #
# Ordering and fallback
# --------------------------------------------------------------------------- #


def test_a_rule_can_prefer_a_lower_priority_kernel() -> None:
    catalog = toy_catalog()
    policy = policy_from_mapping(
        {"rules": [{"id": "r", "match": {"op": "toy"}, "prefer": ["ref.toy"]}]},
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "ref.toy"


def test_reference_rows_are_appended_under_the_default_fallback() -> None:
    catalog = toy_catalog()
    policy = policy_from_mapping(
        {"rules": [{"id": "r", "match": {"op": "toy"}, "prefer": ["fast.toy"]}]},
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    # fp32 makes the preferred row ineligible; the reference still runs.
    assert selector.select(toy_query(dtype={"input": "fp32"})).kernel_id == "ref.toy"


def test_fallback_error_does_not_substitute_a_reference() -> None:
    catalog = toy_catalog()
    policy = policy_from_mapping(
        {
            "defaults": {"fallback": "error"},
            "rules": [{"id": "r", "match": {"op": "toy"}, "prefer": ["fast.toy"]}],
        },
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    with pytest.raises(NoKernelError, match="no kernel can handle"):
        selector.select(toy_query(dtype={"input": "fp32"}))


def test_an_unnamed_optimized_backend_is_never_substituted() -> None:
    """A rule is an ordered allow-list, not a licence to pick something else."""

    catalog = Catalog()
    catalog.register_op(TOY)
    for name, priority in (
        ("alpha.toy", Priority.OPTIMIZED + 2),
        ("beta.toy", Priority.OPTIMIZED + 1),
    ):
        catalog.register(
            Impl(
                kernel_id=name,
                op="toy",
                priority=priority,
                when=dtype_ns.input.is_set(),
                prepare=lambda facts, params, n=name: lambda value: n,
            )
        )
    policy = policy_from_mapping(
        {"rules": [{"id": "r", "match": {"op": "toy"}, "prefer": ["beta.toy"]}]},
        catalog,
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    # alpha is eligible and higher priority, but the rule did not name it and
    # it is not a reference row.
    assert selector.select(toy_query()).kernel_id == "beta.toy"


# --------------------------------------------------------------------------- #
# Capture mode
# --------------------------------------------------------------------------- #


def test_capture_mode_excludes_non_capture_safe_rows() -> None:
    catalog = Catalog()
    catalog.register_op(TOY)
    catalog.register(
        Impl(
            kernel_id="eageronly.toy",
            op="toy",
            priority=Priority.OPTIMIZED + 2,
            capture_safe=False,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: lambda value: value,
        )
    )
    catalog.register(
        Impl(
            kernel_id="ref.toy",
            op="toy",
            priority=Priority.REFERENCE,
            reference=True,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: lambda value: value,
        )
    )
    selector = Selector(catalog, device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "eageronly.toy"
    assert selector.select(toy_query(mode="capture")).kernel_id == "ref.toy"

    trace = selector.explain(toy_query(mode="capture"))
    reason = next(c.reason for c in trace.candidates if c.kernel_id == "eageronly.toy")
    assert "CUDA graph" in reason


def test_graph_aliases_normalize_to_capture() -> None:
    selector = Selector(toy_catalog(), device="nvidia:SM90")
    # ``graph_capturing`` is what phyai.parallel.state.Mode actually produces.
    for alias in ("capture", "graph_capturing", "graph-capturing"):
        assert selector.explain(toy_query(mode=alias)).facts["mode"] == "capture"


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #


def test_repeated_selection_is_cached() -> None:
    prepared: list[str] = []
    selector = Selector(toy_catalog(prepared=prepared), device="nvidia:SM90")
    first = selector.select(toy_query())
    second = selector.select(toy_query())
    assert first is second
    assert prepared == ["fast"]


def test_changing_the_policy_invalidates_the_choice() -> None:
    """The cache key carries both fingerprints, so this cannot go stale."""

    catalog = toy_catalog()
    plain = Selector(catalog, device="nvidia:SM90")
    assert plain.select(toy_query()).kernel_id == "fast.toy"

    policy = policy_from_mapping(
        {"rules": [{"id": "r", "match": {"op": "toy"}, "prefer": ["ref.toy"]}]},
        catalog,
    )
    steered = Selector(catalog, policy, device="nvidia:SM90")
    assert steered.select(toy_query()).kernel_id == "ref.toy"
    assert plain.policy.version != steered.policy.version


def test_clear_cache_forces_repreparation() -> None:
    prepared: list[str] = []
    selector = Selector(toy_catalog(prepared=prepared), device="nvidia:SM90")
    selector.select(toy_query())
    selector.clear_cache()
    selector.select(toy_query())
    assert prepared == ["fast", "fast"]


# --------------------------------------------------------------------------- #
# Errors and traces
# --------------------------------------------------------------------------- #


def test_no_candidate_error_carries_the_full_reasoning() -> None:
    catalog = Catalog()
    catalog.register_op(TOY)
    catalog.register(
        Impl(
            kernel_id="cuda.toy",
            op="toy",
            priority=Priority.OPTIMIZED + 2,
            when=device_ns.vendor == "nvidia",
            prepare=lambda facts, params: lambda value: value,
        )
    )
    selector = Selector(catalog, device="cpu")
    with pytest.raises(NoKernelError) as excinfo:
        selector.select(toy_query(device="cpu"))

    message = str(excinfo.value)
    assert "device.vendor == nvidia" in message
    assert "got 'cpu'" in message


def test_trace_records_the_contract_and_is_json_safe() -> None:
    selector = Selector(build_catalog(), device="nvidia:SM90")
    trace = selector.explain(
        KernelQuery.build(
            "rmsnorm",
            dtype={"input": "bf16", "weight": "bf16"},
            shape={"tokens": 8, "hidden": 4096},
            attrs={"variant": "rms"},
        )
    )
    payload = trace.as_dict()
    assert payload["op"] == "rmsnorm"
    assert payload["facts"]["shape.hidden"] == 4096
    entry = next(c for c in payload["candidates"] if c["id"] == "flashinfer.rmsnorm")
    assert "dtype.input == bf16" in entry["when"]
    trace.to_json()  # must not raise on frozensets or torch dtypes


def test_trace_reports_vacuous_optional_facts() -> None:
    """ "Matched" must never quietly mean "you did not tell us"."""

    catalog = Catalog()
    spec = OpSpec(
        name="toy2",
        dtypes=("input",),
        optional_dtypes=("residual",),
        signature="(x) -> Tensor",
    )
    catalog.register_op(spec)
    catalog.register(
        Impl(
            kernel_id="fast.toy2",
            op="toy2",
            when=all_of(dtype_ns.input == "bf16", dtype_ns.residual == "bf16"),
            prepare=lambda facts, params: lambda value: value,
        )
    )
    selector = Selector(catalog, device="nvidia:SM90")
    trace = selector.explain(KernelQuery.build("toy2", dtype={"input": "bf16"}))
    assert trace.selected == "fast.toy2"
    assert trace.candidates[0].skipped == ("dtype.residual == bf16",)


def test_unknown_operation_is_a_clear_error() -> None:
    selector = Selector(build_catalog(), device="nvidia:SM90")
    with pytest.raises(KeyError, match="unknown operation"):
        selector.select(KernelQuery.build("teleport"))


# --------------------------------------------------------------------------- #
# Autotune
# --------------------------------------------------------------------------- #


def test_autotune_picks_the_fastest_and_persists_it(tmp_path) -> None:
    catalog = toy_catalog()
    timings = {"fast.toy": 5.0, "ref.toy": 2.0}
    calls: list[str] = []

    def benchmark(impl, facts, selection):
        calls.append(impl.kernel_id)
        return timings[impl.kernel_id]

    cache = tmp_path / "autotune.json"
    selector = Selector(
        catalog,
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=benchmark,
        autotune_cache=cache,
    )
    # ref wins on measurement despite fast having the higher priority.
    assert selector.select(toy_query()).kernel_id == "ref.toy"
    assert set(calls) == {"fast.toy", "ref.toy"}
    assert cache.exists() and cache.read_text(encoding="utf-8").strip() != "{}"

    # A fresh selector reads the persisted choice and does not re-measure.
    calls.clear()
    reloaded = Selector(
        catalog,
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=benchmark,
        autotune_cache=cache,
    )
    assert reloaded.select(toy_query()).kernel_id == "ref.toy"
    assert calls == []


def test_autotune_is_skipped_under_graph_capture() -> None:
    """Measuring inside a capture would time graph construction, not the kernel."""

    calls: list[str] = []
    selector = Selector(
        toy_catalog(),
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=lambda impl, facts, selection: calls.append(impl.kernel_id) or 1.0,
    )
    assert selector.select(toy_query(mode="capture")).kernel_id == "fast.toy"
    assert calls == []


def test_a_benchmark_that_raises_does_not_break_selection() -> None:
    selector = Selector(
        toy_catalog(),
        Policy(profile="autotune"),
        device="nvidia:SM90",
        benchmark=lambda impl, facts, selection: 1 // 0,
    )
    assert selector.select(toy_query()).kernel_id == "fast.toy"


def test_a_corrupt_autotune_cache_is_ignored(tmp_path) -> None:
    cache = tmp_path / "autotune.json"
    cache.write_text("{not json", encoding="utf-8")
    selector = Selector(toy_catalog(), Policy(profile="autotune"), device="nvidia:SM90")
    assert selector.select(toy_query()).kernel_id == "fast.toy"


# --------------------------------------------------------------------------- #
# Construction-time parameter dtypes
# --------------------------------------------------------------------------- #


def test_param_dtypes_derives_fp32_gamma_for_layernorm() -> None:
    """Matches today's hardcoded value, but derived from the contracts."""

    selector = Selector(build_catalog(), device="nvidia:SM90")
    chosen = selector.param_dtypes(
        "layernorm", activation="bf16", known={"attrs.bias": True}
    )
    assert chosen == {"weight": "fp32", "bias": "fp32"}


def test_param_dtypes_follows_the_activation_for_rmsnorm() -> None:
    """FlashInfer reads gamma through the input type; fp32 would exclude it."""

    selector = Selector(build_catalog(), device="nvidia:SM90")
    assert selector.param_dtypes("rmsnorm", activation="bf16") == {"weight": "bf16"}


def test_param_dtypes_on_cpu_uses_the_reference_contract() -> None:
    selector = Selector(build_catalog(), device="cpu")
    assert selector.param_dtypes("layernorm", activation="fp32") == {
        "weight": "fp32",
        "bias": "fp32",
    }


def test_param_dtypes_reports_an_unsatisfiable_contract() -> None:
    catalog = Catalog()
    spec = OpSpec(name="toy3", dtypes=("input",), params=("weight",))
    catalog.register_op(spec)
    catalog.register(
        Impl(
            kernel_id="only.toy3",
            op="toy3",
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: None,
            params={"weight": fixed("fp8_e4m3")},
        )
    )
    selector = Selector(catalog, device="nvidia:SM90")
    with pytest.raises(ValueError, match="no dtype satisfies"):
        selector.param_dtypes("toy3", activation="bf16")


def test_param_dtypes_ignores_rows_the_device_rules_out() -> None:
    """On a CPU host, an NVIDIA-only fp32-gamma contract must not decide."""

    catalog = Catalog()
    spec = OpSpec(name="toy4", dtypes=("input",), params=("weight",))
    catalog.register_op(spec)
    catalog.register(
        Impl(
            kernel_id="cuda.toy4",
            op="toy4",
            priority=Priority.OPTIMIZED + 2,
            when=all_of(device_ns.vendor == "nvidia", dtype_ns.input.is_set()),
            prepare=lambda facts, params: None,
            params={"weight": fixed("fp32")},
        )
    )
    catalog.register(
        Impl(
            kernel_id="ref.toy4",
            op="toy4",
            priority=Priority.REFERENCE,
            reference=True,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: None,
            params={"weight": matches_activation()},
        )
    )
    on_cuda = Selector(catalog, device="nvidia:SM90")
    assert on_cuda.param_dtypes("toy4", activation="bf16") == {"weight": "fp32"}

    on_cpu = Selector(catalog, device="cpu")
    assert on_cpu.param_dtypes("toy4", activation="bf16") == {"weight": "bf16"}


def test_param_dtypes_respects_library_availability() -> None:
    """An unavailable backend's contract must not dictate an allocation."""

    catalog = Catalog()
    spec = OpSpec(name="toy5", dtypes=("input",), params=("weight",))
    catalog.register_op(spec)
    catalog.register(
        Impl(
            kernel_id="absent.toy5",
            op="toy5",
            priority=Priority.OPTIMIZED + 2,
            when=all_of(lib.has("phyai_no_such_module_xyz"), dtype_ns.input.is_set()),
            prepare=lambda facts, params: None,
            params={"weight": fixed("fp32")},
        )
    )
    catalog.register(
        Impl(
            kernel_id="ref.toy5",
            op="toy5",
            priority=Priority.REFERENCE,
            reference=True,
            when=dtype_ns.input.is_set(),
            prepare=lambda facts, params: None,
            params={"weight": any_float()},
        )
    )
    selector = Selector(catalog, device="nvidia:SM90")
    assert selector.param_dtypes("toy5", activation="bf16") == {"weight": "bf16"}
