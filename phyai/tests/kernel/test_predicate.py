"""Unit tests for the predicate algebra and the fact vocabulary.

The tests are grouped by the property being pinned rather than by class,
because the properties are what the rest of the system relies on:

* three-state fact semantics (provided / provided-as-None / not provided);
* optional-fact vacuity, and the lint that stops it being abused;
* ``render`` producing the text that lands in traces;
* ``restrict`` doing sound partial evaluation, which is what lets a layer
  pick a parameter dtype before it has a tensor;
* the parenthesization guard, because operator precedence genuinely bites.

The final group writes out the real capability expressions this algebra has
to replace. If those stop fitting, the design is wrong and these fail first.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import (
    Fact,
    FactKind,
    Facts,
    ParensError,
    attrs,
    device,
    dtype,
    lib,
    model,
    quant,
    shape,
)
from phyai.kernel.predicate import (
    FALSE,
    TRUE,
    Const,
    all_of,
    any_of,
    implies,
    none_of,
    predicate_from_literal,
    same,
)


def facts(values: dict[str, object], optional: set[str] | None = None) -> Facts:
    return Facts(values=values, optional=frozenset(optional or ()))


# --------------------------------------------------------------------------- #
# Three-state fact semantics
# --------------------------------------------------------------------------- #


def test_present_value_compares_normally() -> None:
    assert (shape.K >= 4096).eval(facts({"shape.K": 4096})) is None
    assert (shape.K >= 8192).eval(facts({"shape.K": 4096})) is not None


def test_none_value_is_unknown_not_zero() -> None:
    """A CPU device has no SM number; that must not read as "sm 0".

    The old numeric probe returned ``0`` for CPU and unparsable
    architectures, which turned "unknown device" into the much weaker
    "no backend is fast enough here".
    """

    failure = (shape.K >= 4096).eval(facts({"shape.K": None}))
    assert failure is not None
    assert failure.detail == "shape.K is unknown"
    assert "shape.K >= 4096" in str(failure)


def test_absent_required_fact_says_it_was_not_provided() -> None:
    failure = (shape.K >= 4096).eval(facts({}))
    assert failure is not None
    assert "not provided" in failure.detail
    # Distinct from the unknown case above — the caller forgot, versus the
    # device genuinely not having one.
    assert "unknown" not in failure.detail


def test_absent_optional_fact_is_vacuously_satisfied_and_reported() -> None:
    predicate = dtype.residual == "bf16"
    query = facts({"dtype.input": "bf16"}, optional={"dtype.residual"})

    assert predicate.eval(query) is None
    skipped = predicate.skipped(query)
    assert [item.predicate for item in skipped] == ["dtype.residual == bf16"]


def test_provided_optional_fact_is_still_enforced() -> None:
    predicate = dtype.residual == "bf16"
    query = facts({"dtype.residual": "fp32"}, optional={"dtype.residual"})
    assert predicate.eval(query) is not None


# --------------------------------------------------------------------------- #
# Normalization: one vocabulary, many spellings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spelling", ["bf16", "bfloat16", "torch.bfloat16"])
def test_dtype_facts_normalize_spellings(spelling: str) -> None:
    assert (dtype.input == spelling).eval(facts({"dtype.input": "bf16"})) is None


def test_fp8_shorthand_normalizes_to_e4m3_on_both_sides() -> None:
    assert (dtype.input == "fp8").eval(facts({"dtype.input": "fp8_e4m3"})) is None
    assert (dtype.input == "fp8_e4m3").eval(
        facts({"dtype.input": "float8_e4m3fn"})
    ) is None


@pytest.mark.parametrize("spelling", ["nvidia", "NVIDIA"])
def test_vendor_matching_is_case_insensitive(spelling: str) -> None:
    assert (device.vendor == spelling).eval(facts({"device.vendor": "nvidia"})) is None


@pytest.mark.parametrize("spelling", ["nv", "cuda", "rocm", "npu"])
def test_vendor_aliases_are_not_accepted(spelling: str) -> None:
    """One canonical spelling per vendor.

    ``cuda`` and friends used to normalize to ``nvidia`` through an alias
    table; now the only vendor names that exist are the ones the probe
    writes, so an alias simply fails to match — visible in the trace instead
    of silently rewritten.
    """

    failure = (device.vendor == spelling).eval(facts({"device.vendor": "nvidia"}))
    assert failure is not None


def test_torch_dtype_objects_normalize() -> None:
    torch = pytest.importorskip("torch")
    assert (dtype.input == "bf16").eval(facts({"dtype.input": torch.bfloat16})) is None


# --------------------------------------------------------------------------- #
# Operators
# --------------------------------------------------------------------------- #


def test_divisibility() -> None:
    assert (shape.K % 16 == 0).eval(facts({"shape.K": 4096})) is None
    failure = (shape.K % 16 == 0).eval(facts({"shape.K": 4095}))
    assert failure is not None
    assert failure.predicate == "shape.K % 16 == 0"


def test_membership_and_set_facts() -> None:
    assert (
        dtype.input.in_({"bf16", "fp16"}).eval(facts({"dtype.input": "fp16"})) is None
    )
    assert (
        model.tags.has("debug").eval(
            facts({"model.tags": frozenset({"debug", "reference"})})
        )
        is None
    )
    assert (
        model.tags.intersects({"debug", "reference"}).eval(
            facts({"model.tags": frozenset({"debug"})})
        )
        is None
    )


def test_a_pattern_literal_raises_instead_of_matching() -> None:
    with pytest.raises(ValueError, match="enumerate"):
        predicate_from_literal(device.arch, "gfx9*")


def test_between_is_inclusive() -> None:
    predicate = shape.kernel.between(1, 8)
    assert predicate.eval(facts({"shape.kernel": 1})) is None
    assert predicate.eval(facts({"shape.kernel": 8})) is None
    assert predicate.eval(facts({"shape.kernel": 9})) is not None


def test_ordering_on_unordered_fact_is_rejected_at_build_time() -> None:
    """Catch the mistake where it is written, not where it is evaluated."""

    with pytest.raises(TypeError, match="no ordering"):
        _ = device.arch >= "sm90"
    with pytest.raises(TypeError, match="int fact"):
        _ = quant.format % 16 == 0


def test_fact_to_fact_comparison() -> None:
    predicate = same(dtype.input, dtype.key, dtype.value)
    assert (
        predicate.eval(
            facts({"dtype.input": "bf16", "dtype.key": "bf16", "dtype.value": "bf16"})
        )
        is None
    )
    failure = predicate.eval(
        facts({"dtype.input": "bf16", "dtype.key": "fp16", "dtype.value": "bf16"})
    )
    assert failure is not None
    assert "'bf16' vs 'fp16'" in failure.detail


def test_is_none_selects_unquantized_calls() -> None:
    assert quant.format.is_none().eval(facts({"quant.format": None})) is None
    assert quant.format.is_none().eval(facts({"quant.format": "nvfp4"})) is not None


def test_implies_is_vacuous_when_condition_fails() -> None:
    predicate = implies(attrs.bias, dtype.bias == "fp32")
    # No bias: the beta dtype is unconstrained.
    assert predicate.eval(facts({"attrs.bias": False})) is None
    # Bias present and fp32: satisfied.
    assert predicate.eval(facts({"attrs.bias": True, "dtype.bias": "fp32"})) is None
    # Bias present but bf16: rejected.
    assert predicate.eval(facts({"attrs.bias": True, "dtype.bias": "bf16"})) is not None


def test_boolean_fact_usable_bare_in_combinators() -> None:
    predicate = all_of(attrs.causal, device.vendor == "nvidia")
    assert (
        predicate.eval(facts({"attrs.causal": True, "device.vendor": "nvidia"})) is None
    )
    assert (
        predicate.eval(facts({"attrs.causal": False, "device.vendor": "nvidia"}))
        is not None
    )


def test_none_of_rejects_any_match() -> None:
    predicate = none_of(device.vendor == "amd", device.vendor == "ascend")
    assert predicate.eval(facts({"device.vendor": "nvidia"})) is None
    assert predicate.eval(facts({"device.vendor": "amd"})) is not None


# --------------------------------------------------------------------------- #
# Failure reporting
# --------------------------------------------------------------------------- #


def test_conjunction_reports_the_first_failure_in_written_order() -> None:
    predicate = all_of(
        device.vendor == "nvidia", device.arch.at_least("sm100"), shape.K % 16 == 0
    )
    failure = predicate.eval(
        facts({"device.vendor": "nvidia", "device.arch": "sm90", "shape.K": 4095})
    )
    assert failure is not None
    # The reason names the specific leaf, not the whole expression.
    assert failure.predicate == "device.arch >= sm100"
    assert failure.detail == "got 'sm90'"


def test_disjunction_reports_every_alternative() -> None:
    predicate = any_of(quant.format == "bf16", quant.format == "fp16")
    failure = predicate.eval(facts({"quant.format": "nvfp4"}))
    assert failure is not None
    assert failure.detail.count("got") == 2


def test_failure_message_is_actionable() -> None:
    failure = (shape.K >= 8192).eval(facts({"shape.K": 4096}))
    assert str(failure) == "shape.K >= 8192 failed: got 4096"


# --------------------------------------------------------------------------- #
# render()
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("predicate", "text"),
    [
        (device.arch.at_least("sm100"), "device.arch >= sm100"),
        (quant.format == "nvfp4", "quant.format == nvfp4"),
        (shape.K % 16 == 0, "shape.K % 16 == 0"),
        (dtype.input.in_({"bf16"}), "dtype.input in {bf16}"),
        (model.tags.has("debug"), "'debug' in model.tags"),
        # Numbers sort numerically, not by rendered text: the catalog listing
        # exists to be read, and {128, 256, 64} reads as a mistake.
        (shape.head_dim.in_({128, 256, 64}), "shape.head_dim in {64, 128, 256}"),
        (shape.M.in_({10, 9}), "shape.M in {9, 10}"),
        (quant.format.is_none(), "quant.format is none"),
        (attrs.causal.as_predicate(), "attrs.causal"),
        (lib.has("flashinfer"), "lib.flashinfer"),
        (
            implies(attrs.bias, dtype.bias == "fp32"),
            "attrs.bias -> dtype.bias == fp32",
        ),
    ],
)
def test_render_is_readable_and_stable(predicate, text: str) -> None:
    assert predicate.render() == text


def test_render_parenthesizes_looser_operators_inside_a_conjunction() -> None:
    predicate = all_of(
        device.vendor == "nvidia",
        any_of(quant.format == "bf16", quant.format == "fp16"),
    )
    assert predicate.render() == (
        "device.vendor == nvidia & (quant.format == bf16 | quant.format == fp16)"
    )


def test_nested_conjunctions_flatten() -> None:
    predicate = all_of(
        all_of(device.arch.at_least("sm90"), shape.K % 16 == 0), attrs.causal
    )
    assert predicate.render() == (
        "device.arch >= sm90 & shape.K % 16 == 0 & attrs.causal"
    )


# --------------------------------------------------------------------------- #
# facts_used()
# --------------------------------------------------------------------------- #


def test_facts_used_collects_every_path() -> None:
    predicate = all_of(
        lib.has("flashinfer"),
        device.vendor == "nvidia",
        device.arch.at_least("sm100"),
        same(dtype.input, dtype.key),
        implies(attrs.bias, dtype.bias == "fp32"),
    )
    assert predicate.facts_used() == {
        "lib.flashinfer",
        "device.vendor",
        "device.arch",
        "dtype.input",
        "dtype.key",
        "attrs.bias",
        "dtype.bias",
    }


def test_facts_used_is_empty_for_constants() -> None:
    assert TRUE.facts_used() == frozenset()
    assert all_of().facts_used() == frozenset()


# --------------------------------------------------------------------------- #
# restrict() — partial evaluation
# --------------------------------------------------------------------------- #


def test_restrict_folds_satisfied_terms_away() -> None:
    predicate = all_of(
        device.vendor == "nvidia", device.arch.at_least("sm100"), shape.K % 16 == 0
    )
    reduced = predicate.restrict({"device.vendor": "nvidia", "device.arch": "sm100"})
    assert reduced.render() == "shape.K % 16 == 0"


def test_restrict_collapses_to_false_when_contradicted() -> None:
    predicate = all_of(device.vendor == "nvidia", device.arch.at_least("sm100"))
    assert predicate.restrict({"device.vendor": "cpu"}) is FALSE


def test_restrict_collapses_to_true_when_fully_satisfied() -> None:
    predicate = all_of(device.vendor == "nvidia", device.arch.at_least("sm90"))
    assert (
        predicate.restrict({"device.vendor": "nvidia", "device.arch": "sm90"}) is TRUE
    )


def test_restrict_leaves_unknown_facts_symbolic() -> None:
    """Absent from ``known`` means "not yet decided", not "not provided"."""

    predicate = dtype.weight == "fp32"
    assert predicate.restrict({"device.vendor": "nvidia"}) is predicate


def test_restrict_short_circuits_a_disjunction() -> None:
    predicate = any_of(quant.format == "bf16", quant.format == "nvfp4")
    assert predicate.restrict({"quant.format": "bf16"}) is TRUE
    assert predicate.restrict({"quant.format": "fp8_e4m3"}) is FALSE


def test_restrict_answers_the_parameter_dtype_question() -> None:
    """The motivating use: pick a dtype before any tensor exists.

    A layer allocating gamma cannot ask "what did the selector choose?" — no
    input has arrived. It can ask which candidate dtypes keep an
    implementation eligible, which is exactly a restrict-then-test scan.
    """

    flashinfer_layernorm = all_of(
        device.vendor == "nvidia", dtype.input == "bf16", dtype.weight == "fp32"
    )
    construction = {"device.vendor": "nvidia", "dtype.input": "bf16"}
    feasible = {
        candidate
        for candidate in ("bf16", "fp16", "fp32")
        if flashinfer_layernorm.restrict({**construction, "dtype.weight": candidate})
        is not FALSE
    }
    assert feasible == {"fp32"}


def test_restrict_of_implies_handles_both_branches() -> None:
    predicate = implies(attrs.bias, dtype.bias == "fp32")
    # No bias at all: the whole clause disappears.
    assert predicate.restrict({"attrs.bias": False}) is TRUE
    # Bias present: only the requirement remains.
    assert predicate.restrict({"attrs.bias": True}).render() == "dtype.bias == fp32"


# --------------------------------------------------------------------------- #
# The parenthesization guard
# --------------------------------------------------------------------------- #


def test_missing_parens_raises_instead_of_misbehaving() -> None:
    """``a >= 90 & b`` parses as ``a >= (90 & b)``; refuse rather than lie."""

    with pytest.raises(ParensError, match="precedence"):
        _ = bool(shape.K >= 4096)
    with pytest.raises(ParensError):
        _ = bool(device.vendor)
    with pytest.raises(ParensError):
        _ = bool(shape.K % 16)


def test_correctly_parenthesized_expression_composes() -> None:
    predicate = (shape.K >= 4096) & (quant.format == "bf16")
    assert predicate.eval(facts({"shape.K": 4096, "quant.format": "bf16"})) is None


# --------------------------------------------------------------------------- #
# YAML literal parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("fact", "literal", "text"),
    [
        (shape.K, ">=4096", "shape.K >= 4096"),
        (shape.K, "4096", "shape.K == 4096"),
        (shape.K, 4096, "shape.K == 4096"),
        (shape.K, "<4096", "shape.K < 4096"),
        (device.arch, ">=sm100", "device.arch >= sm100"),
        (device.arch, "sm90", "device.arch == sm90"),
        (shape.K, "%16", "shape.K % 16 == 0"),
        (shape.K, "%16==0", "shape.K % 16 == 0"),
        (quant.format, "nvfp4", "quant.format == nvfp4"),
        (dtype.input, ["bf16", "fp16"], "dtype.input in {bf16, fp16}"),
        (model.tags, ["debug"], "model.tags intersects {'debug'}"),
        (model.tags, "debug", "'debug' in model.tags"),
        (quant.format, None, "quant.format is none"),
    ],
)
def test_literals_parse_by_declared_kind(fact: Fact, literal, text: str) -> None:
    assert predicate_from_literal(fact, literal).render() == text


def test_comparator_strings_are_only_honoured_for_ordered_facts() -> None:
    """Nothing is inferred from the shape of a string.

    ``device.arch: "sm90"`` is a name, not a comparison, and the kind is what
    decides that -- ``ARCH`` is unordered, so the integer comparator grammar is
    never consulted for it.

    ``">=sm90"`` does work, but through the arch-specific grammar and only
    because the operand carries its series. A comparator with no series is an
    error rather than an equality test against the literal string ``">=90"``,
    which is what it used to compile to: a rule that could never fire and never
    said why.
    """

    assert predicate_from_literal(device.arch, "sm90").render() == "device.arch == sm90"
    assert (
        predicate_from_literal(device.arch, ">=sm90").render() == "device.arch >= sm90"
    )
    with pytest.raises(ValueError, match="no series prefix"):
        predicate_from_literal(device.arch, ">=90")


def test_list_on_a_scalar_fact_means_membership() -> None:
    predicate = predicate_from_literal(device.vendor, ["nvidia", "amd"])
    assert predicate.eval(facts({"device.vendor": "amd"})) is None
    assert predicate.eval(facts({"device.vendor": "cpu"})) is not None


# --------------------------------------------------------------------------- #
# Fact namespaces
# --------------------------------------------------------------------------- #


def test_closed_namespaces_reject_typos_immediately() -> None:
    with pytest.raises(AttributeError):
        _ = device.archh
    with pytest.raises(AttributeError):
        _ = quant.formatt


def test_op_scoped_namespaces_mint_facts_with_the_right_kind() -> None:
    # Dimensions are ordered ints; dtype roles normalize.
    assert shape.M.kind is FactKind.INT
    assert dtype.a_log.kind is FactKind.DTYPE
    assert attrs.layout.kind is FactKind.ANY
    assert shape.M.path == "shape.M"


def test_op_scoped_namespaces_are_case_sensitive() -> None:
    """``shape.K`` and ``shape.k`` are different paths, not aliases.

    Four spellings of every dimension used to be accepted, which is how the
    alias table grew to forty entries.
    """

    assert shape.K.path != shape.k.path


def test_op_scoped_namespace_ignores_dunder_lookups() -> None:
    with pytest.raises(AttributeError):
        _ = shape.__deepcopy__


def test_quant_field_escape_hatch() -> None:
    predicate = quant.field("scale_mode") == "ue8m0"
    assert predicate.render() == "quant.fields.scale_mode == ue8m0"
    assert predicate.facts_used() == {"quant.fields.scale_mode"}


def test_lib_availability_is_an_ordinary_fact() -> None:
    """So tests can fake it, and traces can explain it."""

    predicate = lib.has("flashinfer")
    assert predicate.facts_used() == {"lib.flashinfer"}
    assert predicate.eval(facts({"lib.flashinfer": True})) is None
    assert predicate.eval(facts({"lib.flashinfer": False})) is not None


# --------------------------------------------------------------------------- #
# The real capability contracts
# --------------------------------------------------------------------------- #
#
# These are the expressions that must replace the existing hand-written
# capability closures and ``can_handle`` methods. Writing them out here is the
# evidence that the algebra is expressive enough -- and that it needs no
# escape hatch to do it.

NVIDIA_FLASHINFER = lib.has("flashinfer") & (device.vendor == "nvidia")
HALF_FLOATS = frozenset({"bf16", "fp16"})


def test_torch_gemm_fp8_scaled_contract() -> None:
    """Replaces the ``fp8_`` branch of ``TorchKernel.can_handle``.

    Note the exact format match. The original tested
    ``spec_id.startswith("fp8_")``, so an e5m2 weight passed and was then fed
    to ``torch._scaled_mm`` under e4m3 assumptions.
    """

    contract = all_of(
        quant.format == "fp8_e4m3",
        quant.granularity.in_({"per_tensor", "per_channel"}),
        device.vendor == "nvidia",
        device.arch.at_least("sm89"),
        shape.K % 16 == 0,
        shape.N % 16 == 0,
    )
    base = {
        "quant.format": "fp8_e4m3",
        "quant.granularity": "per_channel",
        "device.vendor": "nvidia",
        "device.arch": "sm90",
        "shape.K": 4096,
        "shape.N": 4096,
    }
    assert contract.eval(facts(base)) is None
    assert contract.eval(facts({**base, "device.arch": "sm86"})) is not None
    assert contract.eval(facts({**base, "shape.K": 4095})) is not None

    # The e5m2 mis-route is now structurally impossible.
    rejected = contract.eval(facts({**base, "quant.format": "fp8_e5m2"}))
    assert rejected is not None
    assert rejected.predicate == "quant.format == fp8_e4m3"

    # And a CPU host reports "unknown", not "too old".
    cpu = contract.eval(facts({**base, "device.vendor": "cpu", "device.arch": None}))
    assert cpu is not None


def test_flashinfer_gemm_nvfp4_contract() -> None:
    """Replaces the nvfp4 branch of ``FlashInferKernel.can_handle``.

    This box is sm90, so the execution path cannot be exercised locally --
    but the *capability* can, against a synthetic device. That is a concrete
    advantage of expressing eligibility as data rather than as a closure.
    """

    contract = all_of(
        NVIDIA_FLASHINFER,
        quant.format == "nvfp4",
        quant.layout == "128x4",
        quant.block_k == 16,
        device.arch.at_least("sm100"),
        shape.K % 16 == 0,
    )
    base = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "device.arch": "sm100",
        "quant.format": "nvfp4",
        "quant.layout": "128x4",
        "quant.block_k": 16,
        "shape.K": 4096,
    }
    assert contract.eval(facts(base)) is None
    assert contract.eval(facts({**base, "device.arch": "sm90"})) is not None
    # The linear scale layout belongs to the torch reference row instead.
    assert contract.eval(facts({**base, "quant.layout": "linear"})) is not None
    # A host where flashinfer cannot be imported is correctly ineligible.
    assert contract.eval(facts({**base, "lib.flashinfer": False})) is not None


def test_flashinfer_rmsnorm_dtype_contract() -> None:
    """Residual is optional for plain rmsnorm, required for the fused form."""

    contract = all_of(
        NVIDIA_FLASHINFER,
        dtype.input == "bf16",
        dtype.weight == "bf16",
        dtype.residual == "bf16",
    )
    base = {"lib.flashinfer": True, "device.vendor": "nvidia"}
    plain = {**base, "dtype.input": "bf16", "dtype.weight": "bf16"}

    # rmsnorm: no residual supplied, and the op declares it optional.
    assert contract.eval(facts(plain, optional={"dtype.residual"})) is None
    # rmsnorm_add: residual required, and fp32 is rejected.
    fused = {**plain, "dtype.residual": "fp32"}
    assert contract.eval(facts(fused)) is not None
    assert contract.eval(facts({**plain, "dtype.residual": "bf16"})) is None
    # fp32 activations fall through to another implementation.
    assert contract.eval(facts({**plain, "dtype.input": "fp32"})) is not None


def test_flashinfer_layernorm_conditional_bias_contract() -> None:
    """The beta dtype is only constrained when there is a beta."""

    contract = all_of(
        NVIDIA_FLASHINFER,
        dtype.input == "bf16",
        dtype.weight == "fp32",
        implies(attrs.bias, dtype.bias == "fp32"),
    )
    base = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "dtype.input": "bf16",
        "dtype.weight": "fp32",
    }
    assert contract.eval(facts({**base, "attrs.bias": False})) is None
    assert (
        contract.eval(facts({**base, "attrs.bias": True, "dtype.bias": "fp32"})) is None
    )
    assert (
        contract.eval(facts({**base, "attrs.bias": True, "dtype.bias": "bf16"}))
        is not None
    )


def test_flashinfer_gdn_seven_role_contract() -> None:
    """Replaces a 24-line runtime-``raise`` dtype check plus a wrong ``min_sm``.

    Two corrections are baked in. The backend gates on
    ``major in (9, 10)``, but the descriptor only required ``sm >= 90`` -- so
    an sm120 device passed selection and then raised inside the backend, where
    a raise can never drive a fallback. And ``q``/``k``/``v`` having to agree
    was previously inexpressible.
    """

    contract = all_of(
        NVIDIA_FLASHINFER,
        device.arch.family_in({"sm9", "sm10"}),
        dtype.input.in_(HALF_FLOATS),
        dtype.key.in_(HALF_FLOATS),
        dtype.value.in_(HALF_FLOATS),
        dtype.a.in_(HALF_FLOATS),
        dtype.b.in_(HALF_FLOATS),
        dtype.a_log == "fp32",
        dtype.dt_bias.in_({"bf16", "fp32"}),
        same(dtype.input, dtype.key, dtype.value),
    )
    base = {
        "lib.flashinfer": True,
        "device.vendor": "nvidia",
        "device.arch": "sm90",
        "dtype.input": "bf16",
        "dtype.key": "bf16",
        "dtype.value": "bf16",
        "dtype.a": "bf16",
        "dtype.b": "bf16",
        "dtype.a_log": "fp32",
        "dtype.dt_bias": "bf16",
    }
    assert contract.eval(facts(base)) is None
    assert contract.eval(facts({**base, "device.arch": "sm100"})) is None

    # sm120 is now rejected at selection instead of raising during forward.
    sm120 = contract.eval(facts({**base, "device.arch": "sm120"}))
    assert sm120 is not None
    assert sm120.predicate == "device.arch family in {sm9, sm10}"

    # Mismatched q/k/v dtypes, previously only a runtime raise.
    assert contract.eval(facts({**base, "dtype.value": "fp16"})) is not None
    assert contract.eval(facts({**base, "dtype.a_log": "bf16"})) is not None


def test_sdpa_layout_contract_is_positive_not_negative() -> None:
    """Stated as "padded only", which also correctly excludes ``paged``.

    The old form rejected ``ragged`` by name, so a ``paged`` layout -- or an
    omitted one -- silently passed.
    """

    contract = attrs.layout == "padded"
    assert contract.eval(facts({"attrs.layout": "padded"})) is None
    assert contract.eval(facts({"attrs.layout": "ragged"})) is not None
    assert contract.eval(facts({"attrs.layout": "paged"})) is not None
    # Omitting it is an error now, rather than landing here by default.
    assert contract.eval(facts({})) is not None


def test_remaining_real_contracts_need_no_escape_hatch() -> None:
    """The rest of the built-in eligibility rules, for completeness."""

    contracts = {
        "flashinfer prefill head_dim": shape.head_dim.in_({64, 128, 256}),
        "rmsnorm_silu_mul hidden": shape.hidden <= 8192,
        "causal_conv kernel": shape.kernel.between(1, 8),
        "rope full rotary": shape.rotary_dim == shape.head_dim,
        "debug tag": model.tags.has("debug"),
    }
    for predicate in contracts.values():
        assert not isinstance(predicate, Const)
        assert predicate.render()
        assert predicate.facts_used()
