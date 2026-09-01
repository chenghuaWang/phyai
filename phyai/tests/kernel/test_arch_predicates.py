"""``at_least`` / ``family_in``, and the YAML spellings that reach them.

The whole point of a named operator here is that architecture ordering has two
plausible-but-wrong readings — lexicographic and cross-vendor — and a bare
``>=`` picks one of them silently. These tests pin the reading, pin the failure
*messages* (they are the reason a tree is used instead of a callback), and pin
the two literal grammars refusing each other's shape.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import Facts, device, quant, shape
from phyai.kernel.predicate import (
    FALSE,
    TRUE,
    ArchAtLeast,
    ArchFamilyIn,
    is_false,
    predicate_from_literal,
)


def facts_for(arch: object) -> Facts:
    return Facts(values={"device.arch": arch})


def verdict(predicate, arch: object) -> str:
    failure = predicate.eval(facts_for(arch))
    return "ok" if failure is None else failure.detail


# --------------------------------------------------------------------------- #
# at_least
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch, ok",
    [
        ("sm70", False),
        ("sm80", False),
        ("sm86", False),
        ("sm89", True),
        ("sm90", True),
        ("sm100", True),
        ("sm120", True),
    ],
)
def test_at_least_matches_the_numeric_gate_it_replaces(arch, ok) -> None:
    """``device.sm >= 89`` gated five fp8/nvfp4 rows; this must admit the same
    set, or those kernels quietly stop being eligible."""

    assert (verdict(device.arch.at_least("sm89"), arch) == "ok") is ok


def test_at_least_says_which_of_two_reasons_applied() -> None:
    """Too old and wrong-vendor are different problems, and the trace is the
    only place a user finds out which."""

    predicate = device.arch.at_least("sm89")
    assert verdict(predicate, "sm80") == "got 'sm80'"
    assert "not a 'sm' architecture" in verdict(predicate, "gfx942")
    assert "not a 'sm' architecture" in verdict(predicate, "ascend910b")


def test_a_known_absent_architecture_reads_as_unknown() -> None:
    """Not as the string ``'none'``, and not as "too old" — that conflation is
    what made the old numeric fact report "no backend available" on CPU."""

    assert verdict(device.arch.at_least("sm89"), None) == "device.arch is unknown"


def test_a_never_provided_architecture_says_so() -> None:
    failure = device.arch.at_least("sm89").eval(Facts(values={}))
    assert failure is not None
    assert failure.detail == "device.arch was not provided"


def test_at_least_renders_readably() -> None:
    assert device.arch.at_least("sm100").render() == "device.arch >= sm100"


def test_bare_comparison_on_an_arch_fact_still_raises() -> None:
    """``ARCH`` stays an unordered kind on purpose: the guard is what forces the
    named operator, and the named operator is what carries the series."""

    with pytest.raises(TypeError, match="no ordering"):
        device.arch >= "sm90"


@pytest.mark.parametrize("fact", [shape.K, quant.format, device.vendor])
def test_at_least_refuses_a_non_arch_fact(fact) -> None:
    with pytest.raises(TypeError, match="needs an arch fact"):
        ArchAtLeast(fact, "sm90")


def test_at_least_validates_its_bound_at_construction() -> None:
    """Import time, not selection time. A bound that cannot mean anything must
    never get as far as a query."""

    with pytest.raises(ValueError, match="did you mean 'sm100'"):
        device.arch.at_least("sm10")
    with pytest.raises(ValueError, match="not in an ordered series"):
        device.arch.at_least("gfx942")


# --------------------------------------------------------------------------- #
# family_in
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch, ok",
    [
        ("sm90", True),
        ("sm90a", True),
        ("sm100", True),
        ("sm120", False),
        ("sm89", False),
    ],
)
def test_family_in_is_discrete_not_a_floor(arch, ok) -> None:
    """FlashInfer's GDN kernels ship a Hopper path and a Blackwell path and
    nothing else, so ">= sm90" would wrongly claim sm120."""

    assert (verdict(device.arch.family_in({"sm9", "sm10"}), arch) == "ok") is ok


def test_family_in_reports_the_generation_it_saw() -> None:
    detail = verdict(device.arch.family_in({"sm9", "sm10"}), "sm120")
    assert detail == "got 'sm120' (generation sm12)"
    assert verdict(device.arch.family_in({"sm9"}), "ascend910b") == (
        "got 'ascend910b', which has no generation"
    )


def test_family_in_renders_generations_in_version_order() -> None:
    """``{sm10, sm9}`` reads like a mistake, so sort by (series, major) rather
    than by rendered text — same reason ``_render_set`` sorts numbers."""

    rendered = device.arch.family_in({"sm10", "sm9"}).render()
    assert rendered == "device.arch family in {sm9, sm10}"


def test_family_in_refuses_a_full_architecture_name() -> None:
    """The sharpest edge in this vocabulary: ``at_least`` takes ``sm90`` and
    ``family_in`` takes ``sm9``. Each must reject the other's shape rather than
    silently reinterpret it."""

    with pytest.raises(ValueError, match="did you mean"):
        device.arch.family_in({"sm90"})


def test_family_in_needs_at_least_one_generation() -> None:
    with pytest.raises(ValueError, match="at least one generation"):
        ArchFamilyIn(device.arch, [])


def test_family_in_refuses_a_non_arch_fact() -> None:
    with pytest.raises(TypeError, match="an arch fact"):
        ArchFamilyIn(quant.format, {"sm9"})


# --------------------------------------------------------------------------- #
# facts_used and restrict, inherited from Leaf
# --------------------------------------------------------------------------- #


def test_both_nodes_report_the_fact_they_read() -> None:
    """``CallSite``'s memo key is derived from this, so an omission here becomes
    a stale selection."""

    assert device.arch.at_least("sm90").facts_used() == frozenset({"device.arch"})
    assert device.arch.family_in({"sm9"}).facts_used() == frozenset({"device.arch"})


def test_restrict_folds_both_nodes_to_a_constant() -> None:
    """This is what lets ``param_dtypes`` discard a row this host cannot run
    before asking it about parameter dtypes."""

    predicate = device.arch.at_least("sm100")
    # A folded-true predicate ignores facts entirely; that is the observable
    # difference from the unfolded one, which needs ``device.arch`` provided.
    assert predicate.restrict({"device.arch": "sm120"}).eval(Facts()) is None
    assert is_false(predicate.restrict({"device.arch": "sm90"}))
    assert is_false(predicate.restrict({"device.arch": "gfx942"}))
    assert predicate.restrict({"device.vendor": "nvidia"}) is predicate


def test_is_false_catches_what_an_identity_test_against_FALSE_misses() -> None:
    """``Leaf.restrict`` builds a fresh ``Const``, so ``is not FALSE`` silently
    kept single-leaf capabilities alive."""

    folded = device.arch.at_least("sm100").restrict({"device.arch": "sm90"})
    assert folded is not FALSE
    assert is_false(folded)
    assert is_false(FALSE) and not is_false(TRUE)


# --------------------------------------------------------------------------- #
# The YAML surface
# --------------------------------------------------------------------------- #


def test_yaml_comparator_compiles_to_at_least() -> None:
    predicate = predicate_from_literal(device.arch, ">=sm100")
    assert predicate.render() == "device.arch >= sm100"
    assert verdict(predicate, "sm120") == "ok"


def test_yaml_generation_is_spelled_as_an_enumeration() -> None:
    """``family_in`` gets no YAML spelling on purpose: giving ``sm9`` a second
    meaning would make the bare scalar ``device.arch: sm9`` ambiguous between a
    name and a generation. A generation is a short fixed list of majors, so a
    YAML rule enumerates the archs it was validated on."""

    predicate = predicate_from_literal(device.arch, ["sm90", "sm100"])
    assert verdict(predicate, "sm90") == "ok"
    assert verdict(predicate, "sm100") == "ok"
    assert verdict(predicate, "sm120") != "ok"
    assert verdict(predicate, "gfx942") != "ok"


def test_a_pattern_literal_is_an_error_not_a_dead_rule() -> None:
    """``sm9*`` compiled as equality would give a rule that never fires and
    never explains itself — the failure mode this whole system exists to
    remove. Compiled as a glob it hides which devices the rule was validated
    on. So a pattern raises, and the error names the enumeration to write."""

    with pytest.raises(ValueError, match="enumerate"):
        predicate_from_literal(device.arch, ["sm9*"])
    with pytest.raises(ValueError, match="enumerate"):
        predicate_from_literal(device.arch, "gfx9*")


def test_a_pattern_hidden_in_a_mixed_list_is_still_an_error() -> None:
    with pytest.raises(ValueError, match="gfx9"):
        predicate_from_literal(device.arch, ["sm90", "gfx9*"])


def test_a_plain_list_still_compiles_to_membership() -> None:
    predicate = predicate_from_literal(quant.format, ["bf16", "fp8_e4m3"])
    assert predicate.render() == "quant.format in {bf16, fp8_e4m3}"


def test_a_bare_scalar_is_an_exact_name() -> None:
    predicate = predicate_from_literal(device.arch, "sm90")
    assert predicate.render() == "device.arch == sm90"
    assert verdict(predicate, "sm90") == "ok"
    assert verdict(predicate, "sm100") != "ok"


@pytest.mark.parametrize("literal", [">sm90", "<sm90", "<=sm90", "> sm90"])
def test_only_greater_or_equal_is_supported(literal) -> None:
    """A ``<`` rule ("this kernel is for old GPUs") is nearly always a latent bug
    that excludes future hardware, and nothing in the tree needs one."""

    with pytest.raises(ValueError, match="only '>=' is"):
        predicate_from_literal(device.arch, literal)


def test_a_comparator_without_a_series_is_an_error() -> None:
    """It used to compile to an equality test against the literal string
    ``">=90"``: a rule that could never match and never said so."""

    with pytest.raises(ValueError, match="no series prefix"):
        predicate_from_literal(device.arch, ">=90")


def test_a_bare_number_is_an_error_with_the_reason() -> None:
    """A bare number names no series, so it can never match an architecture."""

    with pytest.raises(ValueError, match="no series prefix"):
        predicate_from_literal(device.arch, 100)


# --------------------------------------------------------------------------- #
# The facts that used to exist
# --------------------------------------------------------------------------- #


def test_the_removed_numeric_facts_are_gone() -> None:
    """One architecture fact, not three.

    ``device.sm`` and ``device.sm_major`` put one vendor's naming scheme into the
    fact schema — the next accelerator would have needed ``device.gfx`` beside
    them.
    """

    from phyai.kernel.facts import GLOBAL_FACT_PATHS, DeviceNamespace

    assert "device.sm" not in GLOBAL_FACT_PATHS
    assert "device.sm_major" not in GLOBAL_FACT_PATHS
    assert not hasattr(DeviceNamespace, "sm")
    assert not hasattr(DeviceNamespace, "sm_major")
    assert "device.arch" in GLOBAL_FACT_PATHS


def test_device_facts_covers_every_device_path_in_the_schema() -> None:
    """The structural fix behind ``param_dtypes``.

    A ``device.*`` path missing from ``device_facts`` stays symbolic through
    ``restrict``, so a row this host cannot run survives the filter and votes on
    what dtype a parameter is allocated in — silently. ``device.arch`` and
    ``device.features`` were both missing that way.
    """

    from phyai.kernel.facts import GLOBAL_FACT_PATHS, device_facts
    from phyai.kernel.types import DeviceProfile

    declared = {p for p in GLOBAL_FACT_PATHS if p.startswith("device.")}
    supplied = set(device_facts(DeviceProfile(vendor="nvidia", arch="sm90")))
    assert declared == supplied


def test_device_facts_reports_absence_as_none() -> None:
    from phyai.kernel.facts import device_facts
    from phyai.kernel.types import DeviceProfile

    values = device_facts(DeviceProfile(vendor="cpu"))
    assert values["device.arch"] is None
    assert values["device.vendor"] == "cpu"
