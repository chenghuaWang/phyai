"""The ``Arch`` value type: parsing, ordering, and what it refuses.

Architecture used to be three facts — a name, a compute-capability number, and
its major — because a string is not comparable and the number was the only way
to write ``sm >= 89``. That worked for NVIDIA and put the vendor's naming scheme
into the fact schema, so the next accelerator needed ``device.gfx`` next to
``device.sm``. These tests pin down the single-fact replacement, and in
particular pin the two things it must *not* do: order across vendors, and accept
a bound that cannot mean anything.
"""

from __future__ import annotations

import pytest

from phyai.kernel.types import (
    ARCH_SERIES_GRAMMAR,
    ORDERED_ARCH_SERIES,
    Arch,
    DeviceProfile,
    arch_at_least,
)


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, series, major, minor",
    [
        # sm_XY is major*10 + minor, NVIDIA's own encoding.
        ("sm70", "sm", 7, 0),
        ("sm75", "sm", 7, 5),
        ("sm80", "sm", 8, 0),
        ("sm86", "sm", 8, 6),
        ("sm89", "sm", 8, 9),
        ("sm90", "sm", 9, 0),
        ("sm100", "sm", 10, 0),
        ("sm120", "sm", 12, 0),
        # sm90a is sm90 plus arch-specific instructions, same capability.
        ("sm90a", "sm", 9, 0),
        # AMD writes major.minor.stepping, one position each for the last two,
        # so the major can be two digits and the stepping can be a letter.
        ("gfx908", "gfx", 9, 0),
        ("gfx90a", "gfx", 9, 0),
        ("gfx942", "gfx", 9, 4),
        ("gfx1030", "gfx", 10, 3),
        ("gfx1100", "gfx", 11, 0),
    ],
)
def test_parse_splits_each_series_by_its_own_grammar(
    name, series, major, minor
) -> None:
    arch = Arch.parse(name)
    assert (arch.name, arch.series, arch.major, arch.minor) == (
        name,
        series,
        major,
        minor,
    )


@pytest.mark.parametrize("name", ["ascend910b", "h100", "mi300", "cpu", "", "sm"])
def test_parse_never_raises_and_leaves_unstructured_names_unstructured(name) -> None:
    """Parsing runs during evaluation, so it must not turn an unknown
    accelerator into a crash."""

    arch = Arch.parse(name)
    assert arch.name == name
    assert arch.major is None and arch.family is None and not arch.ordered


def test_parse_is_case_insensitive() -> None:
    assert Arch.parse("SM90") == Arch.parse("sm90")


# --------------------------------------------------------------------------- #
# Ordering — inside a series only
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "arch, expected",
    [("sm80", False), ("sm86", False), ("sm89", True), ("sm90", True), ("sm100", True)],
)
def test_at_least_reproduces_the_numeric_comparison_it_replaces(arch, expected) -> None:
    """``device.sm >= 89`` admitted sm89 through sm120; so must this.

    This is the equivalence that matters: the fp8 rows were gated on ``>= 89``,
    and a family-scoped comparison (sm89 is family sm8, sm90 is family sm9) would
    have silently disqualified every Hopper and Blackwell host.
    """

    assert Arch.parse(arch).at_least(Arch.parse_bound("sm89")) is expected


def test_a_two_digit_major_orders_above_a_one_digit_one() -> None:
    """sm100 > sm90 even though "100" < "90" as text.

    Collapsing ``(major, minor)`` into one integer happened to give this too,
    which is why the old code looked right; comparing the pair says why.
    """

    assert Arch.parse("sm100").at_least(Arch.parse_bound("sm90"))
    assert not Arch.parse("sm90").at_least(Arch.parse_bound("sm100"))


def test_ordering_across_series_is_false_not_a_number_comparison() -> None:
    """gfx942 is not "before sm90" — the question has no answer.

    False is the right eligibility answer (an sm-gated kernel must not run on
    AMD), and it now comes from the series mismatch rather than from a vendor
    conjunct that someone could forget to write.
    """

    assert not Arch.parse("gfx942").at_least(Arch.parse_bound("sm90"))
    assert not Arch.parse("ascend910b").at_least(Arch.parse_bound("sm70"))


def test_only_nvidia_is_ordered() -> None:
    """gfx1100 (RDNA3) is numerically above gfx942 (CDNA3) and supports a
    disjoint set of matrix instructions, so ordering gfx would be confidently
    wrong."""

    assert ORDERED_ARCH_SERIES == frozenset({"sm"})
    assert Arch.parse("gfx1100").ordered is False
    assert Arch.parse("gfx1100").at_least(Arch.parse("gfx942")) is False


# --------------------------------------------------------------------------- #
# Generations
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name, family",
    [
        ("sm90", "sm9"),
        ("sm100", "sm10"),
        ("sm120", "sm12"),
        ("sm89", "sm8"),
        ("gfx942", "gfx9"),
        ("gfx1100", "gfx11"),
    ],
)
def test_family_is_series_plus_major(name, family) -> None:
    assert Arch.parse(name).family == family


# --------------------------------------------------------------------------- #
# Bounds are validated where they are written, not where they are evaluated
# --------------------------------------------------------------------------- #


def test_a_bare_number_is_rejected_with_the_reason() -> None:
    """A bare number carries no series, so it can never be an ordered bound."""

    with pytest.raises(ValueError, match="no series prefix"):
        Arch.parse_bound("100")


def test_an_implausible_major_is_rejected() -> None:
    """``sm10`` parses as major 1 and would hold on every machine ever made —
    a typo that silently admits everything."""

    with pytest.raises(ValueError, match="did you mean 'sm100'"):
        Arch.parse_bound("sm10")


def test_a_generation_name_cannot_be_a_bound() -> None:
    """The other direction of the grammar split: ``at_least`` refuses ``sm9``.

    A single digit used to parse as major 9 *minor 9*, silently turning
    ``at_least("sm9")`` into ">= sm99" — sm90 through sm98 failed it.
    """

    with pytest.raises(ValueError, match="names a generation"):
        Arch.parse_bound("sm9")


@pytest.mark.parametrize("bound", ["gfx942", "ascend910b", "h100", "cpu"])
def test_an_unordered_series_cannot_be_a_bound(bound) -> None:
    with pytest.raises(ValueError, match="not in an ordered series"):
        Arch.parse_bound(bound)


def test_an_empty_bound_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        Arch.parse_bound("")


@pytest.mark.parametrize("family", ["sm9", "sm10", "sm12", "gfx9", "gfx11"])
def test_parse_family_accepts_a_generation(family) -> None:
    assert Arch.parse_family(family) == family


def test_parse_family_rejects_a_full_architecture_name() -> None:
    """The sharpest edge in the design: ``at_least`` takes ``sm90`` and
    ``family_in`` takes ``sm9``. Each grammar must refuse the other's shape."""

    with pytest.raises(ValueError, match="did you mean"):
        Arch.parse_family("sm90")


def test_parse_family_rejects_an_unknown_series() -> None:
    with pytest.raises(ValueError, match="unknown series"):
        Arch.parse_family("zz9")


def test_parse_family_rejects_a_suffixed_name() -> None:
    with pytest.raises(ValueError, match="series followed by a major"):
        Arch.parse_family("sm90a")


# --------------------------------------------------------------------------- #
# The profile helper
# --------------------------------------------------------------------------- #


def test_arch_parts_is_derived_not_stored() -> None:
    """Storing major/minor next to the name would be a second source of truth,
    and would make ``DeviceProfile(arch="sm90", major=12)`` constructible."""

    profile = DeviceProfile(vendor="nvidia", arch="sm90")
    assert profile.arch_parts == Arch.parse("sm90")
    assert not hasattr(profile, "arch_major")


def test_arch_parts_is_none_without_an_architecture() -> None:
    assert DeviceProfile(vendor="cpu").arch_parts is None


@pytest.mark.parametrize(
    "vendor, arch, bound, expected",
    [
        ("nvidia", "sm90", "sm100", False),
        ("nvidia", "sm100", "sm100", True),
        ("nvidia", "sm120", "sm100", True),
        # The bug this helper exists to kill: "h100" scraped to 100 and read as
        # ">= sm100", so a Hopper card got a Blackwell layout.
        ("nvidia", "h100", "sm100", False),
        # And "gfx942" scraped to 942, held off only by a vendor check.
        ("amd", "gfx942", "sm100", False),
        ("cpu", None, "sm100", False),
    ],
)
def test_arch_at_least_on_a_profile(vendor, arch, bound, expected) -> None:
    profile = DeviceProfile(vendor=vendor, arch=arch)
    assert arch_at_least(profile, bound) is expected


def test_a_bare_number_arch_is_rejected_at_construction() -> None:
    """No vendor prefixing: an architecture is always a series-prefixed name.

    ``DeviceProfile(vendor="amd", arch="100")`` used to fabricate ``gfx100``
    — an architecture that does not exist — and rely on the series mismatch
    to fail later. Rejecting the digits at construction keeps only one
    spelling alive.
    """

    with pytest.raises(ValueError, match="no series prefix"):
        DeviceProfile(vendor="nvidia", arch="100")
    with pytest.raises(ValueError, match="write the full name"):
        DeviceProfile(vendor="amd", arch="100")


def test_every_ordered_series_has_a_grammar() -> None:
    """Otherwise ``at_least`` would accept a bound it can never parse."""

    assert ORDERED_ARCH_SERIES <= set(ARCH_SERIES_GRAMMAR)


# --------------------------------------------------------------------------- #
# The synthetic-profile boundary
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text, arch",
    [
        ("nvidia:sm90", "sm90"),
        ("nvidia:SM100", "sm100"),
        ("amd:gfx942", "gfx942"),
        ("ascend:ascend910b", "ascend910b"),
    ],
)
def test_a_vendor_arch_string_is_accepted(text, arch) -> None:
    from phyai.kernel.device import _synthetic_profile

    assert _synthetic_profile(text).arch == arch


@pytest.mark.parametrize(
    "text, complaint",
    [
        # One spelling per device: everything short of 'vendor:arch' names
        # the grammar to write, and points at the canonical form when the
        # input makes the intent obvious.
        ("sm90", "did you mean 'nvidia:sm90'"),
        ("gfx942", "did you mean 'amd:gfx942'"),
        ("90", "'vendor:arch'"),
        ("nvidia", "'vendor:arch'"),
        ("cuda:sm90", "write 'nvidia:sm90'"),
        ("rocm:gfx942", "write 'amd:gfx942'"),
        ("npu:ascend910b", "write 'ascend:ascend910b'"),
        ("intel:xe2", "unknown device vendor"),
        # A prefixed architecture must belong to the vendor that owns the
        # series.
        ("nvidia:gfx942", "does not look like a nvidia architecture"),
        ("amd:sm90", "does not look like a amd architecture"),
        ("nvidia:100", "does not look like a nvidia architecture"),
    ],
)
def test_everything_else_is_rejected_with_guidance(text, complaint) -> None:
    from phyai.kernel.device import _synthetic_profile

    with pytest.raises(ValueError) as excinfo:
        _synthetic_profile(text)
    assert complaint in str(excinfo.value)


@pytest.mark.parametrize(
    "text, replacement",
    [
        ("nvidia:H100", "sm90"),
        ("nvidia:H200", "sm90"),
        ("nvidia:B200", "sm100"),
        ("nvidia:GB200", "sm100"),
        ("nvidia:A100", "sm80"),
        # Edge and workstation parts matter as much as datacenter ones for
        # physical AI: Jetson Orin, Jetson Thor, and consumer Blackwell.
        ("nvidia:AGX-Orin", "sm87"),
        ("nvidia:Thor", "sm110"),
        ("nvidia:RTX5090", "sm120"),
        ("nvidia:RTX4090", "sm89"),
        ("nvidia:DGX-Spark", "sm121"),
        ("amd:MI300", "gfx942"),
        ("amd:MI250", "gfx90a"),
    ],
)
def test_a_product_name_is_refused_and_names_its_architecture(
    text, replacement
) -> None:
    """A product name matched *nothing*: no capability reads one, so every
    architecture-gated kernel silently became ineligible.

    One place read it differently — materialization scraped the digits, so
    ``h100`` became 100 and a Hopper card got the Blackwell NVFP4 scale layout.
    The same string meant two things, and neither was an error.
    """

    from phyai.kernel.device import _synthetic_profile

    with pytest.raises(ValueError, match="product name") as excinfo:
        _synthetic_profile(text)
    assert replacement in str(excinfo.value)


def test_an_unrecognized_name_is_refused_with_the_shape_expected() -> None:
    """Not in the product table either — but still not an architecture, and
    guessing would reintroduce the silent failure."""

    from phyai.kernel.device import _synthetic_profile

    with pytest.raises(ValueError, match="does not look like"):
        _synthetic_profile("nvidia:hopper")


def test_the_product_table_maps_to_real_architectures() -> None:
    """Every replacement it suggests must itself parse, or the error message
    sends people somewhere equally broken — and it must belong to the vendor
    whose table it sits in."""

    from phyai.utils.vendors import VENDORS

    for vendor in VENDORS.values():
        for product, arch in vendor.products.items():
            parsed = Arch.parse(arch)
            assert parsed.major is not None, f"{product} -> {arch} does not parse"
            assert arch.startswith(
                vendor.series
            ), f"{product} -> {arch} is not a {vendor.name} architecture"
