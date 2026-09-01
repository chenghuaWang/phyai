"""Priority bands: what they promise an out-of-tree kernel author.

A bare integer cannot answer "will 60 beat the built-in FlashInfer row?" — you
have to read every registration to find out, and the answer changes whenever
someone bumps a number. These tests pin the contract that replaces that: named
bands, a reserved range above every in-tree row, and a rejection for anything
outside the scheme.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import device, dtype
from phyai.kernel.opspec import (
    PRIORITY_LIMIT,
    Impl,
    OpSpec,
    Priority,
    band_for,
    validate_priority,
)
from phyai.kernel.registry import Catalog, build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


def impl(priority, kernel_id="toy.kernel") -> Impl:
    return Impl(
        kernel_id=kernel_id,
        op="toy",
        when=dtype.input.is_set(),
        prepare=lambda facts, params: None,
        priority=priority,
    )


# --------------------------------------------------------------------------- #
# The bands themselves
# --------------------------------------------------------------------------- #


def test_the_bands_are_ordered_and_four_wide() -> None:
    """Four wide so an offset expresses relative preference inside one contract
    without crossing into the next band's meaning."""

    values = [int(band) for band in Priority]
    assert values == sorted(values)
    assert values == [0, 4, 8, 12, 16]
    assert PRIORITY_LIMIT == 20


def test_plugin_is_the_top_band() -> None:
    """The whole point: an out-of-tree row can outrank every in-tree row
    without auditing the tree."""

    assert int(Priority.PLUGIN) == max(int(band) for band in Priority)
    assert int(Priority.PLUGIN) + 4 == PRIORITY_LIMIT


@pytest.mark.parametrize(
    "value, band",
    [
        (0, Priority.REFERENCE),
        (3, Priority.REFERENCE),
        (4, Priority.GENERAL),
        (8, Priority.OPTIMIZED),
        (10, Priority.OPTIMIZED),
        (12, Priority.SPECIALIZED),
        (16, Priority.PLUGIN),
        (19, Priority.PLUGIN),
    ],
)
def test_band_for_reports_the_containing_band(value, band) -> None:
    assert band_for(value) is band


def test_an_offset_stays_an_int_and_stays_in_its_band() -> None:
    assert Priority.OPTIMIZED + 2 == 10
    assert band_for(Priority.OPTIMIZED + 2) is Priority.OPTIMIZED


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [-1, 20, 100, 1000])
def test_a_priority_outside_the_bands_is_rejected(value) -> None:
    """Rejecting rather than clamping.

    A row written ``priority=100`` under the old free-for-all would silently
    land above the plugin band and outrank everything — the exact coordination
    failure bands exist to prevent. Clamping would hide it.
    """

    with pytest.raises(ValueError, match=r"priority must be in \[0, 20\)"):
        validate_priority(value)


def test_the_rejection_names_the_bands() -> None:
    """So the fix is in the error, not in the source."""

    with pytest.raises(ValueError) as excinfo:
        impl(100)
    message = str(excinfo.value)
    for band in Priority:
        assert band.name in message


def test_a_valid_priority_passes_through_as_a_plain_int() -> None:
    row = impl(Priority.OPTIMIZED + 2)
    assert row.priority == 10
    assert type(row.priority) is int


def test_the_default_is_the_reference_band() -> None:
    """A row that does not say otherwise must not outrank one that does."""

    row = Impl(
        kernel_id="toy.default",
        op="toy",
        when=dtype.input.is_set(),
        prepare=lambda facts, params: None,
    )
    assert row.priority == int(Priority.REFERENCE)


# --------------------------------------------------------------------------- #
# What the bands do and do not affect
# --------------------------------------------------------------------------- #


def test_priority_orders_eligible_rows() -> None:
    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
    catalog.register(impl(Priority.REFERENCE, "low.toy"))
    catalog.register(impl(Priority.PLUGIN, "high.toy"))
    ids = [row.kernel_id for row in catalog.impls("toy")]
    assert ids == ["high.toy", "low.toy"]


def test_a_high_priority_cannot_make_a_row_eligible() -> None:
    """Capability is filtered first; priority is only the tiebreaker among
    survivors. Otherwise a plugin could force itself onto hardware it cannot
    run on."""

    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
    catalog.register(
        Impl(
            kernel_id="blackwell.toy",
            op="toy",
            when=device.arch.at_least("sm100"),
            prepare=lambda facts, params: None,
            priority=Priority.PLUGIN + 3,
        )
    )
    catalog.register(impl(Priority.REFERENCE, "anywhere.toy"))
    selector = Selector(catalog, device="nvidia:SM90")
    trace = selector.explain(KernelQuery.build("toy", dtype={"input": "bf16"}))
    assert trace.selected == "anywhere.toy"


def test_every_in_tree_row_leaves_the_plugin_band_free() -> None:
    """The reserved-headroom promise, checked against the real catalog.

    If an in-tree row ever reaches ``PLUGIN``, a plugin author has no number
    left that reliably wins, and the band stops meaning anything.
    """

    offenders = [
        (row.kernel_id, row.priority)
        for spec in build_catalog().ops()
        for row in build_catalog().impls(spec.name)
        if row.priority >= int(Priority.PLUGIN)
    ]
    assert not offenders, f"in-tree rows in the plugin band: {offenders}"


def test_every_in_tree_row_is_in_a_declared_band() -> None:
    """No row should sit at a number whose band nobody chose."""

    catalog = build_catalog()
    for spec in catalog.ops():
        for row in catalog.impls(spec.name):
            assert 0 <= row.priority < PRIORITY_LIMIT, row.kernel_id
            assert band_for(row.priority) in set(Priority), row.kernel_id


def test_reference_rows_sit_in_the_reference_band() -> None:
    """``reference=True`` and the REFERENCE band are separate concepts — the
    flag drives ``fallback: reference`` and the startup coverage check, the band
    drives ordering — but a reference row that outranked a real one would be a
    mistake in every case."""

    catalog = build_catalog()
    for spec in catalog.ops():
        for row in catalog.impls(spec.name):
            if row.reference:
                assert band_for(row.priority) is Priority.REFERENCE, row.kernel_id
