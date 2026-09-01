"""Catalog structure and the invariants that stop declarations from drifting.

Two of these are guards against specific mistakes the previous design made and
could not detect:

* ``capture_safe`` was hand-copied into the descriptor table and disagreed with
  the backend class it described. Here the two are compared directly, so they
  cannot diverge again.
* the operation modules must not reach into ``phyai.layers`` at import time.
  With the selector now constructible from inside a layer's forward pass, an
  import cycle would be an unpleasant failure to diagnose.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from phyai.kernel.opspec import Impl, OpSpec
from phyai.kernel.ops import OP_MODULES
from phyai.kernel.ops import attention as attention_ops
from phyai.kernel.facts import FactKind, device, dtype
from phyai.kernel.opspec import Priority
from phyai.kernel.predicate import all_of
from phyai.kernel.registry import (
    Catalog,
    UnknownKernelError,
    UnknownOperationError,
    build_catalog,
)


#: Operations that genuinely have no CPU implementation. Written down so a
#: fourth cannot join the set unnoticed.
NO_CPU_PATH = frozenset({"attention_paged", "attention_gdn"})


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #


def test_every_declared_module_registers_something(catalog) -> None:
    assert len(OP_MODULES) == 7
    assert len(catalog.ops()) >= len(OP_MODULES)
    for spec in catalog.ops():
        assert catalog.impls(spec.name), f"{spec.name} has no implementations"


def test_implementations_are_ordered_best_first(catalog) -> None:
    rows = catalog.impls("rmsnorm")
    assert [item.kernel_id for item in rows] == [
        "flashinfer.rmsnorm",
        "phyai_kernel.rmsnorm",
        "torch.rmsnorm",
    ]
    priorities = [item.priority for item in rows]
    assert priorities == sorted(priorities, reverse=True)


def test_ordering_is_independent_of_registration_order() -> None:
    spec = OpSpec(name="toy", dtypes=("input",))

    def build(reverse: bool) -> Catalog:
        catalog = Catalog()
        catalog.register_op(spec)
        rows = [
            Impl(
                kernel_id=f"{name}.toy",
                op="toy",
                priority=priority,
                when=dtype.input.is_set(),
                prepare=lambda facts, params: None,
            )
            for name, priority in (
                ("a", Priority.GENERAL),
                ("b", Priority.OPTIMIZED + 2),
                ("c", Priority.OPTIMIZED),
            )
        ]
        catalog.register_many(reversed(rows) if reverse else rows)
        return catalog

    assert [i.kernel_id for i in build(False).impls()] == [
        i.kernel_id for i in build(True).impls()
    ]


def test_registering_an_impl_for_an_unknown_op_is_rejected() -> None:
    catalog = Catalog()
    with pytest.raises(UnknownOperationError, match="unregistered operation"):
        catalog.register(
            Impl(
                kernel_id="x.nope",
                op="nope",
                when=dtype.input.is_set(),
                prepare=lambda facts, params: None,
            )
        )


def test_duplicate_kernel_ids_are_rejected() -> None:
    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
    row = dict(op="toy", when=dtype.input.is_set(), prepare=lambda facts, params: None)
    catalog.register(Impl(kernel_id="dup.toy", **row))
    with pytest.raises(ValueError, match="duplicate kernel id"):
        catalog.register(Impl(kernel_id="dup.toy", **row))


def test_conflicting_op_schemas_are_rejected() -> None:
    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dims=("M",)))
    with pytest.raises(ValueError, match="different schema"):
        catalog.register_op(OpSpec(name="toy", dims=("N",)))


def test_unknown_lookups_raise_clearly(catalog) -> None:
    with pytest.raises(UnknownKernelError, match="unknown kernel"):
        catalog.get("nope.gemm")
    with pytest.raises(UnknownOperationError, match="unknown operation"):
        catalog.op("teleport")


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def test_no_coverage_gaps(catalog) -> None:
    """Every operation has a reference row, or declares that it cannot."""

    assert catalog.coverage_gaps() == {}


def test_only_the_known_operations_lack_a_cpu_path(catalog) -> None:
    """The exception list is closed.

    A CPU host cannot run the paged attention operations, and that is a
    property of FlashInfer, not an oversight. Pinning the set means a fourth
    operation cannot quietly become CPU-hostile.
    """

    without_reference = {
        spec.name
        for spec in catalog.ops()
        if not any(item.reference for item in catalog.impls(spec.name))
    }
    assert without_reference == NO_CPU_PATH
    for name in NO_CPU_PATH:
        assert not catalog.op(name).requires_reference


def test_a_missing_reference_is_reported() -> None:
    catalog = Catalog()
    catalog.register_op(OpSpec(name="toy", dtypes=("input",)))
    catalog.register(
        Impl(
            kernel_id="fast.toy",
            op="toy",
            when=all_of(device.vendor == "nvidia", dtype.input.is_set()),
            prepare=lambda facts, params: None,
        )
    )
    assert "toy" in catalog.coverage_gaps()


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def test_version_is_a_short_stable_hash(catalog) -> None:
    assert len(catalog.version) == 16
    assert catalog.version == build_catalog().version


def test_version_changes_when_a_capability_changes() -> None:
    """The old descriptor hash omitted the capability, so tightening a
    contract could leave a stale cached choice in place."""

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


def test_manifest_is_json_serializable_and_complete(catalog) -> None:
    payload = catalog.manifest()
    text = json.dumps(payload, sort_keys=True)
    assert payload["version"] == catalog.version
    assert len(payload["operations"]) == len(catalog.ops())
    gemm = next(item for item in payload["operations"] if item["name"] == "gemm")
    assert gemm["signature"] == "(layer, x, bias) -> Tensor"
    ids = {row["id"] for row in gemm["implementations"]}
    assert "flashinfer.gemm.nvfp4_128x4" in ids
    # The rendered contract is the point of the manifest.
    row = next(
        r for r in gemm["implementations"] if r["id"] == "flashinfer.gemm.nvfp4_128x4"
    )
    assert "quant.layout == 128x4" in row["when"]
    assert row["libraries"] == ["flashinfer"]
    assert "nvfp4" in text


def test_manifest_is_byte_stable_across_builds(catalog) -> None:
    """So CI can diff it and see intended changes only."""

    first = json.dumps(build_catalog().manifest(), sort_keys=True)
    second = json.dumps(build_catalog().manifest(), sort_keys=True)
    assert first == second


def test_describe_prints_one_line_per_implementation(catalog) -> None:
    lines = catalog.describe().splitlines()
    assert len(lines) == len(catalog.impls())
    assert any("quant.format == nvfp4" in line for line in lines)


def test_libraries_are_derived_from_capabilities(catalog) -> None:
    """The selector must import only what eligibility actually depends on."""

    assert catalog.libraries() == {"flashinfer", "phyai_kernel", "fla", "flash_qla"}
    assert catalog.libraries("gemm") == {"flashinfer"}
    assert catalog.libraries("attention_gdn") == {"flashinfer", "fla", "flash_qla"}
    # No implementation of the reference-only side needs a library.
    assert catalog.libraries("activation") == {"flashinfer"}


def test_match_ids_expands_a_family(catalog) -> None:
    assert set(catalog.match_ids("flashinfer.gemm.*")) == {
        "flashinfer.gemm.bf16",
        "flashinfer.gemm.fp8_block",
        "flashinfer.gemm.nvfp4_128x4",
    }
    assert catalog.match_ids("nothing.*") == ()


# --------------------------------------------------------------------------- #
# Drift guards
# --------------------------------------------------------------------------- #


ATTENTION_BACKENDS = {
    "flashinfer.attention": attention_ops.ATTENTION_FLASHINFER_CLS,
    "sdpa.attention": attention_ops.ATTENTION_SDPA_CLS,
    "eager.attention": attention_ops.ATTENTION_EAGER_CLS,
    "flashinfer.attention_paged": attention_ops.PAGED_FLASHINFER_CLS,
    "flashinfer.attention_gdn": attention_ops.GDN_FLASHINFER_CLS,
    "fla.attention_gdn": attention_ops.GDN_FLA_CLS,
    "flash_qla.attention_gdn": attention_ops.GDN_FLASH_QLA_CLS,
}


@pytest.mark.parametrize(
    ("kernel_id", "module_path", "class_name"),
    [(k, v[0], v[1]) for k, v in ATTENTION_BACKENDS.items()],
)
def test_capture_safe_matches_the_backend_class(
    catalog, kernel_id: str, module_path: str, class_name: str
) -> None:
    """The declaration and the implementation must agree.

    The previous table claimed ``capture_safe=True`` for every attention
    backend by copying a template. ``gdn.flashinfer`` never overrides
    ``supports_capture()`` and so reports ``False`` — a disagreement nothing
    could detect. Reading the class attribute (no instance, no CUDA, no
    flashinfer install needed) from the exact (module, class) pair the
    catalog row constructs makes the two impossible to separate.
    """

    import importlib

    backend_cls = getattr(importlib.import_module(module_path), class_name)
    declared = catalog.get(kernel_id).capture_safe
    actual = backend_cls.supports_capture(backend_cls)  # unbound, no instance
    assert declared == actual, (
        f"{kernel_id} declares capture_safe={declared} but "
        f"{backend_cls.__name__}.supports_capture() returns {actual}"
    )


def test_gdn_flashinfer_is_not_capture_safe(catalog) -> None:
    """Stated directly, because it is the value that was wrong before.

    Nothing in the tree calls ``init_cuda_graph_state`` on a GDN backend, so
    the captured path has never been exercised.
    """

    assert catalog.get("flashinfer.attention_gdn").capture_safe is False
    assert catalog.get("flash_qla.attention_gdn").capture_safe is False
    assert catalog.get("fla.attention_gdn").capture_safe is True


def test_op_modules_do_not_import_phyai_layers() -> None:
    """Capability is pure data; every backend import is deferred to prepare."""

    code = (
        "import importlib, sys\n"
        f"for name in {list(OP_MODULES)!r}:\n"
        "    importlib.import_module(f'phyai.kernel.ops.{name}')\n"
        "leaked = sorted(k for k in sys.modules if k.startswith('phyai.layers'))\n"
        "print(leaked)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert (
        result.stdout.strip() == "[]"
    ), f"operation modules leaked layer imports: {result.stdout.strip()}"


def test_building_the_catalog_does_not_import_flashinfer() -> None:
    """Constructing the catalog must stay cheap and side-effect free."""

    code = (
        "import sys\n"
        "from phyai.kernel.registry import build_catalog\n"
        "build_catalog()\n"
        "print('flashinfer' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"


def test_every_capability_references_at_least_one_fact(catalog) -> None:
    """A row eligible for everything would be a bug, not a feature."""

    for impl in catalog.impls():
        assert impl.when.facts_used(), f"{impl.kernel_id} constrains nothing"


def test_kind_of_matches_the_namespace_conventions(catalog) -> None:
    gemm = catalog.op("gemm")
    assert gemm.kind_of("shape.K") is FactKind.INT
    assert gemm.kind_of("dtype.input") is FactKind.DTYPE
    assert gemm.kind_of("attrs.anything") is FactKind.ANY
