"""The policy DSL: grammar, compilation, and load-time validation.

The grammar's whole claim is that one rule removes the previous ambiguity:
a list means membership, a mapping is nested-path sugar, and ``any_of`` /
``all_of`` / ``none_of`` are the only reserved words. These tests pin that,
and pin the load-time errors — because the failure mode being replaced is a
mistyped key that silently produced a rule which could never match.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import facts_from_query
from phyai.kernel.policy import (
    Policy,
    PolicyError,
    compile_matcher,
    load_policy,
    policy_from_mapping,
)
from phyai.kernel.registry import build_catalog
from phyai.kernel.types import KernelQuery


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


def facts_for(catalog, op: str, **kwargs):
    spec = catalog.op(op)
    query = KernelQuery.build(op, **kwargs)
    return facts_from_query(query, spec, libraries={"lib.flashinfer": True})


def gemm_facts(catalog, **overrides):
    base = dict(
        device="nvidia:SM100",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "nvfp4", "layout": "128x4", "block_shape": (1, 16)},
        shape={"M": 8, "N": 4096, "K": 4096},
    )
    base.update(overrides)
    return facts_for(catalog, "gemm", **base)


# --------------------------------------------------------------------------- #
# The one grammar rule
# --------------------------------------------------------------------------- #


def test_scalar_means_equality(catalog) -> None:
    predicate = compile_matcher({"device.vendor": "nvidia"})
    assert predicate.render() == "device.vendor == nvidia"


def test_list_means_membership(catalog) -> None:
    predicate = compile_matcher({"device.vendor": ["nvidia", "amd"]})
    assert predicate.render() == "device.vendor in {amd, nvidia}"


def test_list_on_a_set_fact_means_intersection(catalog) -> None:
    predicate = compile_matcher({"model.tags": ["debug", "reference"]})
    assert predicate.render() == "model.tags intersects {'debug', 'reference'}"


def test_mapping_is_nested_path_sugar(catalog) -> None:
    """``device: {vendor: x}`` and ``device.vendor: x`` must compile alike."""

    nested = compile_matcher({"device": {"vendor": "nvidia", "arch": ">=sm90"}})
    flat = compile_matcher({"device.vendor": "nvidia", "device.arch": ">=sm90"})
    assert nested.render() == flat.render()


def test_comparator_string_on_an_ordered_fact(catalog) -> None:
    assert (
        compile_matcher({"device.arch": ">=sm100"}).render() == "device.arch >= sm100"
    )


def test_divisibility_form(catalog) -> None:
    spec_paths = {"shape.K": catalog.op("gemm").kind_of("shape.K")}
    predicate = compile_matcher({"shape.K": "%16"}, spec_paths=spec_paths)
    assert predicate.render() == "shape.K % 16 == 0"


def test_null_means_known_absent(catalog) -> None:
    assert compile_matcher({"quant.format": None}).render() == "quant.format is none"


def test_combinators_are_the_only_reserved_words(catalog) -> None:
    predicate = compile_matcher(
        {
            "any_of": [{"device.vendor": "nvidia"}, {"device.vendor": "amd"}],
        }
    )
    assert predicate.render() == "device.vendor == nvidia | device.vendor == amd"

    predicate = compile_matcher({"none_of": [{"device.vendor": "cpu"}]})
    assert predicate.render() == "!(device.vendor == cpu)"


def test_when_compiles_to_an_implication(catalog) -> None:
    spec = catalog.op("layernorm")
    spec_paths = {path: spec.kind_of(path) for path in spec.known_paths()}
    predicate = compile_matcher(
        {"when": {"if": {"attrs.bias": True}, "then": {"dtype.bias": "fp32"}}},
        spec_paths=spec_paths,
    )
    assert predicate.render() == "attrs.bias == true -> dtype.bias == fp32"


def test_a_combinator_cannot_appear_inside_a_namespace(catalog) -> None:
    """This is the ambiguity the old grammar needed heuristics to resolve."""

    with pytest.raises(PolicyError, match="cannot appear inside"):
        compile_matcher({"device": {"any_of": [{"vendor": "nvidia"}]}})


def test_a_non_namespace_key_cannot_take_a_mapping(catalog) -> None:
    with pytest.raises(PolicyError, match="not a namespace"):
        compile_matcher({"op": {"min": 3}})


# --------------------------------------------------------------------------- #
# Load-time validation: the silent-False replacement
# --------------------------------------------------------------------------- #


def test_unknown_global_fact_is_rejected(catalog) -> None:
    with pytest.raises(PolicyError, match="unknown fact 'device.vendr'"):
        compile_matcher({"device.vendr": "nvidia"})


def test_unknown_fact_suggests_the_right_one(catalog) -> None:
    with pytest.raises(PolicyError, match="did you mean 'device.vendor'"):
        compile_matcher({"device.vendor_": "nvidia"})


def test_op_scoped_typo_is_caught_against_the_declared_schema(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "typo",
                "match": {"op": "gemm", "shape.KK": "%16"},
                "prefer": ["torch.gemm.bf16"],
            }
        ]
    }
    with pytest.raises(PolicyError, match=r"did you mean 'shape\.K'"):
        policy_from_mapping(document, catalog)


def test_case_mismatch_in_a_dimension_is_reported_as_case(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "case",
                "match": {"op": "gemm", "shape.k": "%16"},
                "prefer": ["torch.gemm.bf16"],
            }
        ]
    }
    with pytest.raises(PolicyError, match="case-sensitive"):
        policy_from_mapping(document, catalog)


def test_unknown_op_is_rejected(catalog) -> None:
    document = {
        "rules": [{"id": "x", "match": {"op": "gemmm"}, "prefer": ["torch.gemm.bf16"]}]
    }
    with pytest.raises(PolicyError, match="unknown op 'gemmm'"):
        policy_from_mapping(document, catalog)


def test_unknown_kernel_id_is_rejected_with_a_suggestion(catalog) -> None:
    document = {
        "rules": [{"id": "x", "match": {"op": "gemm"}, "prefer": ["torch.gemm.bf17"]}]
    }
    with pytest.raises(PolicyError, match="did you mean 'torch.gemm.bf16'"):
        policy_from_mapping(document, catalog)


def test_unknown_top_level_field_is_rejected(catalog) -> None:
    with pytest.raises(PolicyError, match="unknown top-level field"):
        policy_from_mapping({"rulez": []}, catalog)


def test_unknown_rule_field_is_rejected(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "x",
                "match": {"op": "gemm"},
                "prefer": ["torch.gemm.bf16"],
                "prefr": [],
            }
        ]
    }
    with pytest.raises(PolicyError, match="unknown field"):
        policy_from_mapping(document, catalog)


def test_a_rule_needs_exactly_one_of_prefer_or_restrict_to(catalog) -> None:
    """A rule that says nothing is a rule that silently never does anything.

    Saying both would be two answers to one question: ``prefer`` orders the full
    candidate list, ``restrict_to`` shortens it.
    """

    neither = {"rules": [{"id": "x", "match": {"op": "gemm"}}]}
    with pytest.raises(PolicyError, match="exactly one of"):
        policy_from_mapping(neither, catalog)

    both = {
        "rules": [
            {
                "id": "x",
                "match": {"op": "gemm"},
                "prefer": ["torch.gemm.bf16"],
                "restrict_to": "flashinfer.gemm.*",
            }
        ]
    }
    with pytest.raises(PolicyError, match="exactly one of"):
        policy_from_mapping(both, catalog)


def test_an_override_needs_exactly_one_of_use_or_restrict_to(catalog) -> None:
    both = {
        "overrides": [
            {
                "id": "x",
                "match": {"op": "gemm"},
                "use": "torch.gemm.bf16",
                "restrict_to": "torch.gemm.*",
            }
        ]
    }
    with pytest.raises(PolicyError, match="exactly one of"):
        policy_from_mapping(both, catalog)

    neither = {"overrides": [{"id": "x", "match": {"op": "gemm"}}]}
    with pytest.raises(PolicyError, match="exactly one of"):
        policy_from_mapping(neither, catalog)


def test_restrict_to_matching_nothing_is_rejected(catalog) -> None:
    document = {
        "overrides": [{"id": "x", "match": {"op": "gemm"}, "restrict_to": "cutlass.*"}]
    }
    with pytest.raises(PolicyError, match="matches no kernel"):
        policy_from_mapping(document, catalog)


def test_unsupported_schema_is_rejected(catalog) -> None:
    with pytest.raises(PolicyError, match="unsupported schema"):
        policy_from_mapping({"schema": "phyai.kernel/v9"}, catalog)


def test_v1_schema_spellings_are_all_accepted(catalog) -> None:
    for schema in ("phyai.kernel/v1", "v1", 1, "1"):
        assert policy_from_mapping({"schema": schema}, catalog).profile == "static"


def test_invalid_profile_and_fallback_are_rejected(catalog) -> None:
    with pytest.raises(PolicyError, match="profile"):
        policy_from_mapping({"profile": "turbo"}, catalog)
    with pytest.raises(PolicyError, match="fallback"):
        policy_from_mapping({"defaults": {"fallback": "explode"}}, catalog)


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #


def test_highest_priority_rule_wins(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "low",
                "priority": 1,
                "match": {"op": "gemm"},
                "prefer": ["torch.gemm.bf16"],
            },
            {
                "id": "high",
                "priority": 2,
                "match": {"op": "gemm"},
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            },
        ]
    }
    policy = policy_from_mapping(document, catalog)
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert decision.matched_rules == ("high",)
    assert decision.candidates == ("flashinfer.gemm.nvfp4_128x4",)


def test_a_priority_tie_is_an_error_not_an_arbitrary_pick(catalog) -> None:
    document = {
        "rules": [
            {"id": "a", "match": {"op": "gemm"}, "prefer": ["torch.gemm.bf16"]},
            {
                "id": "b",
                "match": {"op": "gemm"},
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            },
        ]
    }
    policy = policy_from_mapping(document, catalog)
    with pytest.raises(PolicyError, match="conflicting rules"):
        policy.decide(gemm_facts(catalog), catalog)


def test_an_override_beats_any_rule_and_is_strict(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "r",
                "priority": 999,
                "match": {"op": "gemm"},
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            }
        ],
        "overrides": [
            {
                "id": "o",
                "priority": 1,
                "match": {"op": "gemm"},
                "use": "torch.gemm.bf16",
            }
        ],
    }
    policy = policy_from_mapping(document, catalog)
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert decision.matched_rules == ("o",)
    assert decision.strict
    assert decision.candidates == ("torch.gemm.bf16",)


def test_restrict_to_expands_to_a_kernel_family(catalog) -> None:
    """What "force this backend" has always meant: narrow, then order."""

    document = {
        "overrides": [
            {"id": "o", "match": {"op": "gemm"}, "restrict_to": "torch.gemm.*"}
        ]
    }
    policy = policy_from_mapping(document, catalog)
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert set(decision.candidates) == {
        "torch.gemm.bf16",
        "torch.gemm.fp8_block",
        "torch.gemm.fp8_per_channel",
        "torch.gemm.fp8_per_tensor",
        "torch.gemm.nvfp4_linear",
    }


def test_no_matching_rule_expresses_no_preference(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "amd-only",
                "match": {"op": "gemm", "device.vendor": "amd"},
                "prefer": ["torch.gemm.bf16"],
            }
        ]
    }
    policy = policy_from_mapping(document, catalog)
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert decision.candidates == ()
    assert decision.matched_rules == ()


def test_matcher_narrows_on_quantization_facts(catalog) -> None:
    document = {
        "rules": [
            {
                "id": "nvfp4-only",
                "match": {
                    "op": "gemm",
                    "quant": {"format": "nvfp4", "layout": "128x4"},
                },
                "prefer": ["flashinfer.gemm.nvfp4_128x4"],
            }
        ]
    }
    policy = policy_from_mapping(document, catalog)
    assert policy.decide(gemm_facts(catalog), catalog).matched_rules == ("nvfp4-only",)

    unquantized = gemm_facts(catalog, quant=None)
    assert policy.decide(unquantized, catalog).matched_rules == ()


# --------------------------------------------------------------------------- #
# Versioning
# --------------------------------------------------------------------------- #


def test_version_changes_when_a_matcher_changes(catalog) -> None:
    def build(sm: str) -> Policy:
        return policy_from_mapping(
            {
                "rules": [
                    {
                        "id": "r",
                        "match": {"op": "gemm", "device.arch": sm},
                        "prefer": ["torch.gemm.bf16"],
                    }
                ]
            },
            catalog,
        )

    assert build(">=sm90").version != build(">=sm100").version


def test_version_changes_when_only_params_change(catalog) -> None:
    """Isolated from the matcher, unlike the previous test of this property."""

    def build(tile: int) -> Policy:
        return policy_from_mapping(
            {
                "rules": [
                    {
                        "id": "r",
                        "match": {"op": "gemm"},
                        "prefer": ["torch.gemm.bf16"],
                        "params": {"tile": tile},
                    }
                ]
            },
            catalog,
        )

    assert build(64).version != build(128).version


def test_version_is_stable_across_equivalent_spellings(catalog) -> None:
    """Nested and flat forms compile to the same predicate, so same version."""

    nested = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "r",
                    "match": {"op": "gemm", "device": {"vendor": "nvidia"}},
                    "prefer": ["torch.gemm.bf16"],
                }
            ]
        },
        catalog,
    )
    flat = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "r",
                    "match": {"op": "gemm", "device.vendor": "nvidia"},
                    "prefer": ["torch.gemm.bf16"],
                }
            ]
        },
        catalog,
    )
    assert nested.version == flat.version


# --------------------------------------------------------------------------- #
# Loading from disk
# --------------------------------------------------------------------------- #


def test_no_path_yields_the_deterministic_default(catalog) -> None:
    policy = load_policy(None, catalog)
    assert policy.profile == "static"
    assert policy.fallback == "reference"
    assert policy.rules == ()


def test_load_error_names_the_file(catalog, tmp_path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("rules:\n  - id: x\n    match: {op: nope}\n", encoding="utf-8")
    with pytest.raises(PolicyError, match=str(path.name)):
        load_policy(path, catalog)


def test_a_rule_can_narrow_to_a_backend_family(catalog) -> None:
    """What the deleted ``force this backend`` setting used to do, in YAML.

    ``restrict_to`` with a glob narrows to a *family* and orders its rows
    normally, rather than pinning one specialization — which is what a
    "use FlashInfer for GEMM" instruction has always meant.
    """

    policy = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "flashinfer-gemm",
                    "priority": 900,
                    "match": {"op": "gemm"},
                    "restrict_to": "flashinfer.gemm.*",
                }
            ]
        },
        catalog,
    )
    decision = policy.decide(gemm_facts(catalog), catalog)
    assert decision.matched_rules == ("flashinfer-gemm",)
    assert not decision.strict  # soft: a CPU box falls back, it does not fail
    assert set(decision.candidates) == {
        "flashinfer.gemm.bf16",
        "flashinfer.gemm.fp8_block",
        "flashinfer.gemm.nvfp4_128x4",
    }
    # A gemm rule must not capture some other operation's calls.
    norm = facts_for(
        catalog,
        "rmsnorm",
        device="nvidia:SM90",
        dtype={"input": "bf16", "weight": "bf16"},
        shape={"tokens": 8, "hidden": 4096},
        attrs={"variant": "rms"},
    )
    assert policy.decide(norm, catalog).matched_rules == ()


def test_a_rule_can_narrow_to_one_role(catalog) -> None:
    """The thing the global setting could not do, and the reason it is gone.

    ``PHYAI_LINEAR_BACKEND=torch`` moved *every* linear onto torch. An A/B
    almost always wants one role — everything else has to stay put, or the
    measurement means nothing.
    """

    policy = policy_from_mapping(
        {
            "rules": [
                {
                    "id": "ab-mlp-down",
                    "priority": 100,
                    "match": {"op": "gemm", "role": "mlp.down"},
                    "prefer": ["torch.gemm.bf16"],
                }
            ]
        },
        catalog,
    )
    bf16 = dict(
        device="nvidia:SM100",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "bf16"},
        shape={"M": 512, "N": 4096, "K": 4096},
    )
    targeted = facts_for(catalog, "gemm", role="mlp.down", **bf16)
    assert policy.decide(targeted, catalog).candidates[0] == "torch.gemm.bf16"

    # Same op, same shapes, different role: untouched.
    other = facts_for(catalog, "gemm", role="qkv_proj", **bf16)
    assert policy.decide(other, catalog).matched_rules == ()
