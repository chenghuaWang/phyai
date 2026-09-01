"""FlashInfer prefill-kernel rows: gating, default order, and rule scoping.

Selection is pure fact evaluation, so a synthetic ``device=`` string is enough
to assert what a Hopper or Blackwell box would pick. That is the point of
having these as catalog rows instead of a config field — the choice becomes
testable without the hardware.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from phyai.kernel.ops.attention import FA3_HEAD_DIMS
from phyai.kernel.policy import Policy, PolicyError, policy_from_mapping
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.types import KernelQuery


def _query(op: str, *, head_dim: int = 256, role: str = "", **attrs):
    """A realistic paged/ragged attention query, mirroring shared.py."""
    layout = "ragged" if op == "attention" else "paged"
    base_attrs = {"layout": layout, "causal": True}
    base_attrs.update(attrs)
    shape = {
        "tokens": 512,
        "heads": 8,
        "kv_heads": 8,
        "head_dim": head_dim,
    }
    if op == "attention":
        shape["kv_tokens"] = 512
    return KernelQuery.build(
        op,
        role=role or op.removeprefix("attention_"),
        dtype={"input": "bf16", "key": "bf16", "value": "bf16", "output": "bf16"},
        shape=shape,
        attrs=base_attrs,
        mode="eager",
    )


def _selector(device: str, policy: Policy | None = None, **kw) -> Selector:
    if policy is None:
        return Selector(build_catalog(), device=device, **kw)
    return Selector(build_catalog(), policy, device=device, **kw)


def _select(op: str, *, device: str, policy: Policy | None = None, **kw) -> str:
    return _selector(device, policy).select(_query(op, **kw)).kernel_id


def _eligible(op: str, *, device: str, **kw) -> set[str]:
    """Rows whose capability the call satisfies, per explain()."""
    trace = _selector(device).explain(_query(op, **kw))
    return {item.kernel_id for item in trace.candidates if item.eligible}


def _rejection(op: str, kernel_id: str, *, device: str, **kw) -> str:
    trace = _selector(device).explain(_query(op, **kw))
    for item in trace.candidates:
        if item.kernel_id == kernel_id:
            return item.reason
    raise AssertionError(f"{kernel_id} not in the trace for {op}")


# --------------------------------------------------------------------------- #
# registration shape                                                          #
# --------------------------------------------------------------------------- #


def test_every_prefill_row_is_registered():
    catalog = build_catalog()
    assert set(catalog.match_ids("flashinfer.attention_paged.*")) == {
        "flashinfer.attention_paged.fa2",
        "flashinfer.attention_paged.fa3",
        "flashinfer.attention_paged.cudnn",
        "flashinfer.attention_paged.trtllm-gen",
    }
    # The ragged wrapper accepts two kernels the paged one does not. Under the
    # old global config field these were unreachable: its single valid-name set
    # was written for the paged wrapper.
    ragged = set(catalog.match_ids("flashinfer.attention.*"))
    assert {"flashinfer.attention.cutlass", "flashinfer.attention.cute-dsl"} <= ragged
    assert "flashinfer.attention.trtllm-gen" not in ragged


def test_paged_rows_exclude_cute_dsl():
    """The paged wrapper raises NotImplementedError for it."""
    catalog = build_catalog()
    assert "flashinfer.attention_paged.cute-dsl" not in catalog.kernel_ids()


def test_auto_keeps_the_bare_id_and_outranks_every_pinned_row():
    catalog = build_catalog()
    for op in ("attention", "attention_paged"):
        auto = catalog.get(f"flashinfer.{op}")
        pinned = catalog.match_ids(f"flashinfer.{op}.*")
        assert pinned
        for kernel_id in pinned:
            assert int(catalog.get(kernel_id).priority) < int(auto.priority)


def test_pinned_rows_still_outrank_sdpa():
    catalog = build_catalog()
    sdpa = int(catalog.get("sdpa.attention").priority)
    for kernel_id in catalog.match_ids("flashinfer.attention.*"):
        assert int(catalog.get(kernel_id).priority) > sdpa


def test_rows_record_their_prefill_backend_in_metadata():
    catalog = build_catalog()
    row = catalog.get("flashinfer.attention_paged.fa2")
    assert row.metadata["prefill_backend"] == "fa2"


# --------------------------------------------------------------------------- #
# default selection is unchanged                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_default_selection_is_auto(op):
    """No rule, no autotune -> the same row as before these variants existed."""
    assert _select(op, device="nvidia:SM90") == f"flashinfer.{op}"


# --------------------------------------------------------------------------- #
# gating is generation membership, not a floor                                #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("arch", ("SM80", "SM89", "SM120"))
@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_fa3_is_not_eligible_off_hopper(op, arch):
    """SM120 is the case a `sm >= 90` floor gets wrong.

    FLASHINFER_GDN carries the same lesson: a floor let an sm120 device pass
    selection and then raise inside the backend, where a raise cannot drive a
    fallback.
    """
    assert f"flashinfer.{op}.fa3" not in _eligible(op, device=f"nvidia:{arch}")
    # The trace names the failing leaf, so a regression says which gate moved.
    assert "arch" in _rejection(op, f"flashinfer.{op}.fa3", device=f"nvidia:{arch}")


@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_fa3_is_eligible_on_hopper(op):
    assert f"flashinfer.{op}.fa3" in _eligible(op, device="nvidia:SM90")


@pytest.mark.parametrize("op", ("attention", "attention_paged"))
def test_fa3_needs_a_supported_head_dim(op):
    """Mirrors flashinfer's is_fa3_prefill_head_dim_supported.

    The point is that an unsupported head dim *deselects* FA3 rather than
    reaching the backend and raising, since a raise cannot drive a fallback.
    Since the paged op now carries the same head-dim gate as the no-cache one
    (a 96-dim call used to select fine and die inside ``wrapper.plan``), a
    ragged 96-dim call falls through to the reference row and a paged one has
    no eligible row at all.
    """
    unsupported = 96
    assert unsupported not in FA3_HEAD_DIMS
    eligible = _eligible(op, device="nvidia:SM90", head_dim=unsupported)
    assert f"flashinfer.{op}.fa3" not in eligible

    if op == "attention":
        assert _select(op, device="nvidia:SM90", head_dim=unsupported) == (
            "eager.attention"
        )
    else:
        assert not eligible  # no paged kernel claims an unsupported head_dim


def test_blackwell_only_rows_need_blackwell():
    for arch, expected in (("SM90", False), ("SM100", True)):
        paged = _eligible("attention_paged", device=f"nvidia:{arch}")
        ragged = _eligible("attention", device=f"nvidia:{arch}")
        assert ("flashinfer.attention_paged.trtllm-gen" in paged) is expected
        assert ("flashinfer.attention.cutlass" in ragged) is expected


def test_fa2_is_eligible_everywhere_flashinfer_is():
    """FA2 is the universal fallback; it carries no extra gate."""
    for arch in ("SM80", "SM90", "SM100", "SM120"):
        eligible = _eligible("attention_paged", device=f"nvidia:{arch}")
        assert "flashinfer.attention_paged.fa2" in eligible


# --------------------------------------------------------------------------- #
# rule scoping — the defect the global config field could not express          #
# --------------------------------------------------------------------------- #


def _pi05_policy() -> Policy:
    return policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "pi05-expert-joint-fa2",
                    "priority": 10,
                    "match": {
                        "op": "attention_paged",
                        "role": "expert",
                        "model.family": ["pi05", "pi05_wn"],
                    },
                    "restrict_to": "flashinfer.attention_paged.fa2",
                }
            ],
        },
        build_catalog(),
        source="test",
    )


@pytest.mark.parametrize("family", ("pi05", "pi05_wn"))
def test_pi05_rule_pins_only_the_expert_role(family):
    """The regression test for the old field's grain.

    pi0.5 measured FA2 against auto on the action expert's joint attention
    alone. A global field also pinned the LLM prefix and the vision tower,
    neither of which was measured. With one paged op the scoping moves to
    ``role``: the expert pins, the prefix on the *same op* stays on
    FlashInfer's own heuristic.
    """
    from phyai.kernel.types import ModelContext

    selector = Selector(
        build_catalog(),
        _pi05_policy(),
        device="nvidia:SM90",
        model=ModelContext(family=family),
    )
    assert (
        selector.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged.fa2"
    )
    # Untouched: the prefix role and the vision tower keep the auto heuristic.
    assert (
        selector.select(_query("attention_paged", role="prefix")).kernel_id
        == "flashinfer.attention_paged"
    )
    assert selector.select(_query("attention")).kernel_id == "flashinfer.attention"


def test_pi05_rule_does_not_leak_to_other_models():
    from phyai.kernel.types import ModelContext

    selector = Selector(
        build_catalog(),
        _pi05_policy(),
        device="nvidia:SM90",
        model=ModelContext(family="cosmos3"),
    )
    assert (
        selector.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged"
    )


def test_rule_params_override_a_rows_pinned_backend():
    """A row pins its kernel by existing; `params:` still wins over it."""
    catalog = build_catalog()
    policy = policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "override",
                    "priority": 20,
                    "match": {"op": "attention_paged"},
                    "restrict_to": "flashinfer.attention_paged.fa2",
                    "params": {"prefill_backend": "fa3"},
                }
            ],
        },
        catalog,
        source="test",
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    decision = selector.select(_query("attention_paged"))
    assert decision.kernel_id == "flashinfer.attention_paged.fa2"
    assert decision.params["prefill_backend"] == "fa3"


def test_a_rule_param_the_backend_cannot_accept_is_an_error():
    """A mis-spelled rule parameter raises instead of silently vanishing.

    The prepare step checks rule params against the backend constructor's
    signature, and the failure propagates even without strict mode — a
    configuration error downgraded to a fallback would be invisible.
    """
    catalog = build_catalog()
    policy = policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "typo",
                    "priority": 20,
                    "match": {"op": "attention_paged"},
                    "restrict_to": "flashinfer.attention_paged.fa2",
                    "params": {"prefill_backend_": "fa2"},
                }
            ],
        },
        catalog,
        source="test",
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    with pytest.raises(PolicyError, match="prefill_backend_"):
        selector.select(_query("attention_paged"))


def test_rule_params_do_not_leak_to_reference_fallbacks():
    """Params target the kernels the rule names.

    When the target is ineligible, the reference fallback the policy
    appends runs without the params instead of failing on them.
    """
    catalog = build_catalog()
    policy = policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "tuned",
                    "priority": 20,
                    "match": {"op": "attention"},
                    "restrict_to": "flashinfer.attention.fa2",
                    "params": {"prefill_backend": "fa2"},
                }
            ],
        },
        catalog,
        source="test",
    )
    selector = Selector(catalog, policy, device="cpu")
    selection = selector.select(_query("attention"))
    assert selection.kernel_id == "eager.attention"
    assert selection.params == {}


def test_rule_can_still_force_the_auto_row():
    catalog = build_catalog()
    policy = policy_from_mapping(
        {
            "schema": 1,
            "rules": [
                {
                    "id": "force-auto",
                    "priority": 20,
                    "match": {"op": "attention_paged"},
                    "restrict_to": "flashinfer.attention_paged",
                }
            ],
        },
        catalog,
        source="test",
    )
    selector = Selector(catalog, policy, device="nvidia:SM90")
    assert (
        selector.select(_query("attention_paged")).kernel_id
        == "flashinfer.attention_paged"
    )


# --------------------------------------------------------------------------- #
# optional pi0.5 policy example                                               #
# --------------------------------------------------------------------------- #

#: Models do not load this file automatically. A user may select it through
#: PHYAI_KERNEL_CONFIG; otherwise the catalog priority order is the fallback.
PI05_EXAMPLE_POLICY = (
    Path(__file__).parents[3] / "examples" / "pi05" / "kernel_policy.yaml"
)


def _pi05_example_policy() -> Policy:
    from phyai.kernel.policy import load_policy

    return load_policy(PI05_EXAMPLE_POLICY, build_catalog())


def test_pi05_example_policy_parses():
    """A typo in the YAML must fail here, not silently lose 2.5x at runtime."""
    assert PI05_EXAMPLE_POLICY.exists()
    policy = _pi05_example_policy()
    assert [rule.rule_id for rule in policy.rules] == ["pi05-expert-joint-fa2"]
    rule = policy.rules[0]
    assert rule.restrict_to == "flashinfer.attention_paged.fa2"
    # Both plugin names, or the wn variant silently loses the pin.
    assert rule.source_match["model.family"] == ["pi05", "pi05_wn"]
    # The measurement covered the expert's joint attention only; the prefix
    # shares the op, so the role is what scopes the pin.
    assert rule.source_match["role"] == "expert"
    # The measurement was taken on sm90; the rule must not reach devices it
    # was never validated on.
    assert rule.source_match["device.arch"] == "sm90"


@pytest.mark.parametrize("family", ("pi05", "pi05_wn"))
def test_pi05_example_policy_pins_the_expert_and_nothing_else(family):
    from phyai.kernel.types import ModelContext

    selector = Selector(
        build_catalog(),
        _pi05_example_policy(),
        device="nvidia:SM90",
        model=ModelContext(family=family),
    )
    assert (
        selector.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged.fa2"
    )
    assert (
        selector.select(_query("attention_paged", role="prefix")).kernel_id
        == "flashinfer.attention_paged"
    )
    assert selector.select(_query("attention")).kernel_id == "flashinfer.attention"


def test_pi05_example_policy_leaves_other_models_alone():
    from phyai.kernel.types import ModelContext

    selector = Selector(
        build_catalog(),
        _pi05_example_policy(),
        device="nvidia:SM90",
        model=ModelContext(family="cosmos3"),
    )
    assert (
        selector.select(_query("attention_paged", role="expert")).kernel_id
        == "flashinfer.attention_paged"
    )


def test_pi05_example_policy_leaves_other_devices_alone():
    """The FA2 pin is an sm90 measurement; on Thor (sm110) or Blackwell the
    rule must not fire, and the expert keeps FlashInfer's own heuristic."""
    from phyai.kernel.types import ModelContext

    for arch in ("SM110", "SM100"):
        selector = Selector(
            build_catalog(),
            _pi05_example_policy(),
            device=f"nvidia:{arch}",
            model=ModelContext(family="pi05"),
        )
        assert (
            selector.select(_query("attention_paged", role="expert")).kernel_id
            == "flashinfer.attention_paged"
        )


def test_fa2_row_raises_the_workspace_floor():
    """The floor moved from the engine config onto the resolved kernel."""
    from phyai.layers.attention.utils import (
        PREFILL_WORKSPACE_FLOORS,
        resolve_workspace_bytes,
    )

    base = resolve_workspace_bytes()
    assert resolve_workspace_bytes(prefill_backend="fa2") == max(
        base, PREFILL_WORKSPACE_FLOORS["fa2"]
    )
    # Kernels without a floor are unaffected, which is the point of moving it:
    # a run that never resolves to FA2 no longer pays for FA2's scratch.
    assert resolve_workspace_bytes(prefill_backend="fa3") == base
    assert resolve_workspace_bytes(prefill_backend="auto") == base
    assert resolve_workspace_bytes() == base


def test_removed_env_var_names_the_rule_to_write():
    import os

    from phyai.engine_config import EngineConfig

    os.environ["PHYAI_FLASHINFER_PREFILL_BACKEND"] = "fa2"
    try:
        with pytest.raises(ValueError, match="restrict_to"):
            EngineConfig.from_env()
    finally:
        del os.environ["PHYAI_FLASHINFER_PREFILL_BACKEND"]
