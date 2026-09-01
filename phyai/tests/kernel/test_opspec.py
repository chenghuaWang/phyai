"""Unit tests for operation specifications and implementation rows.

Three properties matter here:

* An unknown fact path is a **loud** error with a suggestion. The old matcher
  returned ``False`` for an unrecognised key, which turned a typo into a rule
  that could never match while every test stayed green.
* A capability that only constrains optional facts is rejected, because
  omitting those facts would make the implementation unconditionally
  eligible.
* Parameter dtypes are *derived* from declared contracts, which is what lets a
  layer allocate gamma before any input tensor exists.
"""

from __future__ import annotations

import pytest

from phyai.kernel.facts import attrs, device, dtype, lib, quant, shape
from phyai.kernel.opspec import (
    Impl,
    OpSpec,
    ParamRule,
    Priority,
    any_float,
    fixed,
    matches_activation,
    resolve_param_dtypes,
    returns_callable,
    returns_instance,
)
from phyai.kernel.predicate import all_of, implies


GEMM = OpSpec(
    name="gemm",
    dims=("M", "N", "K"),
    dtypes=("input", "output"),
    optional_dtypes=("weight",),
    attributes=(),
    signature="(layer, x, bias) -> Tensor",
)

LAYERNORM = OpSpec(
    name="layernorm",
    dims=("tokens", "hidden"),
    dtypes=("input", "weight"),
    optional_dtypes=("bias",),
    attributes=("bias",),
    params=("weight", "bias"),
    signature="(x, weight, bias, eps) -> Tensor",
)


def impl(when, **kwargs) -> Impl:
    defaults = {
        "kernel_id": "test.gemm",
        "op": "gemm",
        "when": when,
        "prepare": lambda facts: None,
    }
    return Impl(**{**defaults, **kwargs})


# --------------------------------------------------------------------------- #
# Schema declaration
# --------------------------------------------------------------------------- #


def test_known_paths_enumerates_the_declared_schema() -> None:
    assert GEMM.known_paths() == {
        "shape.M",
        "shape.N",
        "shape.K",
        "dtype.input",
        "dtype.output",
        "dtype.weight",
    }


def test_optional_paths_covers_dtypes_and_attributes() -> None:
    assert LAYERNORM.optional_paths == {"dtype.bias"}


def test_a_role_cannot_be_both_required_and_optional() -> None:
    with pytest.raises(ValueError, match="both required and optional"):
        OpSpec(name="bad", dtypes=("input",), optional_dtypes=("input",))


def test_op_name_is_normalized_and_non_empty() -> None:
    assert OpSpec(name="  GEMM ").name == "gemm"
    with pytest.raises(ValueError, match="non-empty"):
        OpSpec(name="   ")


# --------------------------------------------------------------------------- #
# Path validation: the silent-False replacement
# --------------------------------------------------------------------------- #


def test_unknown_dimension_is_rejected_with_a_suggestion() -> None:
    with pytest.raises(ValueError, match=r"did you mean 'shape\.K'"):
        GEMM.validate_paths({"shape.KK"}, context="test")


def test_case_mismatch_is_caught_and_reported_as_case() -> None:
    """``shape.k`` is not an alias for ``shape.K``; it is a typo.

    Answered by an exact case-insensitive lookup rather than by ``difflib``,
    which scores ``shape.N``/``shape.M``/``shape.K`` identically for this
    input and would break the tie arbitrarily.
    """

    with pytest.raises(ValueError, match="case-sensitive") as excinfo:
        GEMM.validate_paths({"shape.k"}, context="test")
    assert "did you mean 'shape.K'" in str(excinfo.value)


def test_unknown_dtype_role_is_rejected() -> None:
    with pytest.raises(ValueError, match="has no fact 'dtype.residual'"):
        GEMM.validate_paths({"dtype.residual"}, context="test")


def test_global_facts_are_not_validated_against_the_op_schema() -> None:
    """``device.*`` / ``quant.*`` / ``lib.*`` are declared centrally."""

    GEMM.validate_paths(
        {"device.arch", "quant.format", "lib.flashinfer", "op", "mode"},
        context="test",
    )


def test_registering_an_impl_validates_its_capability_paths() -> None:
    good = impl(all_of(device.arch.at_least("sm90"), shape.K % 16 == 0))
    good.check_against(GEMM)

    bad = impl(all_of(device.arch.at_least("sm90"), shape.KK % 16 == 0))
    with pytest.raises(ValueError, match=r"capability of 'test\.gemm'"):
        bad.check_against(GEMM)


def test_impl_op_must_match_the_spec_it_registers_under() -> None:
    row = impl(shape.K % 16 == 0, op="layernorm")
    with pytest.raises(ValueError, match="registered under 'gemm'"):
        row.check_against(GEMM)


def test_impl_cannot_constrain_an_undeclared_parameter() -> None:
    row = Impl(
        kernel_id="test.layernorm",
        op="layernorm",
        when=dtype.input == "bf16",
        prepare=lambda facts: None,
        params={"gamma": fixed("fp32")},
    )
    with pytest.raises(ValueError, match="has no parameter"):
        row.check_against(LAYERNORM)


# --------------------------------------------------------------------------- #
# The vacuity lint
# --------------------------------------------------------------------------- #


def test_capability_constraining_only_optional_facts_is_rejected() -> None:
    """Otherwise omitting those facts makes the row always eligible."""

    row = impl(dtype.weight == "bf16")  # dtype.weight is optional for gemm
    with pytest.raises(ValueError, match="only optional facts"):
        row.check_against(GEMM)


def test_mixing_a_required_fact_makes_the_capability_valid() -> None:
    row = impl(all_of(dtype.input == "bf16", dtype.weight == "bf16"))
    row.check_against(GEMM)


def test_capability_using_only_global_facts_is_allowed() -> None:
    """A device-only gate is legitimate — nothing is being skipped."""

    impl(all_of(device.vendor == "nvidia", lib.has("flashinfer"))).check_against(GEMM)


# --------------------------------------------------------------------------- #
# Impl metadata
# --------------------------------------------------------------------------- #


def test_libraries_are_derived_from_the_capability() -> None:
    """So the selector imports only what some kernel's eligibility needs."""

    row = impl(all_of(lib.has("flashinfer"), device.arch.at_least("sm90")))
    assert row.libraries == {"flashinfer"}
    assert impl(device.arch.at_least("sm90")).libraries == frozenset()


def test_kernel_id_is_normalized() -> None:
    assert impl(
        device.arch.at_least("sm90"), kernel_id="  FlashInfer.Gemm  "
    ).kernel_id == ("flashinfer.gemm")
    with pytest.raises(ValueError, match="non-empty"):
        impl(device.arch.at_least("sm90"), kernel_id="  ")


# --------------------------------------------------------------------------- #
# Return conventions
# --------------------------------------------------------------------------- #


def test_callable_is_the_default_convention() -> None:
    assert returns_callable().kind == "callable"
    assert not returns_callable().is_instance


def test_instance_convention_records_its_construction_arguments() -> None:
    """Paged attention backends own runner-scoped buffers.

    They cannot be built at selection time, so the operation declares that its
    result *is* an object constructed with the runner — rather than this being
    a fourth ad-hoc calling convention.
    """

    class ARBackend:
        pass

    returns = returns_instance(ARBackend, constructed_with=("runner",))
    assert returns.is_instance
    assert returns.protocol is ARBackend
    assert returns.constructed_with == ("runner",)


def test_unknown_return_kind_is_rejected() -> None:
    from phyai.kernel.opspec import Returns

    with pytest.raises(ValueError, match="unknown Returns kind"):
        Returns("magic")


# --------------------------------------------------------------------------- #
# Parameter contracts
# --------------------------------------------------------------------------- #


def test_param_contract_shapes_are_validated() -> None:
    with pytest.raises(ValueError, match="requires a dtype"):
        from phyai.kernel.opspec import ParamContract

        ParamContract(ParamRule.FIXED)
    with pytest.raises(ValueError, match="must not name a dtype"):
        from phyai.kernel.opspec import ParamContract

        ParamContract(ParamRule.ANY_FLOAT, "fp32")


def test_layernorm_derives_fp32_gamma() -> None:
    """The value matches today's hardcoded choice, but is now derived.

    Previously the layer wrote ``float32 if backend == "flashinfer" or a
    resolver exists``. Adding a bf16-gamma kernel later changes this
    derivation instead of requiring that string test to be edited.
    """

    impls = [
        impl(
            dtype.input == "bf16",
            kernel_id="flashinfer.layernorm",
            op="layernorm",
            priority=Priority.OPTIMIZED + 2,
            params={"weight": fixed("fp32"), "bias": fixed("fp32")},
        ),
        impl(
            dtype.input == "bf16",
            kernel_id="phyai_kernel.layernorm",
            op="layernorm",
            priority=Priority.OPTIMIZED,
            params={"weight": any_float(), "bias": any_float()},
        ),
        impl(
            dtype.input == "bf16",
            kernel_id="torch.layernorm",
            op="layernorm",
            priority=Priority.REFERENCE,
            params={"weight": matches_activation(), "bias": matches_activation()},
        ),
    ]
    assert resolve_param_dtypes(LAYERNORM, impls, activation="bf16") == {
        "weight": "fp32",
        "bias": "fp32",
    }


def test_rmsnorm_derives_activation_dtype_params() -> None:
    """FlashInfer RMSNorm reads gamma through the input type.

    That was documented only in prose, as a footgun warning. Here it is a
    contract that produces the right allocation by construction.

    Note this is the case that rules out "keep *some* implementation
    eligible": fp32 gamma would satisfy the permissive torch reference while
    disqualifying the fast kernel, quietly choosing the slow path.
    """

    spec = OpSpec(name="rmsnorm", dtypes=("input", "weight"), params=("weight",))
    impls = [
        impl(
            dtype.input == "bf16",
            kernel_id="flashinfer.rmsnorm",
            op="rmsnorm",
            priority=Priority.OPTIMIZED + 2,
            params={"weight": matches_activation()},
        ),
        impl(
            dtype.input == "bf16",
            kernel_id="torch.rmsnorm",
            op="rmsnorm",
            priority=Priority.REFERENCE,
            params={"weight": any_float()},
        ),
    ]
    assert resolve_param_dtypes(spec, impls, activation="bf16") == {"weight": "bf16"}
    # With an fp32 activation the fast kernel is out of reach either way, and
    # the reference contract decides.
    assert resolve_param_dtypes(spec, impls, activation="fp32") == {"weight": "fp32"}


def test_preferred_dtype_wins_when_it_keeps_an_impl_eligible() -> None:
    spec = OpSpec(name="rmsnorm", dtypes=("input",), params=("weight",))
    impls = [
        impl(
            dtype.input == "bf16",
            kernel_id="torch.rmsnorm",
            op="rmsnorm",
            params={"weight": any_float()},
        )
    ]
    chosen = resolve_param_dtypes(
        spec, impls, activation="bf16", preferred={"weight": "bf16"}
    )
    assert chosen == {"weight": "bf16"}


def test_unsatisfiable_parameter_constraints_fail_at_construction() -> None:
    """Better to learn at construction than at the first forward."""

    spec = OpSpec(name="norm", dtypes=("input",), params=("weight",))
    impls = [
        impl(
            dtype.input == "bf16",
            kernel_id="only.norm",
            op="norm",
            params={"weight": fixed("fp8_e4m3")},
        )
    ]
    with pytest.raises(ValueError, match="no dtype satisfies parameter 'weight'"):
        resolve_param_dtypes(spec, impls, activation="bf16")


def test_an_unconstrained_parameter_follows_the_activation_dtype() -> None:
    """No contract means no reason to deviate from the activation dtype.

    fp32 affine parameters in a bf16 model cost memory and add per-forward
    casts, so "nothing constrains it" must not silently mean fp32.
    """

    spec = OpSpec(name="norm", dtypes=("input",), params=("weight",))
    impls = [impl(dtype.input == "bf16", kernel_id="lax.norm", op="norm")]
    assert resolve_param_dtypes(spec, impls, activation="bf16") == {"weight": "bf16"}
    assert resolve_param_dtypes(spec, impls, activation="fp32") == {"weight": "fp32"}


# --------------------------------------------------------------------------- #
# Reference-implementation requirement
# --------------------------------------------------------------------------- #


def test_reference_requirement_defaults_on_and_is_opt_out() -> None:
    """The paged attention ops genuinely have no CPU candidate.

    Writing that down means a fourth operation cannot silently join the set of
    ops that cannot run on a CPU host.
    """

    assert OpSpec(name="gemm").requires_reference
    assert not OpSpec(
        name="attention_paged", requires_reference=False
    ).requires_reference


def test_capture_safe_defaults_true_and_is_recorded_per_row() -> None:
    assert impl(device.arch.at_least("sm90")).capture_safe
    assert not impl(device.arch.at_least("sm90"), capture_safe=False).capture_safe


def test_conditional_bias_capability_passes_schema_validation() -> None:
    """The real FlashInfer LayerNorm contract, end to end."""

    row = Impl(
        kernel_id="flashinfer.layernorm",
        op="layernorm",
        when=all_of(
            lib.has("flashinfer"),
            device.vendor == "nvidia",
            dtype.input == "bf16",
            dtype.weight == "fp32",
            implies(attrs.bias, dtype.bias == "fp32"),
        ),
        prepare=lambda facts: None,
        params={"weight": fixed("fp32"), "bias": fixed("fp32")},
    )
    row.check_against(LAYERNORM)
    assert row.libraries == {"flashinfer"}


def test_quant_field_escape_hatch_is_not_op_scoped() -> None:
    """``quant.fields.*`` is global, so no op needs to declare it."""

    impl(all_of(quant.format == "nvfp4", quant.field("mode") == "x")).check_against(
        GEMM
    )
