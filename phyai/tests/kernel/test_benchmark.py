"""The default autotune benchmark: env-only wiring, synthesis, and scope.

The contract under test: ``profile: autotune`` works from configuration
alone. ``initialize_kernel_system`` injects ``default_benchmark`` when the
policy asks for tuning and no programmatic hook was given, the hook
synthesizes inputs for ops that declare ``bench_args`` (GEMM, the norms),
measures on the query's device, persists winners to ``autotune_cache``, and
declines the attention family instead of guessing.
"""

from __future__ import annotations

import json

import pytest

from phyai.kernel.config import KernelConfig
from phyai.kernel.registry import build_catalog
from phyai.kernel.selector import Selector
from phyai.kernel.benchmark import default_benchmark
from phyai.kernel.bootstrap import initialize_kernel_system, resolve_policy
from phyai.kernel.types import KernelQuery


def rmsnorm_query(tokens: int = 8) -> KernelQuery:
    return KernelQuery.build(
        "rmsnorm",
        role="bench",
        dtype={"input": "fp32", "weight": "fp32"},
        shape={"tokens": tokens, "hidden": 64},
        attrs={"variant": "rms"},
    )


def gemm_query() -> KernelQuery:
    return KernelQuery.build(
        "gemm",
        role="bench",
        dtype={"input": "bf16", "output": "bf16"},
        quant={"format": "bf16"},
        shape={"M": 8, "N": 64, "K": 64},
    )


def autotune_selector(tmp_path, **kw) -> Selector:
    catalog = build_catalog()
    config = KernelConfig(
        profile="autotune", autotune_cache=str(tmp_path / "tune.json")
    )
    return Selector(
        catalog,
        resolve_policy(config, catalog),
        device="cpu",
        benchmark=default_benchmark(catalog),
        autotune_cache=config.autotune_cache,
        **kw,
    )


def test_env_only_wiring_installs_the_default_hook(tmp_path):
    """PHYAI_KERNEL_PROFILE=autotune must tune without any code."""

    selector = initialize_kernel_system(
        KernelConfig(profile="autotune", autotune_cache=str(tmp_path / "t.json")),
        device="cpu",
    )
    assert selector.benchmark is not None
    selection = selector.select(rmsnorm_query())
    assert selection.kernel_id == "torch.rmsnorm"
    data = json.loads((tmp_path / "t.json").read_text())
    assert list(data.values()) == ["torch.rmsnorm"]


def test_static_profile_gets_no_default_hook():
    selector = initialize_kernel_system(KernelConfig(), device="cpu")
    assert selector.benchmark is None


def test_norm_and_gemm_are_synthesizable_on_cpu(tmp_path):
    selector = autotune_selector(tmp_path)
    assert selector.select(rmsnorm_query()).kernel_id == "torch.rmsnorm"
    assert selector.select(gemm_query()).kernel_id == "torch.gemm.bf16"
    cache = json.loads((tmp_path / "tune.json").read_text())
    assert sorted(cache.values()) == ["torch.gemm.bf16", "torch.rmsnorm"]


def test_explain_reports_the_tuned_winner_and_its_measurement(tmp_path):
    selector = autotune_selector(tmp_path)
    selector.select(rmsnorm_query())
    trace = selector.explain(rmsnorm_query())
    assert trace.autotuned
    assert trace.selected == "torch.rmsnorm"
    measured = {c.kernel_id: c.benchmark_ms for c in trace.candidates}
    assert measured["torch.rmsnorm"] is not None
    assert measured["torch.rmsnorm"] > 0


def test_explain_matches_select_before_any_tuning(tmp_path):
    """explain() must not report a cache select() could not have written."""

    selector = autotune_selector(tmp_path)
    trace = selector.explain(rmsnorm_query())
    assert not trace.autotuned
    # Without a benchmark hook the same profile never consults the cache.
    catalog = build_catalog()
    config = KernelConfig(profile="autotune")
    bare = Selector(catalog, resolve_policy(config, catalog), device="cpu")
    bare.select(rmsnorm_query())
    assert not bare.explain(rmsnorm_query()).autotuned


def test_attention_declines_instead_of_guessing(tmp_path):
    """No bench_args -> the hook raises and priority order stands."""

    catalog = build_catalog()
    assert catalog.op("attention_paged").bench_args is None

    selector = autotune_selector(tmp_path)
    query = KernelQuery.build(
        "attention",
        role="bench",
        dtype={"input": "bf16", "key": "bf16", "value": "bf16", "output": "bf16"},
        shape={"tokens": 8, "kv_tokens": 8, "heads": 2, "kv_heads": 2, "head_dim": 64},
        attrs={"layout": "padded", "causal": False},
    )
    selection = selector.select(query)
    hook = selector.benchmark
    with pytest.raises(NotImplementedError, match="bench_args"):
        hook(selection.impl, selection.facts, selection)
    # Nothing was measured, so nothing was cached.
    assert not (tmp_path / "tune.json").exists()


def test_quantized_gemm_declines(tmp_path):
    """A synthesized layer cannot fake scale layouts; those calls raise."""

    catalog = build_catalog()
    facts = {"quant.format": "fp8_e4m3", "shape.M": 8, "shape.N": 64, "shape.K": 64}
    from phyai.kernel.facts import Facts

    with pytest.raises(NotImplementedError, match="dense"):
        catalog.op("gemm").bench_args(Facts(values=facts), "cpu")
