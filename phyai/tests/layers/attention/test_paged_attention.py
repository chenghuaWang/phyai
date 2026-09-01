"""Tests for the unified paged-KV attention layer — ``PagedAttention``.

The paged stack is **flashinfer-only** (GPU): ``"flashinfer"`` is the only
backend in the subpackage. Layer construction validates the backend name
against the kernel catalog without instantiating the backend (so no
flashinfer import is triggered), which keeps these construction /
validation tests runnable without CUDA. Forward-numerical coverage lives
in the CUDA-gated ``test_flashinfer_paged.py``.
"""

from __future__ import annotations

import pytest

from phyai.layers.attention import (
    FlashInferPagedBackend,
    PagedAttention,
    PagedAttentionBackend,
)


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


def test_paged_attention_flashinfer_backend_constructs():
    """Construction resolves the flashinfer factory by name without
    instantiating it (no flashinfer import at construction time)."""
    attn = PagedAttention(
        num_heads=4,
        head_dim=8,
        layer_id=0,
        causal=True,
        num_kv_heads=4,
        backend="flashinfer",
    )
    assert attn.backend == "flashinfer"
    assert attn.num_heads == 4
    assert attn.num_kv_heads == 4
    assert attn.head_dim == 8
    assert attn.layer_id == 0
    assert attn.causal is True


def test_causality_is_a_layer_trait_not_a_subsystem():
    """The old design shipped two identical subsystems whose only
    executable difference was the ``causal`` default. The merged layer
    makes the flag explicit per call site instead."""
    prefix = PagedAttention(
        num_heads=4, head_dim=8, layer_id=0, causal=False, kernel_role="prefix"
    )
    decoder = PagedAttention(
        num_heads=4, head_dim=8, layer_id=0, causal=True, kernel_role="decoder"
    )
    assert prefix.causal is False and decoder.causal is True
    assert prefix.kernel_role == "prefix"
    assert decoder.kernel_role == "decoder"


def test_paged_attention_rejects_sdpa_backend():
    """SDPA cannot serve the paged space — only registered in the
    no-cache stack, so layer construction must reject it."""
    with pytest.raises(ValueError, match="unknown backend"):
        PagedAttention(num_heads=4, head_dim=8, layer_id=0, causal=True, backend="sdpa")


def test_paged_attention_rejects_eager_backend():
    """The paged stack is flashinfer-only — ``"eager"`` is registered
    only in the no-cache stack."""
    with pytest.raises(ValueError, match="unknown backend"):
        PagedAttention(
            num_heads=4, head_dim=8, layer_id=0, causal=True, backend="eager"
        )


def test_paged_attention_rejects_invalid_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        PagedAttention(
            num_heads=4,
            head_dim=8,
            layer_id=0,
            causal=True,
            backend="not-a-backend",
        )


def test_paged_attention_rejects_bad_gqa():
    with pytest.raises(ValueError, match="must be a positive multiple"):
        PagedAttention(
            num_heads=4,
            head_dim=8,
            layer_id=0,
            causal=True,
            num_kv_heads=3,
            backend="flashinfer",
        )


def test_paged_attention_rejects_negative_layer_id():
    with pytest.raises(ValueError, match="layer_id must be non-negative"):
        PagedAttention(
            num_heads=4, head_dim=8, layer_id=-1, causal=True, backend="flashinfer"
        )


def test_flashinfer_backend_implements_the_paged_contract():
    """Class relationship only — instantiation would import flashinfer."""
    assert issubclass(FlashInferPagedBackend, PagedAttentionBackend)
