"""Scaled L2 norm: original eager semantics, loading and kernel dispatch."""

import pytest
import torch
import torch.nn.functional as F

from phyai.layers import RMSNorm
from phyai.kernel.call import select
from phyai.kernel.bootstrap import initialize_kernel_system
from phyai.kernel.config import KernelConfig


def make_l2_norm(dim, channel_first=True, images=True, bias=False, **kwargs):
    kwargs.setdefault("device", "cpu")
    return RMSNorm(
        dim,
        variant="l2",
        norm_dim=1 if channel_first else -1,
        affine_shape=(dim, *((1, 1) if images else (1, 1, 1)))
        if channel_first
        else (dim,),
        bias=bias,
        **kwargs,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("layout", ["image", "video", "last"])
@pytest.mark.parametrize("bias", [False, True])
def test_original_formula_exact(device, dtype, layout, bias):
    torch.manual_seed(1140)
    first = layout != "last"
    shape = {"image": (2, 7, 4, 6), "video": (1, 7, 3, 4, 6), "last": (2, 4, 7)}[layout]
    layer = make_l2_norm(7, first, layout != "video", bias, device=device, dtype=dtype)
    with torch.no_grad():
        layer.gamma.normal_()
        if bias:
            layer.bias.normal_()
    # Slicing retains a noncontiguous spatial/channel stride.
    wide = list(shape)
    wide[-1] *= 2
    x = torch.randn(wide, device=device, dtype=dtype)[..., ::2]
    before = x.clone()
    for scale in (1.0, 1e-14, 0.0):
        value = x if scale == 1 else x * scale
        ref = (
            F.normalize(value, dim=1 if first else -1) * (7**0.5) * layer.gamma
            + layer.bias
        )
        actual = layer(value)
        assert actual.dtype == ref.dtype
        assert actual.shape == ref.shape
        assert torch.equal(actual, ref)
        assert torch.isfinite(actual).all()
    assert torch.equal(x, before)
    assert {s.kernel_id for s in layer._plain_call._memo.values()} == {
        "torch.rmsnorm_l2"
    }


@pytest.mark.parametrize("parameter_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_autocast_and_mixed_affine_dtype(parameter_dtype, autocast):
    layer = make_l2_norm(
        96, images=False, bias=True, device="cuda", dtype=parameter_dtype
    )
    x = torch.randn(1, 96, 3, 8, 8, device="cuda", dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        expected = F.normalize(x, dim=1) * (96**0.5) * layer.gamma + layer.bias
        actual = layer(x)
    assert actual.dtype == expected.dtype
    assert torch.equal(actual, expected)


def test_original_checkpoint_shapes_and_cpu_to_cuda_loading():
    layer = make_l2_norm(96, images=False, bias=True, prefix="encoder.norm")
    assert layer.gamma.device.type == "cpu"
    state = {"gamma": torch.randn(96, 1, 1, 1), "bias": torch.randn(96, 1, 1, 1)}
    layer.load_state_dict(state, strict=True)
    for key, param in layer.named_parameters():
        assert param.hf_keys == [(f"encoder.norm.{key}", None)]
        param.weight_loader(param, state[key], None)
    layer.cuda().bfloat16()
    x = torch.randn(1, 96, 2, 4, 4, device="cuda", dtype=torch.bfloat16)
    expected = (
        F.normalize(x, dim=1) * (96**0.5) * state["gamma"].cuda().bfloat16()
        + state["bias"].cuda().bfloat16()
    )
    assert torch.equal(layer(x), expected)
    assert set(make_l2_norm(96).state_dict()) == {"gamma"}


def test_kernel_uses_existing_rmsnorm_op_and_clamps_norm():
    handle = select(
        "rmsnorm",
        device="cpu",
        dtype={"input": torch.float32, "weight": torch.float32},
        shape={"hidden": 2, "tokens": 1},
        attrs={"variant": "l2", "norm_dim": -1, "bias": False},
    )
    assert handle.kernel_id == "torch.rmsnorm_l2"
    x = torch.tensor([[3e-14, 4e-14]])
    gamma = torch.ones(2)
    actual = handle.execute(x, gamma, 1e-12)
    torch.testing.assert_close(actual, x / 1e-12 * (2**0.5), rtol=0, atol=0)
    # A soft backend preference cannot select an incompatible implementation.
    layer = make_l2_norm(2, channel_first=False, backend="flashinfer")
    assert torch.equal(layer(x), actual)
    assert {h.kernel_id for h in layer._plain_call._memo.values()} == {
        "torch.rmsnorm_l2"
    }


def test_capture_replays_torch_kernel():
    layer = make_l2_norm(96, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, 96, 8, 8, device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(3):
            layer(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        actual = layer(x)
    x.fill_(0.25)
    graph.replay()
    assert torch.equal(actual, layer(x))


def test_yaml_override_selects_the_registered_kernel(tmp_path):
    policy = tmp_path / "norm.yaml"
    policy.write_text(
        "schema: phyai.kernel/v1\nprofile: static\noverrides:\n"
        "  - id: vae-norm\n    match: {op: rmsnorm, attrs.variant: l2}\n"
        "    use: torch.rmsnorm_l2\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(policy)))
    layer = make_l2_norm(7, channel_first=False, device="cpu")
    x = torch.randn(3, 7)
    assert torch.equal(
        layer(x), F.normalize(x, dim=-1) * (7**0.5) * layer.gamma + layer.bias
    )
    assert {s.kernel_id for s in layer._plain_call._memo.values()} == {
        "torch.rmsnorm_l2"
    }


def test_standard_and_gemma_keep_their_kernel_and_semantics():
    from phyai.layers import GemmaRMSNorm

    for cls in (RMSNorm, GemmaRMSNorm):
        layer = cls(7, device="cpu")
        x = torch.randn(3, 7)
        result = layer(x)
        weight = (
            1 + layer.weight.float() if cls is GemmaRMSNorm else layer.weight.float()
        )
        expected = (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
            * weight
        ).to(x.dtype)
        assert torch.equal(result, expected)
        assert {h.kernel_id for h in layer._plain_call._memo.values()} == {
            "torch.rmsnorm"
        }
        assert set(layer.state_dict()) == {"weight"}


def test_l2_rejects_fused_residual_and_invalid_axis():
    layer = make_l2_norm(7, channel_first=False)
    x = torch.randn(3, 7)
    with pytest.raises(ValueError, match="residual"):
        layer(x, x.clone())
    with pytest.raises(ValueError, match="dimension"):
        make_l2_norm(7)(torch.randn(7))


@pytest.mark.parametrize(
    "variant,kernel", [("l2", "torch.rmsnorm"), ("rms", "torch.rmsnorm_l2")]
)
def test_policy_cannot_force_incompatible_semantics(tmp_path, variant, kernel):
    from phyai.kernel.selector import NoKernelError

    policy = tmp_path / "incompatible.yaml"
    policy.write_text(
        "schema: phyai.kernel/v1\nprofile: static\noverrides:\n"
        "  - id: incompatible\n    match: {op: rmsnorm}\n"
        f"    use: {kernel}\n"
    )
    initialize_kernel_system(KernelConfig(config_path=str(policy)))
    layer = RMSNorm(7, variant=variant, device="cpu")
    with pytest.raises(NoKernelError, match="cannot handle"):
        layer(torch.randn(3, 7))


def test_gemma_rejects_conflicting_l2_variant():
    from phyai.layers import GemmaRMSNorm

    with pytest.raises(ValueError, match="subclass variant"):
        GemmaRMSNorm(7, variant="l2", device="cpu")
