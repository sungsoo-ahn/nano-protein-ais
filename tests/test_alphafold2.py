"""Tests for AlphaFold2 model."""

import pytest
import torch

from alphafold2.model import (
    AlphaFold2,
    AttentionWithPairBias,
    DiffusionModule,
    PairformerBlock,
    RigidTransform,
    SE3Diffusion,
    TriangularAttention,
    TriangularMultiplicativeUpdate,
    backbone_from_frames,
    fape_loss,
    sample_igso3,
)


@pytest.fixture
def model():
    return AlphaFold2(
        c_s=64,
        c_z=16,
        c_atom=64,
        num_pairformer_blocks=1,
        pairformer_heads=4,
        num_denoise_blocks=1,
        denoise_heads=4,
        num_plddt_bins=10,
    )


@pytest.fixture
def protein():
    L = 15
    torch.manual_seed(42)
    coords_CA = torch.randn(L, 3) * 3
    coords_N = coords_CA + torch.randn(L, 3) * 0.3
    coords_C = coords_CA + torch.randn(L, 3) * 0.3
    coords_O = coords_C + torch.randn(L, 3) * 0.2
    return {
        "coords_N": coords_N,
        "coords_CA": coords_CA,
        "coords_C": coords_C,
        "coords_O": coords_O,
        "sequence": torch.randint(0, 20, (L,)),
        "mask": torch.ones(L, dtype=torch.bool),
    }


def test_triangular_multiplicative_update():
    B, L, c_z = 2, 10, 16
    for mode in ("outgoing", "incoming"):
        tri = TriangularMultiplicativeUpdate(c_z, c_z, mode=mode)
        out = tri(torch.randn(B, L, L, c_z))
        assert out.shape == (B, L, L, c_z)
        assert torch.isfinite(out).all()


def test_triangular_attention():
    B, L, c_z = 2, 10, 16
    for mode in ("starting", "ending"):
        tri = TriangularAttention(c_z, n_heads=4, mode=mode)
        out = tri(torch.randn(B, L, L, c_z))
        assert out.shape == (B, L, L, c_z)
        assert torch.isfinite(out).all()


def test_attention_with_pair_bias_shape():
    B, L, c_s, c_z = 2, 10, 64, 16
    attn = AttentionWithPairBias(c_s, c_z, n_heads=4)
    single = torch.randn(B, L, c_s)
    pair = torch.randn(B, L, L, c_z)
    out = attn(single, pair)
    assert out.shape == (B, L, c_s)
    assert torch.isfinite(out).all()


def test_pairformer_block_shape():
    B, L, c_s, c_z = 2, 10, 64, 16
    block = PairformerBlock(c_s, c_z, n_heads=4)
    single, pair = block(torch.randn(B, L, c_s), torch.randn(B, L, L, c_z))
    assert single.shape == (B, L, c_s)
    assert pair.shape == (B, L, L, c_z)
    assert torch.isfinite(single).all()
    assert torch.isfinite(pair).all()


def test_so3_igso3_valid_rotation():
    """Sampled rotation matrices should be valid SO(3): R^T R = I, det(R) = 1."""
    R = sample_igso3((5, 10), sigma=1.0)
    assert R.shape == (5, 10, 3, 3)
    eye = torch.eye(3).expand(5, 10, 3, 3)
    RtR = R.transpose(-1, -2) @ R
    assert torch.allclose(RtR, eye, atol=1e-5)
    det = torch.det(R)
    assert torch.allclose(det, torch.ones_like(det), atol=1e-5)


def test_se3_forward_marginal():
    """SE(3) forward marginal preserves frame structure."""
    B, L = 2, 10
    frames = RigidTransform.identity((B, L))
    diffusion = SE3Diffusion()
    t = torch.full((B, L), 0.5)
    noisy, noise = diffusion.forward_marginal(frames, t)
    # Rotations should still be valid SO(3)
    RtR = noisy.rots.transpose(-1, -2) @ noisy.rots
    eye = torch.eye(3).expand(B, L, 3, 3)
    assert torch.allclose(RtR, eye, atol=1e-5)
    assert torch.isfinite(noisy.trans).all()


def test_diffusion_module_shape():
    B, L, c_s, c_z, c_atom = 2, 10, 64, 16, 64
    module = DiffusionModule(c_s=c_s, c_z=c_z, c_atom=c_atom, n_blocks=1, n_heads=4)
    noisy_frames = RigidTransform.identity((B, L))
    t = torch.ones(B) * 0.5
    single = torch.randn(B, L, c_s)
    pair = torch.randn(B, L, L, c_z)
    pred_frames = module(noisy_frames, t, single, pair)
    assert pred_frames.rots.shape == (B, L, 3, 3)
    assert pred_frames.trans.shape == (B, L, 3)
    assert torch.isfinite(pred_frames.rots).all()
    assert torch.isfinite(pred_frames.trans).all()


def test_backbone_from_frames():
    """Ideal bond lengths from identity frames."""
    frames = RigidTransform.identity((1, 5))
    N, CA, C = backbone_from_frames(frames)
    # CA should be at origin (identity translation)
    assert torch.allclose(CA, torch.zeros(1, 5, 3), atol=1e-6)
    # Check CA-C distance matches ideal
    ca_c_dist = (C - CA).norm(dim=-1)
    assert torch.allclose(ca_c_dist, torch.tensor(1.523), atol=1e-3)
    # Check N-CA distance matches ideal
    n_ca_dist = (N - CA).norm(dim=-1)
    assert torch.allclose(n_ca_dist, torch.tensor(1.458), atol=1e-3)


def test_fape_loss_zero_for_perfect():
    """Same frames -> FAPE loss ~= 0."""
    B, L = 2, 10
    frames = RigidTransform.identity((B, L))
    ca = torch.randn(B, L, 3)
    loss = fape_loss(frames, frames, ca, ca)
    assert loss.item() < 1e-3  # epsilon from sqrt(1e-8) ≈ 0.0001


def test_full_model_forward(model, protein):
    model.eval()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    with torch.no_grad():
        outputs = model(batch)
    assert torch.isfinite(outputs["loss"])
    assert outputs["loss"].item() > 0


def test_model_param_count(model):
    assert 50_000 <= model.count_parameters() <= 5_000_000
    prod = AlphaFold2()
    assert 1_000_000 <= prod.count_parameters() <= 50_000_000


def test_loss_is_finite(model, protein):
    model.train()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    outputs = model(batch)
    assert torch.isfinite(outputs["loss"])
    assert not torch.isnan(outputs["loss"])
