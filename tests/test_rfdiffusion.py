"""Tests for RFDiffusion model."""

import pytest
import torch

from rfdiffusion.model import (
    DenoisingNetwork,
    R3Diffuser,
    RFDiffusion,
    RigidTransform,
    SE3Diffusion,
    SinusoidalTimestepEmbedding,
    SO3Diffuser,
    sample_igso3,
)


@pytest.fixture
def model():
    return RFDiffusion(
        node_dim=32,
        pair_dim=16,
        num_blocks=1,
        n_heads=4,
        n_qk_points=2,
        n_v_points=2,
        num_timesteps=10,
        self_conditioning=True,
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


def test_igso3_valid_rotation():
    L = 10
    rots = sample_igso3((L,), sigma=0.5)
    assert rots.shape == (L, 3, 3)
    RtR = torch.einsum("...ij,...ik->...jk", rots, rots)
    eye = torch.eye(3).expand(L, 3, 3)
    assert torch.allclose(RtR, eye, atol=1e-5)
    dets = torch.linalg.det(rots)
    assert torch.allclose(dets, torch.ones(L), atol=1e-5)


def test_so3_forward_preserves_orthogonality():
    L = 10
    so3 = SO3Diffuser(sigma_max=1.5)
    rots_0 = torch.eye(3).expand(L, 3, 3).clone()
    t = torch.full((L,), 0.5)
    rots_t, _ = so3.forward_marginal(rots_0, t)
    RtR = torch.einsum("...ij,...ik->...jk", rots_t, rots_t)
    assert torch.allclose(RtR, torch.eye(3).expand(L, 3, 3), atol=1e-4)


def test_r3_forward_statistics():
    r3 = R3Diffuser()
    trans_0 = torch.zeros(1000, 3)
    t_0 = torch.full((1000,), 0.01)
    trans_t, _ = r3.forward_marginal(trans_0, t_0)
    assert trans_t.std() < 0.5
    t_1 = torch.full((1000,), 0.99)
    trans_t, _ = r3.forward_marginal(trans_0, t_1)
    assert trans_t.std() > 0.5


def test_r3_alpha_bar_range():
    r3 = R3Diffuser()
    ts = torch.linspace(0, 1, 100)
    ab = r3.alpha_bar(ts)
    assert (ab >= 0).all()
    assert (ab <= 1.01).all()
    assert ab[0] > ab[-1]


def test_se3_round_trip():
    L = 10
    se3 = SE3Diffusion()
    frames = RigidTransform.identity((L,))
    noisy, _ = se3.forward_marginal(frames, torch.full((L,), 0.5))
    assert noisy.rots.shape == (L, 3, 3)
    assert noisy.trans.shape == (L, 3)
    assert torch.isfinite(noisy.rots).all()
    assert torch.isfinite(noisy.trans).all()


def test_timestep_embedding():
    embed = SinusoidalTimestepEmbedding(64)
    t = torch.tensor([0.0, 0.5, 1.0])
    out = embed(t)
    assert out.shape == (3, 64)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out[0], out[1])


def test_denoising_network_shape():
    net = DenoisingNetwork(
        node_dim=32,
        pair_dim=16,
        num_blocks=1,
        n_heads=4,
        n_qk_points=2,
        n_v_points=2,
        self_conditioning=True,
    )
    B, L = 2, 10
    frames = RigidTransform.identity((B, L))
    pred = net(frames, torch.tensor([0.5, 0.3]))
    assert pred.rots.shape == (B, L, 3, 3)
    assert pred.trans.shape == (B, L, 3)
    assert torch.isfinite(pred.rots).all()
    assert torch.isfinite(pred.trans).all()


def test_full_model_forward(model, protein):
    model.eval()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    with torch.no_grad():
        outputs = model(batch)
    assert "loss" in outputs
    assert "trans_loss" in outputs
    assert "rot_loss" in outputs
    assert torch.isfinite(outputs["loss"])
    assert outputs["loss"].item() >= 0


def test_loss_is_finite(model, protein):
    model.train()
    batch = {k: v.unsqueeze(0) for k, v in protein.items()}
    outputs = model(batch)
    assert torch.isfinite(outputs["loss"])
    assert not torch.isnan(outputs["loss"])


def test_model_param_count():
    small = RFDiffusion(node_dim=32, pair_dim=16, num_blocks=1)
    assert 10_000 <= small.count_parameters() <= 1_000_000
